import os

try:
    from itertools import batched
except ImportError:

    def batched(iterable, n):
        """Batch data into tuples of length n. The last batch may be shorter."""
        iterator = iter(iterable)
        while True:
            batch = tuple(itertools.islice(iterator, n))
            if not batch:
                break
            yield batch


import itertools

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from nimare.correct import FDRCorrector
from nimare.dataset import Dataset
from nimare.extract import download_abstracts, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from transformers import AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_specter_model():
    """Load SPECTER model with adapters for proximity and ad-hoc query."""
    model = AutoAdapterModel.from_pretrained(
        "allenai/specter2_aug2023refresh_base"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/specter2_aug2023refresh_base"
    )
    query_adapter = model.load_adapter(
        "allenai/specter2_aug2023refresh_adhoc_query",
        source="hf",
        set_active=True,
    )
    proximity_adapter = model.load_adapter(
        "allenai/specter2_aug2023refresh", source="hf", set_active=True
    )
    model.set_active_adapters(proximity_adapter)

    return model, tokenizer, query_adapter, proximity_adapter


def query_dataset(
    query_text,
    model,
    tokenizer,
    embeddings,
    dset_table,
    adapter_name,
    top_k=5,
    device=device,
):
    """Query dataset with SPECTER model and return nearest neighbors.

    Parameters
    ----------
    query_text : str
        Text to query the dataset.
    model : AutoAdapterModel
        SPECTER model with adapters.
    tokenizer : AutoTokenizer
        Tokenizer for the SPECTER model.
    embeddings : torch.Tensor
        Embeddings of the dataset to query against.
    dset_table : pd.DataFrame
        DataFrame containing the dataset metadata.
    adapter_name : str or None
        Name of the adapter to use for the query. If None, use proximity adapter.
    top_k : int or None
        Number of top results to return. If None, return all results.
    device : str
        Device to run the model on. Default is 'cpu' or 'cuda'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the nearest neighbors and their distances.
    """

    old_adapters = None
    if adapter_name is None:
        # Assuming that if no adapter is selected, then proximity is set
        query = "Neuroimaging" + tokenizer.sep_token + query_text
    else:
        old_adapters = model.active_adapters[0]
        model.set_active_adapters(adapter_name)
        query = query_text

    query_input = tokenizer(
        [query],
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        query_output = model(
            input_ids=query_input["input_ids"].to(device),
            attention_mask=query_input["attention_mask"].to(device),
        )

    # take the first token in the batch as the embedding
    query_embeddings = query_output.last_hidden_state[:, 0, :]

    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # similarities = cos(query_embeddings, embeddings)
    distances = torch.from_numpy(
        euclidean_distances(embeddings.cpu(), query_embeddings.cpu()).flatten()
    )
    if top_k is not None:
        topk = torch.topk(distances, k=top_k, largest=False)
        res = dset_table.iloc[topk.indices.numpy()].copy()
        res["distance"] = topk.values.numpy()
    else:
        res = dset_table.copy()
        res["distance"] = distances
    if old_adapters is not None:
        model.set_active_adapters(old_adapters)
    return res.sort_values("distance", ascending=True)


def get_nsynth_dataset(db_ns_file):
    """Get or create Neurosynth dataset."""
    if not os.path.exists(db_ns_file):
        files = fetch_neurosynth(
            data_dir=nsynth_data_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
        neurosynth_db = files[0]

        neurosynth_dset = convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        neurosynth_dset = download_abstracts(
            neurosynth_dset, "example@example.edu"
        )
        neurosynth_dset.save(db_ns_file)
    else:
        neurosynth_dset = Dataset.load(db_ns_file)
    return neurosynth_dset


def run_metanalisys(neurosynth_dset, res_ns):
    """Run MKDA meta-analysis on Neurosynth dataset given selected studies.

    Parameters
    ----------
    neurosynth_dset : Dataset
        Neurosynth dataset.
    res_ns : pd.DataFrame
        DataFrame with selected studies.

    Returns
    -------
    map_cres : nibabel.Nifti1Image
        Resulting meta-analysis map.
    """
    decoding_results_negative = neurosynth_dset.slice(
        list(set(neurosynth_dset.ids) - set(res_ns.id.tolist()))
    )
    decoding_results = neurosynth_dset.slice(res_ns.id.tolist())
    mkda = MKDAChi2(kernel__r=10)
    results = mkda.fit(decoding_results, decoding_results_negative)
    corrector = FDRCorrector(method="indep", alpha=0.05)
    cres = corrector.transform(results)
    map_name = "z_desc-association_level-voxel_corr-FDR_method-indep"
    map_cres = cres.get_map(map_name)

    return map_cres


def main(nsynth_data_dir, verbose=True):
    """
    Main function to run the SPECTER model on the Neurosynth dataset and
    perform meta-analysis on the most relevant studies for each concept.
    Parameters
    ----------
    nsynth_data_dir : str
        Directory containing the Neurosynth dataset.
    verbose : bool
        If True, print additional information during processing.
    """
    # Load downloaded Neurosynth dataset
    db_ns_file = os.path.join(nsynth_data_dir, "neurosynth_dataset.pkl.gz")
    neurosynth_dset = get_nsynth_dataset(db_ns_file)

    # Create dataset table with titles and abstracts from Neurosynth dataset
    dset_ns_table = pd.merge(
        neurosynth_dset.metadata,
        neurosynth_dset.texts[["study_id", "abstract"]],
        left_on="study_id",
        right_on="study_id",
        how="inner",
    )

    # Get SPECTER model and tokenizer
    model, tokenizer, query_adapter, proximity_adapter = get_specter_model()
    model = model.to(device)
    model.eval()
    model.set_active_adapters(proximity_adapter)

    # Create a batch of combined title and abstracts from Neurosynth.
    text_ns_batch = (
        dset_ns_table.title + tokenizer.sep_token + dset_ns_table.abstract
    ).fillna("")

    # Tokenizer: get a dictionary-like object containing the tokenized and
    # preprocessed text data ready for model input.
    # The inputs object will contain these main elements:
    # -> input_ids: A PyTorch tensor of shape (batch_size, sequence_length)
    # containing the token IDs that represent the text
    # -> attention_mask: A PyTorch tensor of the same shape with 1s for real
    # tokens and 0s for padding tokens
    inputs = tokenizer(
        text_ns_batch.values.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )

    # Generate embeddings from tokenized inputs
    if hasattr(getattr(torch, device), "empty_cache"):
        getattr(torch, device).empty_cache()
    with torch.no_grad():
        n = 128
        embeddings_ns = []
        for input_ids, attention_mask in tqdm(
            zip(
                batched((inputs["input_ids"]), n),
                batched(inputs["attention_mask"], n),
            ),
            total=len(inputs["input_ids"]) // n + 1,
        ):
            input_ = {
                "input_ids": torch.stack(input_ids, axis=0).to(device),
                "attention_mask": torch.stack(attention_mask, axis=0).to(
                    device
                ),
            }
            output = model(**input_)
            # take the first token in the batch as the embedding
            embeddings_ns.append(output.last_hidden_state[:, 0, :].cpu())
            if hasattr(getattr(torch, device), "empty_cache"):
                getattr(torch, device).empty_cache()
    embeddings_ns_tensor = torch.cat(embeddings_ns, axis=0)
    embeddings_ns_tensor.shape

    # Test with CogAtlas concepts: get definitions as queries
    concepts_df = pd.read_csv("concept_definitions.tsv", sep="\t")

    # For every query, get nearest neighbors from Neurosynth dataset
    # and run meta-analysis
    for concept_name in concepts_df["Concept"]:
        print(f"Processing concept: {concept_name}")
        query_text = concepts_df.loc[
            concepts_df["Concept"] == concept_name, "Definition"
        ].values[0]

        # Semantic similarity search using the trained SPECTER model:
        # -> Query Processing: Converts the search text into an embedding using the
        # SPECTER model that created the dataset embeddings
        # -> Similarity Calculation: It computes distances between the query
        # embedding and all pre-computed dataset embeddings
        # Ranking & Results: Returns the closest matches ranked by similarity
        res_ns = query_dataset(
            query_text,
            model=model.to(device),
            tokenizer=tokenizer,
            embeddings=embeddings_ns_tensor,
            dset_table=dset_ns_table,
            adapter_name=query_adapter,
            top_k=None,
        )

        threshold_level = 5.0
        threshold = np.percentile(res_ns["distance"], threshold_level)

        # Select studies below the distance threshold
        res_ns = res_ns[res_ns["distance"] < threshold]
        if verbose:
            print(res_ns.shape)
            res_ns.head()

        # Run meta-analysis on selected studies
        map_cres = run_metanalisys(neurosynth_dset, res_ns)
        map_name = os.path.join(
            nsynth_data_dir,
            f"specter_nsynth_mkda_{concept_name.replace(' ', '_')}.nii.gz",
        )
        map_cres.to_filename(map_name)


if __name__ == "__main__":
    nsynth_data_dir = os.path.abspath("example_data/")
    main(nsynth_data_dir, verbose=True)
