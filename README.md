# Concept-level brain mapping: Leveraging a cognitive ontology to map brain function from deep-phenotyping fMRI datasets

## Description
Code accompanying the **Concept-level brain mapping** article.

## Preliminar naming convention for files:

0(#) : numbered experiments

0(#)_vis : visualization scripts of the experiment, results or data

utils_* : utility scripts, common to different experiments

## Organization of experiments
The scripts are organized in the right order, so running one by one will ensure complete replication of all experiments in the manuscript.

01: Data preparation with matrix conditioning and maps reliability assessments.

02: Computing cognitive component maps using a binary matrix (occurrence of tags).

03: Weighting procedure: moving towards a weighted matrix to compute the cognitive maps.

04: Validation experiments: using the binary and weighted matrices to predict unseen conditions. Replication from [Walters and colleagues](https://github.com/waltersjonathon/cognitive_encoding_models).

05: Validation experiments using Nimare: compare cognitive component maps to meta-analytic maps from Neurosynth.

06: Validation experiments, predicting behavioral responses: experiments to relate behavioral responses (response time, category) to the cognitive maps.

## Replication

This code was developed and tested with Python 3.13.5. Clone this repository and install the dependencies:
```bash
python -m pip install -r requirements.txt
```

This code relies on the [IBC public analysis code](https://github.com/individual-brain-charting/public_analysis_code), which is included in `requirements.txt`. To verify your installation, try:
```python
from ibc_public.utils_data import data_parser
```
You will need to point the code to your local copy of the IBC dataset. Specifically, update the data directory in the `data_parser` function to match your local path.

## Data access
The IBC data can be accessed through their [fetcher tool](https://github.com/individual-brain-charting/api). Data are hosted by [Ebrains](https://search.kg.ebrains.eu/instances/131add71-e838-4dab-b953-7b7a69ac5d8f).
Check the data descriptors for details:

> Pinho, A., Amadon, A., Ruest, T. et al. Individual Brain Charting, a high-resolution fMRI dataset for cognitive mapping. Sci Data 5, 180105 (2018). https://doi.org/10.1038/sdata.2018.105

> Ponce, A.F., Aggarwal, H., Shankar, S. et al. Individual Brain Charting: fifth release of high-resolution fMRI data for cognitive mapping. Sci Data 13, 593 (2026). https://doi.org/10.1038/s41597-026-06869-1
