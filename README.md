This repository contains the final project for the course:

# [Python for (open) Neuroscience](https://github.com/vigji/python-cimec)

taught by [Luigi Petrucco](https://github.com/vigji) at the Doctoral school in [Cognitive and Brain Sciences](https://www.unitn.it/drcimec/), Center for Mind/Brain Sciences (CIMeC), University of Trento. 

---

The project consists of a preprocessing pipeline for electroencephalography (EEG) data. The pipeline follows the steps described in the Materials & Methods section of a randomly chosen EEG paper, in the order they are found in the paper. This paper is: 

> Alatorre-Cruz, G. C., Downs, H., Hagood, D., Sorensen, S. T., Williams, D. K., & Larson-Prior, L. J. (2022). Effect of Obesity on Arithmetic Processing in Preteens With High and Low Math Skills: An Event-Related Potentials Study. Frontiers in Human Neuroscience, 16, 760234. [https://doi.org/10.3389/fnhum.2022.760234](https://doi.org/10.3389/fnhum.2022.760234)

BIDS-formatted data from this study are publicly available on OpenNeuro at [this HTTPS URL](https://openneuro.org/datasets/ds004019/versions/1.0.0).

In practice, the project consists of four files:
- `matteos_functions.py`: a custom Python module that contains all the functions used throughout the pipeline
- `master_script.py`: the pipeline itself. Calls the functions contained in `matteos_functions.py` in appropriate order
- `config.py`: a set of dictionaries that contain parameters used across the pipeline (e.g., cut-off frequencies for signal filters, epoch limits...)
- `clean_eeg_epochs.fif`: cleaned and epoched (i.e., already processed by me) EEG data

Dependencies:
- [MNE-Python](https://github.com/mne-tools/mne-python) for EEG signal processing functions 
- [openneuro-py](https://github.com/hoechenberger/openneuro-py) to download OpenNeuro data automatically

The pipeline skips most preprocessing steps by default, jumping straight to the computation and plotting of event-related potentials (ERPs) from the data contained in `clean_eeg_epochs.fif`. This is because some steps may have a relatively high computational cost (especially independent component analysis - ICA) that might make a full run unpractical or undesirable. 

To run the pipeline end-to-end, open `config.py` and set `done` to `0`. Note that this will:
- Download EEG data from OpenNeuro onto your machine
- Take some time to complete (~30 minutes)
- Require your manual intervention to select independent components and bad channels 

The pipeline can process data from an arbitrary number of subjects. However, its default behaviour is to work on one subject only (`sub-01`) for convenience. If you want to process the whole dataset or a larger part of it, you can open `config.py` and modify the `which_files` option.

Note that this project is **not** meant to reproduce the scientific results obtained by Alatorre-Cruz et al., nor to develop the best possible preprocessing pipeline for EEG data. The goal is merely to:
1. Explore the functionalities of MNE-Python by working with real, publicly available EEG data
2. Start learning how to develop robust and reusable preprocessing pipelines in Python
