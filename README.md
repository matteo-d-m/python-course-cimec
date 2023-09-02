This repository contains the final project for the course:

# [Python for (open) Neuroscience](https://github.com/vigji/python-cimec)

taught by [Luigi Petrucco](https://vigji.github.io) at the Doctoral school in [Cognitive and Brain Sciences](https://www.unitn.it/drcimec/), Center for Mind/Brain Sciences (CIMeC), University of Trento. 

---

The project consists of a preprocessing pipeline for electroencephalography (EEG) data. The pipeline follows the steps described in the Materials & Methods section of the randomly chosen EEG study: 

> [Alatorre-Cruz, G. C., Downs, H., Hagood, D., Sorensen, S. T., Williams, D. K., & Larson-Prior, L. J. (2022). Effect of Obesity on Arithmetic Processing in Preteens With High and Low Math Skills: An Event-Related Potentials Study. Frontiers in Human Neuroscience, 16, 760234](https://www.frontiersin.org/articles/10.3389/fnhum.2022.760234/full)

BIDS-formatted data from this study are publicly available on OpenNeuro at [this URL](https://openneuro.org/datasets/ds004019/versions/1.0.0).

The project is structured across four files:
- `matteos_functions.py`: a custom Python module that contains all the functions used throughout the pipeline
- `master_script.py`: the pipeline itself. Calls the functions contained in `matteos_functions.py` in appropriate order
- `config.py`: a set of dictionaries that contain parameters used across the pipeline (e.g., cut-off frequencies for signal filters, epoch limits...)
- `clean_eeg_epochs.fif` contains cleaned and epoched EEG data

Dependencies:
- [MNE-Python](https://github.com/mne-tools/mne-python) for EEG signal processing functions 
- [openneuro-py](https://github.com/hoechenberger/openneuro-py) to download OpenNeuro data automatically

Some preprocessing steps may have a relatively high computational cost (especially independent component analysis - ICA). `config.py` contains an option that skips those steps by default, jumping straight to the computation and plotting of event-related potentials (ERPs) from the data contained in `clean_eeg_epochs.fif`. 

If you don't mind waiting for ICA to run, you can open `config.py` and set `done` to `0`. This will run the pipeline end-to-end (including the download of EEG data from OpenNeuro onto your machine).

In order to save computational resources, the code works on data from one subject only. This too can be changed from `config.py`.

Note that this project is **not** meant to reproduce the scientific results obtained by Alatorre-Cruz et al., nor to propose the best possible preprocessing pipeline for EEG data. The goal is merely to:
1. Explore the functionalities of MNE-Python by working with real, publicly available EEG data
2. Start learning how to develop robust and reusable preprocessing pipelines in Python
