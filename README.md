# RAiSE: simulation-based analytical model of AGN jets and lobes
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5875155.svg)](https://doi.org/10.5281/zenodo.5875155)

_Radio AGN in Semi-Analytic Environments_ (RAiSE) model for the expansion and evolution of the jets and lobes emanating from extraglactic supermassive black holes. The RAiSE _HD_ (hydrodynamics) version of this model adapts Lagrangian particles from a hydrodynamical simulation to the dynamics from the analytical theory, yielding a physically-based magnetic field structure on both large and local scales. This code release enables the user to generate radio-frequency and X-ray wavelength surface brigtness images of Fanaroff-Riley Type-II radio AGNs across their evolutionary history, including for the jet, active lobe and remnant lobe. Parallised code can be run to generate a catalogue of mock radio AGNs to, for example: run parameter inversions to measure the energetics of observed objects; or produce high-resolution animations for data visualisations. The code is written in Python 2/3 and has detailed documentation and worked examples available on GitHub (https://github.com/rossjturner/RAiSEHD).

## Installation

This package can either be installed using _pip_ or from a .zip file downloaded from the GitHub repository using the standard Python package _distutils_.

### Install using pip
The following command will install the latest version of the RAiSE HD code from the Python Packaging Index (PyPI):

```bash
pip install RAiSEHD
```

### Install from GitHub repository

Important: the particle files must be downloaded individually (i.e., not just cloning the reop) as these are stored with GitHub LFS, or use pip install.

The package can be downloaded from the GitHub repository at https://github.com/rossjturner/RAiSEHD, or cloned with _git_ using:

```bash
git clone https://github.com/rossjturner/RAiSEHD.git
```

The package is installed by running the following command as an administrative user:

```bash
python setup.py install
```

## Documentation and Examples

Full documentation of the functions included in the RAiSE HD package, in addition to worked examples, is included in [RAiSEHD_user.pdf](https://github.com/rossjturner/RAiSEHD/blob/main/RAiSEHD_user.pdf) on the GitHub repository. The worked examples are additionally included in the following Jupyter notebook: [RAiSEHD_example.ipynb](https://github.com/rossjturner/RAiSEHD/blob/main/RAiSEHD_example.ipynb).

## Contact

Ross Turner <<turner.rj@icloud.com>>
