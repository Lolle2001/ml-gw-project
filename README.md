## 🚀 Glitch Amelioration in Gravitational Waves

This repository contains files for a machine learning project on intermediate mass black hole (IMBH) detection using gravitational waves. It focusses on ameliorating glitches in gravitational wave signals originating from cosmic events with intermediate mass black holes.

## 📝 Project Folder Setup

* 📁 **data**: contains the data files used for this project.
* 📁 **main**: contains the scripts that we have written ourselves.
* 📁 **lib**: contains the scripts that we have from an external source.
* 📁 **info**: contains the documentation for this project.
* 📁 **model**: contains the generated data, including settings and the neural network weights.

## Project Script Setup

The model consist out of a framework (`modelframe.py`) which contains the general settings and training loop for the different neural networks that are trained (models.py). The models are ran in the scripts `main.ipynb`, `multiclass_network.ipynb` and `test_networks.ipynb`. Results and settings are stored and plotted.
