# att-OfSVNet
Code to run Offline Signature Verification from our paper Offline Signature Verification with Attention.


# Installation
## Basic
- `Python` >= 3.6
## Modules
```shell
pip install -r requirements.txt
```

# Data Preparation
The samples were paired using the following script. Many of the datasets were collected by other researchers. Please cite their papers if you use the data.
```shell
python prepare_data.py --dataset dataset
```
- `CEDAR`: 1320 images genuine signature images and 1320 forged signature images from the [CEDAR dataset](https://cedar.buffalo.edu/NIJ/data/) [[Citation](https://github.com/Cancoekfai/att-OfSVNet/blob/main/datasets/bibtex/CEDAR.tex)].
- `BHSig260`: [BHSig260 dataset](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/edit?resourcekey=0-MUNnTzBi4h_VE0J84NDF3Q) [[Citation](https://github.com/Cancoekfai/att-OfSVNet/blob/main/datasets/bibtex/BHSig.tex)] contains two sub-datasets, BHSig-B and BHSig-H. The BHSig-B dataset has 2400 images genuine signature images and 3000 forged signature images. The BHSig-H dataset has 3840 images genuine signature images and 4800 forged signature images.


# Training & Evaluation
```shell
python main.py
```
