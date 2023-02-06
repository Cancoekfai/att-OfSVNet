# att-OfSVNet

Code to run Offline Signature Verification from our paper [Offline Signature Verification with Attention](https://cedar.buffalo.edu/NIJ/data/).

# Prepare data

This is used to split dataset to train/test partitions.

```shell
python prepare_data.py --dataset dataset
```

- `CEDAR`: 1320 images genuine signature images and 1320 forged signature images from the [CEDAR dataset](https://cedar.buffalo.edu/NIJ/data/) [[Citation](https://github.com/junyanz/CycleGAN/blob/master/datasets/bibtex/facades.tex)].
- `BHSig`: [BHSig dataset](https://www.cityscapes-dataset.com/) [[Citation](https://github.com/junyanz/CycleGAN/blob/master/datasets/bibtex/facades.tex)] contains two sub-datasets, BHSig-B and BHSig-H. The BHSig-B dataset has 2400 images genuine signature images and 3000 forged signature images. The BHSig-H dataset has 3840 images genuine signature images and 4800 forged signature images.

# Model training and evaluation

This is used to split dataset to train/test partitions.

```shell
python main.py
```

# Citation

If you use this code for your research, please cite our paper:

