# MedMNIST

*MedMNIST*, a collection of 10 pre-processed medical open datasets, is standardized to perform classification tasks on lightweight 28 * 28 images, which requires no background knowledge. Covering the primary data modalities in medical image analysis, it is diverse on data scale (from 100 to 100,000) and tasks (binary/multi-class, ordinal regression and multi-label). MedMNIST could be used for educational purpose, rapid prototyping, multi-modal machine learning or AutoML in medical image analysis. Moreover, MedMNIST Classification Decathlon is designed to benchmark AutoML algorithms on all 10 datasets. The datasets, evaluation code and baseline methods for MedMNIST are publicly available at https://medmnist.github.io/.

![MedMNIST_Decathlon](MedMNIST_Decathlon.png)

More details, please refer to our paper:

**MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis**

Jiancheng Yang, Rui Shi, Bingbing Ni

[arXiv](https://arxiv.org/abs/2010.14925)

# Code Structure
* [`medmnist/`](medmnist/):
    * [`dataset.py`](medmnist/dataset.py): dataloaders of medmnist.
    * [`models.py`](medmnist/models.py): *ResNet-18* and *ResNet-50* models.
    * [`evaluator.py`](medmnist/evaluator.py): evaluate metrics.
    * [`environ.py`](medmnist/environ.py): roots.
* [`train.py`](train.py): the training script.

# Requirements
* Python 3 (Anaconda 3.6.3 specifically)
* PyTorch\==0.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2

Higher versions should also work (perhaps with minor modifications).


# Dataset

Our MedMNIST dataset is available on [Dropbox](https://www.dropbox.com/sh/upxrsyb5v8jxbso/AADOV0_6pC9Tb3cIACro1uUPa?dl=0).

The dataset contains ten subsets, and each subset (e.g., `pathmnist.npz`) is comprised of `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` and `test_labels`.


# LICENSE
The code is under Apache-2.0 License.