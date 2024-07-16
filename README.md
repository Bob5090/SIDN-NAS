# SIDN-NAS

## Installation and Basic Usages

**Installation**

Make sure you have **pip** available in your system, and simply run:

```pip install -r requirements.txt```

**Basic Flags**

- ```--exp```: your experimentation indicator. Needs to be unique for different runs.
- ```--iter```: define iteration time. Our BiX-Net requires iter=3.
- ```--train_data```: specify the path to load training set.
- ```--valid_data```: specify the path to load validation set.
- ```--valid_dataset```: validation dataset choose from [monuseg, tnbc].
- ```--lr```: define learning rate.
- ```--epochs```: define the number of epochs.
- ```--batch_size```: batch_size for training or searching.

## Dataset
Please download **MoNuSeg**, **TNBC** and **CHAOS** dataset to default path.

*NOTE: Default path to load data is ```./data```*


## Usage
```
example:
python ./CORE/main.py --Phase1 --exp EXP_1 --save_gene ./MRI_Results/MRI_Left_Kidney_iter1 --batch_size 8 --iter 1 --in_channel 1 --train_data CORE/CHAOS_MRI --valid_data CORE/CHAOS_MRI --data_name Left_Kidney --valid_dataset MRI --epochs 100 --lr 0.001
```

