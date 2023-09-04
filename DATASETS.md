
## Here we give Three different glass-like object segmentation datasets example, including Trans10k, GSD, and PMD.

## Trans10k

First of all, please download the dataset from [Trans10K website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).
Then put the data under 'PATH/TO/YOUR/DATASETS/Trans10k'. Data structure is shown below.
```
Trans10k/
├── test
│   ├── images
│   └── masks
├── test_easy
│   ├── images
│   └── masks
├── test_hard
│   ├── images
│   └── masks
├── train
│   ├── images
│   └── masks
└── validation
    ├── images
    └── masks
```
 Note that, we keep the same data organization form as [EBLNet](https://openaccess.thecvf.com/content/ICCV2021/html/He_Enhanced_Boundary_Learning_for_Glass-Like_Object_Segmentation_ICCV_2021_paper.html).


## GSD

First, please request and download the dataset from [GSD_website](https://jiaying.link/cvpr2021-gsd/). 
Then put the data under 'PATH/TO/YOUR/DATASETS/GSD'. Data structure is shown below.
```
GSD/
├── test
│   ├── images
│   └── masks
└── train
    ├── images
    └── masks
```

## PMD

First, please download the dataset from [PMD_website](https://jiaying.link/cvpr2020-pgd/).
Then put the data under 'PATH/TO/YOUR/DATASETS/PMD'. Data structure is shown below.
```
PMD/
├── test
│   ├── images
│   └── masks
└── train
    ├── images
    └── masks
```
After that, you can either change the `config.py` or do the soft link according to the default path in config.

For example, 

Suppose you soft line all your datasets at `./data`, then update the dataset path in `config.py`,
```
mkdir data
ln -s PATH/TO/YOUR/DATASETS/* PATH/TO/YOUR/CODE/data
# PMD Dataset Dir Location
__C.DATASET.PMD_DIR = './data/PMD'
# GSD Dataset Dir Location
__C.DATASET.GSD_DIR = './data/GSD'
# Trans10k Dataset Dir Location
__C.DATASET.TRANS10K_DIR = './data/Trans10k'
``` 
