# How to apply the LBM with MNAR missing data on the data of French parliament 2018.

**The use of at least one GPU is necessary to run the model on the French parliament dataset.**

The French parliament 2018 dataset is availble in folder *data_parliament*

## Installation

### Pytorch installation

The model is implemented with pytorch.
To install pytorch we refer the reader to the [Pytorch website](https://pytorch.org/get-started/locally/)

With conda:
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

With pip:
```bash
pip install torch torchvision
```
### Other requirements

With conda:
```bash
conda install numpy
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
conda install -c conda-forge argparse

```

With pip:
```bash
pip install numpy scipy matplotlib argparse
```


## Usage

To run the model on the dataset, use the script *run_on_dataset_parliament.py*:
```bash
python run_on_dataset_parliament.py
```
The default number of row classes is 3 and column classes is 5.



To run with a GPU use the argument *device* and specify the cuda index of desired gpu (often 0):
```bash
python run_on_dataset_parliament.py --device=0
```

To run with higher number of classes, use the arguments *nb_row_classes* and *nb_col_classes* as:
```bash
python run_on_dataset_parliament.py --nb_row_classes=3 --nb_col_classes=5
```

With higher number of classes, the memory of your GPU may overflow. In that case, you can use a second GPU with the argument *device2* (index cuda needs to be specify):

```bash
python run_on_dataset_parliament.py --device=0 --device2=1 --nb_row_classes=3 --nb_col_classes=8
```


The script can be keyboard interrupted  at any moment. In that case, the algorithm returns the MPs and texts classes and a plot of the voting matrix re-ordereded according to class memberships.

## License
[MIT]
