
# Directed Hierarchical Graph for Zero-shot Learning on ImageNet

## Requirements

* python 3
* pytorch 0.4.0
* nltk

## Instructions

### Materials Preparation

There is a folder `materials/`, which contains some meta data and programs already.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `materials/`.

#### Build Word-Net Graphs
1. `cd materials/`
2. Run `python make_induced_graph.py`, get `imagenet-induced-graph.json`
3. Run `python make_dense_graph.py`, get `imagenet-dense-graph.json`  # 200082 edges
3. Run `python make_dense_grouped_graph.py`, get `imagenet-dense-grouped-graph.json`

#### Build NELL Graphs
1. `cd materials/`
2. find the construct section in 'construct_multi_weight_graph.ipynb'.
#### Obtain Pretrained ResNet101
 `cd materials/`, run `python process_resnet.py`, get `fc-weights.json` and `resnet101-base.pth`

#### ImageNet and AwA2

Download ImageNet by your own. and AwA2, create the softlinks (command `ln -s`): `materials/datasets/imagenet` and `materials/datasets/awa2`, to the root directory of the dataset.

An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

An AwA2 root directory should contain the folder JPEGImages.

### Training

Make a directory `save/` for saving models.

In most programs, use `--gpu` to specify the devices to run the code (default: use gpu 0).

#### Train Graph Networks
* DHG: Run `python train_gcn_att.py`, get results in `save/gcn-att`
* DHG-r: Run `python  train_gcn_att_r.py`, get results in `save/gcn-att-r`, in which edges are grouped by directions.

In the results folder:
* `*.pth` is the state dict of Graph Networks model
* `*.pred` is the prediction file, which can be loaded by `torch.load()`. It is a python dict, having two keys: `wnids` - the wordnet ids of the predicted classes, `pred` - the predicted fc weights

#### Finetune ResNet
Run `python train_resnet_fit.py` with the args:
* `--pred`: the `.pred` file for finetuning
* `--train-dir`: the directory contains 1K imagenet training classes, each class with a folder named by its wordnet id
* `--save-path`: the folder you want to save the result, e.g. `save/resnet-fit-xxx`

python train_resnet_fit.py --pred save/gcn-dense-att/epoch-3000.pred --train-dir materials/datasets/imagenet --save-path save/resnet-fit

(In the paper's setting, --train-dir is the folder composed of 1K classes from fall2011.tar, with the missing class "teddy bear" from ILSVRC2012.)

### Testing

#### ImageNet
Run `python evaluate_imagenet.py` with the args:
* `--cnn`: path to resnet101 weights, e.g. `materials/resnet101-base.pth` or `save/resnet-fit-xxx/x.pth`
* `--pred`: the `.pred` file for testing
* `--test-set`: load test set in `materials/imagenet-testsets.json`, choices: `[2-hops, 3-hops, all]`

python evaluate_imagenet.py --cnn materials/resnet101-base.pth --pred save/gcn-dense-att/epoch-3000.pred --test-set 2-hops

* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train classes images only.

#### AwA2
Run `python evaluate_awa2.py` with the args:
* `--cnn`: path to resnet50 weights, e.g. `materials/resnet101-base.pth` or `save/resnet-fit-xxx/x.pth`
* `--pred`: the `.pred` file for testing
* (optional) `--consider-trains` to include training classes' classifiers
