# MAML for domain adaptation

## Download data

Download zip files, one per domain :

- Real : http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
- Quickdraw : http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
- Painting : http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip
- Infograph : http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
- Clipart : http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip
- Sketch : http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip

Extract theses files to data/raw/

data/raw/ should look like this : 

data/raw
    │
    ├ real
    │  ├ aircraft-carrier
    │  ├ airplane
    │  ├ alarm_clock
    │  └ ...
    │
    ├ quickdraw
    │  └ ...
    │
    ├ painting
    │  └ ...
    │
    ├ infograph
    │  └ ...
    │
    ├ clipart
    │  └ ...
    │
    └ sketch
       └ ...
		
## Pre encode images

To avoid computing several times the ResNet18 embedding of a single image, execute data/encode.py to pre encode all images with ResNet18. These embeddings are stored in data/json/.

## Baselines

### Baseline 1 : random network and source only training

Train a randomly initialised classifier using only source domain images, then test model accuracy on target domain. Training is performed on 10-way classification problems.

Execute randomSO.py, specify source and target desired (real, quickdraw, painting, infograph, clipart, sketch). Example :

python3 randomSO.py --source real --target quickdraw

### Baseline 2 : pretrained network and source only training

First pretrain the classifier on a 200-way classification problem with both source and target images, using train classes. Then, fine-tune the pretrained classifier using only source domain images, on 10-way classification problems, with test classes.

python3 pretrainSO.py --source real --target quickdraw

### Baseline 3 : random network and DANN training

Train a randomly initialised classifier using DANN algorithm on 10-way classification problems.

python3 randomDANN.py --source real --target quickdraw

### Baseline 4 : pretrained model and DANN training

First pretrain the classifier on a 200-way classification problem with both source and target images, using train classes. Then, fine-tune the pretrained classifier using DANN algorithm, on 10-way classification problems, with test classes.

python3 pretrainDANN.py --source real --target quickdraw

## MAML

Meta learn a good pretraining with 10-way classification problems made with train classes then test the pretraining on 10-way classification problems made with test classes.

python3 MAML-2DOM.py --source real --target quickdraw

The script generates graphs of accuracy and loss, so you can keep track of the performances of the computed pretraining. These graphs are saved in figures/ .
