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

- raw
	- real	
		- aircraft-carrier
		- airplane
		- alarm_clock
		- ...
	- quickdraw
		- ...
	- painting
		- ...
	- infograph
		- ...
	- clipart
		- ...
	- sketch
		- ...
## Pre encode images

To avoid computing several times the ResNet18 embedding of a single image, execute data/encode.py to pre encode all images with ResNet18. These embeddings are stored in data/json/.

## Baselines

### Baseline 1 : random network and source only training

Train a randomly initialised model using only source domain images, then test model accuracy on target domain. Training is performed on 10 way classification problems.

Execute randomSO.py, specify source and target desired. Example :

python3 randomSO.py --source real --target quickdraw

### Baseline 2 : pretrained network and source only training

First pretrain the model on a 200 way classification problem with both source and target images, using train classes. Then, train the pretrained model using only source domain images, on 10 way classification problems, with test classes.

python3 pretrainSO.py --source real --target quickdraw

### Baseline 3 : random network and DANN training

Train a randomly initialised model using DANN algorithm on 10 way classification problems.

python3 randomDANN.py --source real --target quickdraw

### Baseline 4 : pretrained model and DANN training

First pretrain the model on a 200 way classification problem with both source and target images, using train classes. Then, train the pretrained model using DANN algorithm, on 10 way classification problems, with test classes.

python3 pretrainDANN.py --source real --target quickdraw

## MAML

Meta learn a good pretraining with tasks made with train classes, and test the pretraining on tasks made with test classes.

python3 MAML-2DOM.py --source real --target quickdraw

You can keep track of the performances of the computed pretraining task after task in figures/
