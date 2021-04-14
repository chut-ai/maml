import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from maml.meta.meta_test import meta_test
from maml.office31.data.task_generator import OfficeTask
from maml.office31.base.model import DenseNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_spt = 36
n_qry = 100
n_inner_loop = 20
task_bsize = 50

office = OfficeTask(n_qry, n_spt)
source = "amazon"
target = "dslr"

classifier = torch.load("./saved/classifier_visda")
# classifier = DenseNet()


num_features = classifier.layers[9].in_features
classifier.layers[9] = nn.Linear(num_features, 10)
classifier.to(device)

acc = meta_test(office, source, target, classifier, n_inner_loop, task_bsize, device)
print(acc)

