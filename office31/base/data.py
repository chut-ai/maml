from torch.utils.data import Dataset
import json
import random
import torch

class EncodedVisdaDataset(Dataset):
    def __init__(self, domains, labels, n_instances=-1):
        self.labels = sorted(labels)
        self.domains = domains

        self.img_list = []

        for domain in domains:
            print(domain)
            path = "/home/louishemadou/VisDA_encoded/{}.json".format(domain)
            with open(path, "r") as f:
                dic = json.load(f)
            f.close()
            for key in dic.keys():
                if int(key) in labels:
                    for img in dic[key]:
                           self.img_list.append([img, int(key)])
        random.shuffle(self.img_list)
        if n_instances != -1:
            self.img_list = self.img_list[:n_instances]

    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img, label = self.img_list[idx]
        img = torch.Tensor(img)
        label = self.labels.index(label)

        return img, label
