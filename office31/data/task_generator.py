import json
import numpy as np
import random
import torch

def open_json(domain):
    path = "/home/louishemadou/office31_encoded/{}.json".format(domain)
    with open(path, "r") as f:
        jsonfile = json.load(f)
    f.close()
    return jsonfile

class OfficeTask:
    def __init__(self, n_qry, n_spt):

        self.domains = ["webcam", "dslr", "amazon"]
        self.data = {domain:open_json(domain) for domain in self.domains}
        self.n_qry = n_qry
        self.n_spt = n_spt

    def task(self, src, tgt):

        src_data = self.data[src]
        tgt_data = self.data[tgt]
        
        spt_data = []
        labels = range(10)
        for label in labels:
            all_instances = src_data[str(label)]
            indexes = np.random.choice(len(all_instances), self.n_spt, False)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
                spt_data.append([instance, label])
        random.shuffle(spt_data)

        qry_data = []
        for label in labels:
            all_instances = tgt_data[str(label)]
            n_img = len(all_instances)
            if self.n_qry < n_img:
                indexes = np.random.choice(n_img, self.n_qry, False)
            else:
                indexes = range(n_img)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
                qry_data.append([instance, label])
        random.shuffle(qry_data)
        
        instances_spt = [elem[0] for elem in spt_data]
        labels_spt = [elem[1] for elem in spt_data]

        instances_qry = [elem[0] for elem in qry_data]
        labels_qry = [elem[1] for elem in qry_data]

        x_spt = torch.stack(instances_spt, 0)
        x_qry = torch.stack(instances_qry, 0)
        y_spt = torch.Tensor(labels_spt)
        y_qry = torch.Tensor(labels_qry)

        return x_spt, x_qry, y_spt, y_qry

    def test_task_batch(self, src, tgt, task_bsize):

        tasks = [self.task(src, tgt) for _ in range(task_bsize)]

        return tasks
