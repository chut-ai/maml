import os
import json


def make_classes_json(root):

    domain = "clipart_for_json"
    index_to_class = {}
    class_to_index = {}

    directory = "/".join([root, domain])
    for label_class in os.listdir(directory):
        class_directory = "/".join([directory, label_class])
        img = os.listdir(class_directory)[0]
        class_index = int(img[8:11])
        index_to_class[class_index] = label_class
        class_to_index[label_class] = class_index
        
    with open("index_to_class.json", "w") as f1:
        json.dump(index_to_class, f1)

    with open("class_to_index.json", "w") as f2:
        json.dump(class_to_index, f2)




make_classes_json("/home/louishemadou/VisDA")
