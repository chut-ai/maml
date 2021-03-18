import os
import json

root = "/home/louishemadou/VisDA/quickdraw"

with open("./class_to_index.json") as f:
    class_to_index = json.load(f)


for class_label in os.listdir(root):
    class_index = class_to_index[class_label]
    str_class_index = str(class_index).zfill(3)
    index = 1
    directory = "/".join([root, class_label])
    for img in os.listdir(directory):
        str_index = str(index).zfill(6)
        src = "/".join([directory, img])
        dst = "{}/quickdraw_{}_{}.jpg".format(directory,
                                              str_class_index, str_index)
        os.rename(src, dst)
        index += 1
