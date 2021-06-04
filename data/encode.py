from PIL import Image
import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.upsample = nn.Upsample((224, 224))
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):

        with torch.no_grad():
            x = self.upsample(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return x


resnet = ResNet().cuda()
resnet.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, (1, 1), (1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def encode(domain):
    cls_to_int = {}
    id_ = 0
    for cls in os.listdir("./raw/{}/".format(domain)):
        cls_to_int[cls] = id_
        id_ += 1

    encoded_dict = {id_: [] for id_ in cls_to_int.values()}
    cls_list = os.listdir("./raw/{}/".format(domain))
    for i, cls in enumerate(cls_list):
        message = "Encoding {} data, {:.1f}%".format(
            domain, 100*(i+1)/len(cls_list))
        print(message, end="\r")
        cls_int = cls_to_int[cls]
        for img_name in os.listdir("./raw/{}/{}/".format(domain, cls)):
            path = "./raw/{}/{}/{}".format(domain, cls, img_name)
            img = Image.open(path)
            img = transform(img).unsqueeze(0).cuda()
            encoded_img = resnet(img).squeeze().cpu().tolist()
            encoded_dict[cls_int].append(encoded_img)
    print("")
    print("Saving {} data...".format(domain))

    json_name = "./json/{}.json".format(domain)
    f = open(json_name, "w")
    json.dump(encoded_dict, f)
    f.close()

encode("infograph")
encode("real")
encode("quickdraw")
encode("painting")
encode("sketch")
encode("clipart")
