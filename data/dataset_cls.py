import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader

import cv2
import os
import torch



tf = transforms.Compose([transforms.ToTensor()])

class MyDataset(Dataset):

    def __init__(self, dir):
        self.dataset = []
        self.labels = []
        f = open(dir,'r')
        for linecontent in f.readlines():
            # print(linecontent.strip().split()[2:-1])
            linecontent = linecontent.replace('[','').replace(']','')
            boxes = []
            for i,box in enumerate(linecontent.strip().split()[1:]):
                if i<3:
                    boxes.append(float(box.split(',')[0])/30)
                else:
                    boxes.append(float(box.split(',')[0]) )
            self.dataset.append(boxes)
            self.labels.append([int(linecontent.strip().split()[0])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        input_data = torch.FloatTensor(data)
        # img_data = img_data.view(-1,12)
        labels = torch.LongTensor(self.labels[index])


        return input_data,labels

if __name__ == '__main__':

    myDataset = MyDataset(r'labels.txt')
    dataloader = DataLoader(myDataset, batch_size=3, shuffle=True)
    for img, label in dataloader:
        print(img.shape)
        print(label.shape)
        exit()



