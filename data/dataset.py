import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader
from data.autoaugment import ImageNetPolicy

import os,json
import torch

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


tf = transforms.Compose([transforms.ToTensor()])



class MyDataset(Dataset):

    def __init__(self, dir,istrain=True):
        self.dataset = []
        self.labels = []
        self.train = istrain

        f = open(dir,'r',encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            self.dataset.append(line)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        size = 416
        data = self.dataset[index]
        image = Image.open(data).convert('RGB')
        jsonpath = data.replace('.jpg','.json').replace('pics','label')
        with open(jsonpath,'r') as jsf:
            points = json.load(jsf)
        names = ['C2_c','C3_c','C4_c'] #ROI point
        # names = ['C2p','C2d','C2a'] #C2 landmarks
        # names = ['C3up','C3um','C3ua','C3pm','C3am','C3lp','C3ld','C3la'] #C3 landmarks
        # names = ['C4up','C4um','C4ua','C4pm','C4am','C4lp','C4ld','C4la'] #C4 landmarks
        all_point = []
        for name in names:
            all_point.extend(points[name])


        img = Image.new('RGB',(max(image.size),max(image.size)),(255,255,255))
        img.paste(image,(0,0))
        LHC = np.array(all_point) / max(image.size)
        image = img.resize((size, size))
        policy = ImageNetPolicy()
        if self.train:
            image = policy(image)
        img_data = tf(image)
        points = torch.FloatTensor(LHC)

        return img_data,points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # img = Image.open(r"E:\口内照上颚\upper_image\0/3717.jpg").resize((224,224))
    # print("原图大小：", img.size)
    # data1 = transforms.RandomResizedCrop(224)(img)
    # print("随机裁剪后的大小:", data1.size)
    # data2 = transforms.RandomResizedCrop(224)(img)
    # data3 = transforms.RandomResizedCrop(224)(img)
    #
    # plt.subplot(2, 2, 1), plt.imshow(img), plt.title("原图")
    # plt.subplot(2, 2, 2), plt.imshow(data1), plt.title("转换后的图1")
    # plt.subplot(2, 2, 3), plt.imshow(data2), plt.title("转换后的图2")
    # plt.subplot(2, 2, 4), plt.imshow(data3), plt.title("转换后的图3")
    # plt.show()
    # get_random_data('Q:\Img_yuan\img1/0.jpg 945,856,1027,895,0 317,937,388,978,0 735,918,852,992,0',(416,416))
    myDataset = MyDataset(r'D:\projects\px_analyze\px_coccidium\train_data\test_ori_L.txt')
    # myDataset = MyDataset(r"U:\person_car\UA-DETRAC/label.txt")
    dataloader = DataLoader(myDataset, batch_size=1, shuffle=True)
    for img,points, label in dataloader:
        print(img.shape)
        print(points.shape)
        exit()


        for i, (image,points, label) in enumerate(zip(sample[0], sample[1])):
            image = image.numpy()
            h, w = image.shape[:2]
            label = label.view(-1,1)
            print(points.shape)
            print(label)

        break

    # dir = r'Q:\data_448'
    # for i in os.listdir(dir):
    #     for jpgfile in os.listdir(os.path.join(dir,i)):
    #         path = os.path.join(os.path.join(dir,i),jpgfile)
    #
    #     break