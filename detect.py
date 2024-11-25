import torch
import numpy as np
import cv2,json
from models import netcls,net_points
from PIL import Image
from torchvision import transforms
import math
import copy
from data.measure import Analyzer
class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net_ROI = net_points.MainNet(3).cuda().eval()
        self.net_ROI.load_state_dict(torch.load(r'./weight/weights_ROI.pt'))
        self.net_C2 = net_points.MainNet(3).cuda().eval()
        self.net_C2.load_state_dict(torch.load(r'./weight/weights_C2.pt'))
        self.net_C3 = net_points.MainNet(8).cuda().eval()
        self.net_C3.load_state_dict(torch.load(r'./weight/weights_C3.pt'))
        self.net_C4 = net_points.MainNet(8).cuda().eval()
        self.net_C4.load_state_dict(torch.load(r'./weight/weights_C4.pt'))

        self.net_cls = netcls.MainNet().cuda().eval()
        self.net_cls.load_state_dict(torch.load(r'./weight/weights_cls.pt'))
        self.tf = transforms.Compose([transforms.ToTensor()])
        self.stages = {1:'CS1',2:'CS2',3:'CS3',4:'CS4',5:'CS5',6:'CS6'}





    def forward(self, path, isswith=True):
        self.point = {}
        imgs = Image.open(path).convert('RGB')
        width, heigh = imgs.size
        image = cv2.imread(path)

        ROI_point = self.detect_ROI(imgs)
        ROI_point_copy = copy.deepcopy(ROI_point)
        alldata,positions ,degree_FH,image_roate = self.imgcrop(ROI_point,image)

        self.detect_C2(alldata['C2_c'],positions['C2_c'])
        self.detect_C3(alldata['C3_c'],positions['C3_c'])
        self.detect_C4(alldata['C4_c'],positions['C4_c'])

        matRotation = cv2.getRotationMatrix2D((int(ROI_point_copy['C4_c'][0]), int(ROI_point_copy['C4_c'][1])), -degree_FH, 1)

        for p_str in self.point:
            p = np.array(self.point[p_str])
            _p = np.append(p, np.array([1.]))
            _p = np.dot(matRotation, _p.T).T
            self.point[p_str] = _p.tolist()
        # for name in self.point:
        #     cv2.circle(image, (int(self.point[name][0]), int(self.point[name][1])), 5, (0, 0, 250))
        # cv2.namedWindow('image', 0)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        stage = self.detect_cls()
        return stage
    def imgcrop(self,ROI_point,image):

        C2_c = np.array(ROI_point['C2_c'])
        C3_c = np.array(ROI_point['C3_c'])
        C4_c = np.array(ROI_point['C4_c'])


        degree_FH = self.angle_counting(C2_c - C4_c, np.array([0, -1]))

        # 判断旋转方向
        if C2_c[0] < C4_c[0]:
            degree_FH = -degree_FH
        width, heigh = image.shape[1], image.shape[0]
        matRotation = cv2.getRotationMatrix2D((int(C4_c[0]), int(C4_c[1])), degree_FH, 1)
        lens = self.p2p_distance_counting(C3_c, C4_c)
        image = cv2.warpAffine(image, matRotation, (width, heigh), borderValue=(255, 255, 255))
        for p_str in ROI_point:
            p = np.array(ROI_point[p_str])
            _p = np.append(p, np.array([1.]))
            _p = np.dot(matRotation, _p.T).T
            ROI_point[p_str] = _p.tolist()
        dataall = {}
        positions = {}
        for name in ['C2_c','C3_c','C4_c']:
            C = np.array(ROI_point[name])
            C_x1 = int(C[0] - 0.8*lens)
            C_y1 = int(C[1] - 0.8*lens)
            C_x2 = int(C[0] + 0.8*lens)
            C_y2 = int(C[1] + 0.8*lens)
            img_C = image[C_y1:C_y2, C_x1:C_x2, :]
            img_c = Image.fromarray(img_C,'RGB')

            w,h = img_c.size
            positions[name] = [C_x1,C_y1,w,h]
            img = Image.new('RGB', (max(w,h), max(w,h)), (255, 255, 255))
            img.paste(img_c, (0, 0))
            img_ = img.resize((416, 416))
            img_data = self.tf(img_).cuda()
            img_data = img_data.unsqueeze(0)
            dataall[name] = img_data
        return dataall,positions,degree_FH,image
    def p2p_distance_counting(self,p1, p2):  # 计算两点之间的距离
        dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return dis



    def angle_counting(self, vector1, vector2):  # 计算两直线夹角
        cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos)
        angle = angle / math.pi * 180.
        return angle

    def detect_ROI(self, image):

        imgbg = Image.new('RGB', (max(image.size), max(image.size)), (255, 255, 255))
        imgbg.paste(image, (0, 0))
        img = imgbg.resize((416, 416))
        w, h = image.size
        img_data = self.tf(img).cuda()
        img_data = img_data.unsqueeze(0)


        out = self.net_ROI(img_data).cpu().detach().numpy()[0]


        ROI_point = {}
        for i,name in enumerate(['C2_c','C3_c','C4_c']):

            x = round(out[i*2 + 0] * max(w, h), 2)
            y = round(out[i*2 + 1] * max(w, h), 2)
            ROI_point[name] = [x, y]
        return ROI_point

    def detect_C2(self, imagedata,position):
        names = ['C2p','C2d','C2a']
        out = self.net_C2(imagedata)
        out = out.cpu().data.numpy()[0]

        for i ,name in enumerate(names):

            _x = round(out[i*2 + 0] * max(position[2:]) +position[0], 2)
            _y = round(out[i*2 + 1] * max(position[2:]) +position[1], 2)
            self.point[name] = [_x, _y]

    def detect_C3(self, imagedata,position):
        names = ['C3up','C3um','C3ua','C3pm','C3am','C3lp','C3ld','C3la']
        out = self.net_C3(imagedata)
        out = out.cpu().data.numpy()[0]
        for i ,name in enumerate(names):
            _x = round(out[i*2 + 0] * max(position[2:]) +position[0], 2)
            _y = round(out[i*2 + 1] * max(position[2:]) +position[1], 2)
            self.point[name] = [_x, _y]
    def detect_C4(self, imagedata,position):
        names = ['C4up','C4um','C4ua','C4pm','C4am','C4lp','C4ld','C4la']
        out = self.net_C4(imagedata)
        out = out.cpu().data.numpy()[0]
        for i ,name in enumerate(names):
            _x = round(out[i*2 + 0] * max(position[2:]) +position[0], 2)
            _y = round(out[i*2 + 1] * max(position[2:]) +position[1], 2)
            self.point[name] = [_x, _y]

    def detect_cls(self):
        analyzer = Analyzer(self.point)
        result = analyzer.public_item_counting()
        input = np.array(result)
        input[0:3] = input[0:3]/20
        input_data = torch.FloatTensor(input)
        input_data = input_data.unsqueeze(0).cuda()
        outputs = self.net_cls(input_data)
        out = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(out, 1)
        stage = self.stages[int(pred[0])]
        return stage
if __name__=="__main__":
    detect = Detector()
    path = r'.\imagedata\pics\006.jpg'
    stage = detect(path)
    print(stage)

















