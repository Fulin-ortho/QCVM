import os.path

import cv2,os
import glob
import json
import numpy as np
import math


def p2p_distance_counting( p1, p2):  # 计算两点之间的距离
    dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return dis
def angle_counting( vector1, vector2):  # 计算两直线夹角
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos)
    angle = angle / math.pi * 180.
    return angle

def imgCrop_ROI(imagepath,savepath_C2,savepath_C3,savepath_C4):
    for imgfile in glob.glob(imagepath):
        jsonfile = imgfile.replace('pics','label').replace('.jpg','.json')
        with open(jsonfile,'r') as jsf:
            label = json.load(jsf)
        image = cv2.imread(imgfile)
        C2_c = np.array(label['C2_c'])
        C3_c = np.array(label['C3_c'])
        C4_c = np.array(label['C4_c'])
        degree_FH = angle_counting(C2_c - C4_c, np.array([0, -1]))
        print(degree_FH)
        # 判断旋转方向
        if C2_c[0] < C4_c[0]:
            degree_FH = -degree_FH

        width,heigh = image.shape[1],image.shape[0]
        matRotation = cv2.getRotationMatrix2D((int(C4_c[0]), int(C4_c[1])), degree_FH, 1)
        for p_str in label:
            p = np.array(label[p_str])
            _p = np.append(p, np.array([1.]))
            _p = np.dot(matRotation, _p.T).T
            label[p_str] = _p.tolist()

        lens = p2p_distance_counting(C3_c,C4_c)
        image = cv2.warpAffine(image, matRotation, (width,heigh), borderValue=(255, 255, 255))


        C2_c = np.array(label['C2_c'])
        C3_c = np.array(label['C3_c'])
        C4_c = np.array(label['C4_c'])
        C2_x1 = int(C2_c[0] -  0.8*lens)
        C2_y1 = int(C2_c[1] - 0.8*lens)
        C2_x2 = int(C2_c[0] + 0.8*lens)
        C2_y2 = int(C2_c[1] + 0.8*lens)
        img_C2 = image[C2_y1:C2_y2,C2_x1:C2_x2,:]
        cv2.imwrite(os.path.join(savepath_C2,imgfile.split(os.sep)[-1]),img_C2)
        C2_names = ['C2p', 'C2d', 'C2a']
        C2_points = {}
        for name in C2_names:
            C2_points[name] = (np.array(label[name])-np.array([C2_x1,C2_y1])).tolist()
        with open(os.path.join(savepath_C2,imgfile.split(os.sep)[-1]).replace('.jpg','.json').replace('pics','label'),'w') as jsf:
            json.dump(C2_points,jsf)

        C3_x1 = int(C3_c[0] -  0.8*lens)
        C3_y1 = int(C3_c[1] -  0.8*lens)
        C3_x2 = int(C3_c[0] +  0.8*lens)
        C3_y2 = int(C3_c[1] +  0.8*lens)
        img_C3 = image[C3_y1:C3_y2, C3_x1:C3_x2, :]
        cv2.imwrite(os.path.join(savepath_C3, imgfile.split(os.sep)[-1]), img_C3)
        C3_names = ['C3up','C3um','C3ua','C3pm','C3am','C3lp','C3ld','C3la']
        C3_points = {}
        for name in C3_names:
            C3_points[name] = (np.array(label[name])-np.array([C3_x1,C3_y1])).tolist()
        with open(os.path.join(savepath_C3,imgfile.split(os.sep)[-1]).replace('.jpg','.json').replace('pics','label'),'w') as jsf:
            json.dump(C3_points,jsf)

        C4_x1 = int(C4_c[0] -  0.8*lens)
        C4_y1 = int(C4_c[1] -  0.8*lens)
        C4_x2 = int(C4_c[0] +  0.8*lens)
        C4_y2 = int(C4_c[1] +  0.8*lens)
        img_C4 = image[C4_y1:C4_y2, C4_x1:C4_x2, :]
        cv2.imwrite(os.path.join(savepath_C4, imgfile.split(os.sep)[-1]), img_C4)
        C4_names = ['C4up','C4um','C4ua','C4pm','C4am','C4lp','C4ld','C4la']
        C4_points = {}
        for name in C4_names:
            C4_points[name] = (np.array(label[name]) - np.array([C4_x1, C4_y1])).tolist()
        with open(os.path.join(savepath_C4, imgfile.split(os.sep)[-1]).replace('.jpg', '.json').replace('pics', 'label'),
                  'w') as jsf:
            json.dump(C4_points, jsf)

if __name__=="__main__":
    imagepath = r'H:\test_data300\lunwen\pics\*'
    savepath_C2 = r'H:\test_data300\lunwen\C2\pics'
    savepath_C3 = r'H:\test_data300\lunwen\C3\pics'
    savepath_C4 = r'H:\test_data300\lunwen\C4\pics'
    imgCrop_ROI(imagepath,savepath_C2,savepath_C3,savepath_C4)


