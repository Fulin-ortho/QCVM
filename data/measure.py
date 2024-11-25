import json
import numpy as np
from sympy import *
# from detects import detect_jingzhui_cls

# detect = detect_jingzhui_cls.Detector_cls()
class Analyzer:
    def __init__(self, input_dic):


        self.C2p = np.array(input_dic['C2p'])
        self.C2d = np.array(input_dic['C2d'])
        self.C2a = np.array(input_dic['C2a'])

        self.C3up = np.array(input_dic['C3up'])
        self.C3ua = np.array(input_dic['C3ua'])
        self.C3lp = np.array(input_dic['C3lp'])
        self.C3la = np.array(input_dic['C3la'])
        self.C3ld = np.array(input_dic['C3ld'])
        self.C3um = np.array(input_dic['C3um'])
        self.C3am = np.array(input_dic['C3am'])
        self.C3pm = np.array(input_dic['C3pm'])

        self.C4up = np.array(input_dic['C4up'])
        self.C4um = np.array(input_dic['C4um'])
        self.C4am = np.array(input_dic['C4am'])
        self.C4lp = np.array(input_dic['C4lp'])
        self.C4la = np.array(input_dic['C4la'])
        self.C4ld = np.array(input_dic['C4ld'])
        self.C4ua = np.array(input_dic['C4ua'])
        self.C4pm = np.array(input_dic['C4pm'])





    def angle_counting(self, vector1, vector2): #计算两直线夹角
        cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos)
        angle = angle / 3.14 * 180.
        return angle

    def p2line_distance_counting(self, p, p1, p2):# 计算点到直线的距离，p为直线外的点，p1,p2为直线上的点
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p1[1] * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1])
        # dis = ((p2[1] - p1[1]) * p[0] + (p2[0] - p1[0]) * p[1] - p1[0] * p2[1] + p2[0] * p1[1]) / (((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) ** 0.5)
        dis = np.abs((A * p[0] + B * p[1] + C) / (A ** 2 + B ** 2) ** 0.5)
        return dis

    def foot_point_counting(self, p, p1, p2): # 计算点与直线的垂足点 p为直线外的点，p1,p2为直线上的点
        u = ((p1[0] - p2[0]) * (p[0] - p1[0]) + (p1[1] - p2[1]) * (p[1] - p1[1])) / ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        x = p1[0] + u * (p1[0] - p2[0])
        y = p1[1] + u * (p1[1] - p2[1])
        return np.array([x, y])

    def p2p_distance_counting(self, p1, p2): #计算两点之间的距离
        dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return dis

    def public_item_counting(self):

        PH3 = self.p2line_distance_counting(self.C3up, self.C3lp, self.C3la)
        AH3 = self.p2line_distance_counting(self.C3ua, self.C3lp, self.C3la)

        PH4 = self.p2line_distance_counting(self.C4up, self.C4lp, self.C4la)
        AH4 = self.p2line_distance_counting(self.C4ua, self.C4lp, self.C4la)

        H3 = self.p2line_distance_counting(self.C3um, self.C3lp, self.C3la)
        W3 = self.p2line_distance_counting(self.C3am, self.C3lp, self.C3up)

        H4 = self.p2line_distance_counting(self.C4um, self.C4lp, self.C4la)
        W4 = self.p2line_distance_counting(self.C4am, self.C4lp, self.C4up)


        angle_C2 = self.angle_counting(self.C2a - self.C2p, self.C2d - self.C2p)
        angle_C3 = self.angle_counting(self.C3la-self.C3lp,self.C3ld-self.C3lp)
        angle_C4 = self.angle_counting(self.C4la-self.C4lp,self.C4ld-self.C4lp)

        result = []
        result.extend((angle_C2,angle_C3,angle_C4,AH3/PH3,AH4/PH4,H3/W3,H4/W4,))
        return result

if __name__=="__main__":
    import json,glob,os
    names_cls = {'CS1':'1','CS2':'2','CS3':'3','CS4':'4','CS5':'5','CS6':'CS6'}
    dir = r'..\imagedata\label\CS6\*'
    f = open('label.txt','w')
    for file in glob.glob(dir):
        with open(file,'r') as jsf:
            out = json.load(jsf)
        analyzer = Analyzer(out)
        result = analyzer.public_item_counting()
        f.write(names_cls[file.split(os.sep)[-2]]+' {}'.format(result))




