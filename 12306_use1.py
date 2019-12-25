from keras.models import load_model,Sequential
import numpy as np
from PIL import Image
import json
import os

class Detector:
    image_shape = (66, 66, 3)

    @staticmethod
    def load_model() -> Sequential:
        return load_model("12306——new.h5")

    def __init__(self):
        self.model = Detector.load_model()
        with open('key_map.json', 'r') as fd:
            self.labels = json.load(fd)


    @staticmethod
    def get_array(image):
        return np.asarray(image.resize((Detector.image_shape[0], Detector.image_shape[1])))

    def predict_path(self, image_path):
        return self.predict(Image.open(image_path))

    def predict_dir(self, dir):
        images = []
        for item in os.listdir(dir):
            images.append(Image.open(dir + "/" + item))
        return self.predict_batch(images)

    def predict_batch(self,images):
        images = [x.resize((Detector.image_shape[0], Detector.image_shape[1])) for x in images]
        batch = np.asarray([np.asarray(x) for x in images])
        result = self.__predict(batch)
        result_arr = []
        for x in result:
            idx = np.argmax(x)
            print(idx)
            result_arr.append(self.labels[str(idx)])
        return result_arr

    def predict(self,image):
        input_x = self.get_array(image)
        input_x = input_x[np.newaxis,:]
        result =  self.__predict(input_x)
        result_arr = []
        for x in result:
            idx = np.argmax(x)
            result_arr.append(self.labels[str(idx)])
        return result_arr

    def __predict(self,arr):
        arr = arr / 255
        return self.model.predict(arr)


labels = ["中国结",
"仪表盘",
"公交卡",
"冰箱",
"创可贴",
"刺绣",
"剪纸",
"印章",
"卷尺",
"双面胶",
"口哨",
"啤酒",
"安全帽",
"开瓶器",
"手掌印",
"打字机",
"护腕",
"拖把",
"挂钟",
"排风机",
"文具盒",
"日历",
"本子",
"档案袋",
"棉棒",
"樱桃",
"毛线",
"沙包",
"沙拉",
"海报",
"海苔",
"海鸥",
"漏斗",
"烛台",
"热水袋",
"牌坊",
"狮子",
"珊瑚",
"电子秤",
"电线",
"电饭煲",
"盘子",
"篮球",
"红枣",
"红豆",
"红酒",
"绿豆",
"网球拍",
"老虎",
"耳塞",
"航母",
"苍蝇拍",
"茶几",
"茶盅",
"药片",
"菠萝",
"蒸笼",
"薯条",
"蚂蚁",
"蜜蜂",
"蜡烛",
"蜥蜴",
"订书机",
"话梅",
"调色板",
"跑步机",
"路灯",
"辣椒酱",
"金字塔",
"钟表",
"铃铛",
"锅铲",
"锣",
"锦旗",
"雨靴",
"鞭炮",
"风铃",
"高压锅",
"黑板",
"龙舟"]

xx = Detector()
for x in labels:
    print(x)
    print(xx.predict_dir("archive/"+x))
