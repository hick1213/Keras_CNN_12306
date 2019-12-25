from PIL import  Image
import numpy as np
#
# image_shape= (66,66,3)
#
# image = Image.open("archive/中国结/0c6fff1f-7401-4f0c-8b7b-5b791cf4f0db.png")
# image_data = np.asarray(image.resize((image_shape[0],image_shape[1])))
# # print(image_data.shape)
# # print(image_data)
#
# from keras.preprocessing.image import ImageDataGenerator
# image_data_generator = ImageDataGenerator()
# train_generator = image_data_generator.flow_from_directory("archive",target_size=(64, 64),batch_size=1,save_to_dir='pic')
#
# print(train_generator.next())
# x = ["中国结",
# "仪表盘",
# "公交卡",
# "冰箱",
# "创可贴",
# "刺绣",
# "剪纸",
# "印章",
# "卷尺",
# "双面胶",
# "口哨",
# "啤酒",
# "安全帽",
# "开瓶器",
# "手掌印",
# "打字机",
# "护腕",
# "拖把",
# "挂钟",
# "排风机",
# "文具盒",
# "日历",
# "本子",
# "档案袋",
# "棉棒",
# "樱桃",
# "毛线",
# "沙包",
# "沙拉",
# "海报",
# "海苔",
# "海鸥",
# "漏斗",
# "烛台",
# "热水袋",
# "牌坊",
# "狮子",
# "珊瑚",
# "电子秤",
# "电线",
# "电饭煲",
# "盘子",
# "篮球",
# "红枣",
# "红豆",
# "红酒",
# "绿豆",
# "网球拍",
# "老虎",
# "耳塞",
# "航母",
# "苍蝇拍",
# "茶几",
# "茶盅",
# "药片",
# "菠萝",
# "蒸笼",
# "薯条",
# "蚂蚁",
# "蜜蜂",
# "蜡烛",
# "蜥蜴",
# "订书机",
# "话梅",
# "调色板",
# "跑步机",
# "路灯",
# "辣椒酱",
# "金字塔",
# "钟表",
# "铃铛",
# "锅铲",
# "锣",
# "锦旗",
# "雨靴",
# "鞭炮",
# "风铃",
# "高压锅",
# "黑板",
# "龙舟"]
# maps = {}
# for i in range(len(x)):
#     maps[i] = x[i]
#
# print(maps)
#
import json
# with open('key_map.json','w') as fd:
#     json.dump(maps,fd)


# m = {}
# with open('key_map.json','r') as fd:
#     m = json.load(fd)
#     print(m)

'''x = ["中国结",
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
"龙舟"]'''
import json,os
maps = {}
g = os.walk("archive")
for path,dir_list,file_list in g:
    for i in range(len(dir_list)):
        maps[i] = dir_list[i]
    break
print(maps)

with open('key_map.json','w') as fd:
    json.dump(maps,fd)
os._exit(0)

import os
from PIL import Image
import numpy as np
g = os.walk("archive")
y = []
x = []
for path,dir_list,file_list in g:
    for dir_name in dir_list:
        item = os.path.join(path, dir_name)
        print(item)
        f = os.walk(item)
        for path_,dir_list_,file_list_ in f:
            for it in file_list_:
                if it == '.DS_Store':
                    continue

                n = np.asarray(Image.open(path_ + "/" + it))
                x.append(n)
                y.append(maps[dir_name])
print("changed")
x = np.array(x,dtype='float32')
y = np.array(y,dtype='float32')

###数据增强
from keras.preprocessing.image import ImageDataGenerator
image_generator = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

generator = image_generator.flow(x, y, batch_size=1)
m = x
n = y
for batch in range(1,5,2):
    for _ in range(5):
        (tempX,tempY) = generator.next()
        m = np.concatenate((m,tempX),axis=0)
        n = np.concatenate((n,tempY),axis=0)
    print("saving")
    np.savez_compressed("data%d.npz" % batch,x=m,y=n)
    print(x.shape)
    print(y.shape)
    del m
    del n
    m = x
    n = y
    import gc
    gc.collect()

'''
from sklearn.preprocessing import LabelBinarizer

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
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(labels)
'''
