# -*- coding:utf-8 -*-
import codecs
import random

data = []
with codecs.open("cam_rest_676_clean_new_add.txt","r","utf-8") as file:
    tmp = ""
    for line in file:
        if line.strip():
            tmp += line
        else:
            data.append(tmp)
            tmp = ""


random.shuffle(data)

with codecs.open("train.txt","w","utf-8") as file:
    for item in data[:int(len(data)*0.6)]:
        file.write(item)
        file.write("\n")

with codecs.open("dev.txt","w","utf-8") as file:
    for item in data[int(len(data)*0.6):int(len(data)*0.8)]:
        file.write(item)
        file.write("\n")

with codecs.open("test.txt","w","utf-8") as file:
    for item in data[int(len(data)*0.8):]:
        file.write(item)
        file.write("\n")
