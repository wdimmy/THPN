# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict

entity = set()
with codecs.open("cam_rest_kb_clean_new.txt","r","utf-8") as file:
    skip = 0
    cnt = 0
    for line in file:
        cnt == 0
        lines = line.strip().split()
        if "poi" in line:
            entity.add(lines[-1])
            continue
        if len(lines) == 4 or len(lines) == 5 :
            entity.add("_".join(lines[1:3]))
        else:
            skip += 1


with codecs.open("temp/kb.txt", "w", "utf-8") as file:
    for item in entity:
        file.write(item+"\n")



with codecs.open("dev.txt","r","utf-8") as file:
    new_file = codecs.open("temp/dev.txt","w","utf-8")
    local_entities = defaultdict(list)
    topic_entities = dict()
    weather_map = dict()
    local_key = []
    temp_key = ""
    target_key = ""
    target_list = []
    category = ""
    user = set()
    cnt = 0
    for line in file:
        if len(line.strip()) ==0:
            new_file.write(line)
            local_entities = defaultdict(list) # set to None
            topic_entities = dict()
            weather_map = dict()
            user = set()
            continue

        if "\t" not in line:
            new_file.write(line)
            lines = line.strip().split()

            if len(lines) == 4:
                local_entities[lines[-1]].append((lines[-3], lines[-2]))
                topic_entities[lines[1]] = [lines[1]]
            if len(lines) == 5:
                local_entities[lines[-1]].append((lines[-4], lines[-3]))
                local_entities[lines[-2]].append((lines[-4], lines[-3]))
                topic_entities[lines[1]] = [lines[1]]

        else:
            lines = line.strip().split("\t")
            if len(lines) != 3:
                print(line)
                print("error")
                exit(0)

            else:

                for item in lines[0].split(" "):
                    for key, value in local_entities.items():
                        if item == key or item in value:
                            user.add(item)
                for item in lines[1].split(" "):
                    for key, value in local_entities.items():
                        if item in key or item in value:
                            user.add(item)

                systemRes = lines[1]
                entities = eval(lines[2])
                tmp_entities = entities[:]

                for i in range(len(entities)):
                    item = entities[i]
                    if item in local_entities:
                        result = local_entities[item]
                    elif item in topic_entities:
                        result = topic_entities[item]
                    else:
                        result = []
                    if len(result) == 1:
                        try:
                           systemRes = systemRes.replace(item, "_".join(result))
                           tmp_entities[i] = "_".join(result)
                        except:
                           systemRes = systemRes.replace(item, "_".join(result[0]))
                           tmp_entities[i] = "_".join(result[0])

                    else:
                        for candi in result:
                            for userEnt in user:
                                if userEnt in candi:
                                    systemRes = systemRes.replace(item, "_".join(candi))
                                    tmp_entities[i] =  "_".join(candi)
                                    break
                new_file.write(lines[0].strip()+"\t"+systemRes+"\t"+ "["+",".join(["'"+ a + "'" for a in tmp_entities])+"]" + "\n")


