import json

file_path_kb='C:/Users/David_PC/Desktop/MNAG_back/MNAG/datasets/camrest/CamRest676/CamRestDB.json'
kb=[]
kb_res_name=[]
kb_dict={}
with open(file_path_kb, 'r') as f:
    data = json.load(f)
    for item in data:
        name = item['name'].strip().lower()
        kb_context=''
        address=item['address'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'address' + ' ' + address.replace(' ','_') + '\n'
        area=item['area'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'area' + ' ' + area.replace(' ','_') + '\n'
        food=''
        if 'food' in item.keys():
            food=item['food'].strip().lower()
            kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'food' + ' ' + food.replace(' ','_') + '\n'
        location=item['location'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'location' + ' ' + location.replace(' ','_') + '\n'
        phone=''
        if 'phone' in item.keys():
            phone=item['phone'].strip().lower()
            kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'phone' + ' ' + phone.replace(' ','_') + '\n'

        pricerange=item['pricerange'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'pricerange' + ' ' + pricerange.replace(' ','_') + '\n'
        postcode=item['postcode'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'postcode' + ' ' + postcode.replace(' ','_') + '\n'
        type=item['type'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'type' + ' ' + type.replace(' ','_') + '\n'
        id=item['id'].strip().lower()
        kb_context = kb_context + '1 ' + name.replace(' ','_') + ' ' + 'id' + ' ' + id.replace(' ','_') + '\n'

        if address not in kb:
            kb.append(address)
        if area not in kb:
            kb.append(area)
        if food !='' and food not in kb:
            kb.append(food)
        if location not in kb:
            kb.append(location)
        if phone!=''and phone not in kb:
            kb.append(phone)
        if pricerange not in kb:
            kb.append(pricerange)
        if postcode not in kb:
            kb.append(postcode)
        if type not in kb:
            kb.append(type)
        if id not in kb:
            kb.append(id)
        if name not in kb:
            kb.append(name)
            kb_res_name.append(name)
            kb_dict[name]=kb_context

file_path='C:/Users/David_PC/Desktop/MNAG_back/MNAG/datasets/camrest/CamRest676/CamRest676.json'
file_res_path='C:/Users/David_PC/Desktop/MNAG/data/camrest/cam_rest_676_clean_new_add.txt'
cam_res=open(file_res_path,'w')
# appear restraunt name
with open(file_path, 'r') as f:
    data = json.load(f)
    for item in data:
        candi_name=[]
        for ind in item['dial']:
            turn = ind['turn']
            u = ind['usr']['transcript'].strip().lower()
            s = ind['sys']['sent'].strip().lower()
            for pp in kb_res_name:
                if pp in u or pp in s:
                    if pp not in candi_name:
                        candi_name.append(pp)
        for j in candi_name:
            cam_res.write(kb_dict[j])

        for ind in item['dial']:
            turn=ind['turn']
            u=ind['usr']['transcript'].strip().lower()
            s = ind['sys']['sent'].strip().lower()
            ent=[]
            for k in kb:
                if k in u:
                    kk=k.replace(' ','_')
                    u=u.replace(k,kk)
                    if kk not in ent:
                        ent.append(kk)
                if k in s:
                    kk=k.replace(' ','_')
                    s=s.replace(k,kk)
                    if kk not in ent:
                        ent.append(kk)
            line=str(turn)+' '+u+'\t'+s+'\t'+str(ent)
            cam_res.write(line)
            cam_res.write('\n')
        cam_res.write('\n')

# file_path_kb='./data/camres/CamRestDB.json'
# file_res_kb__path='./data/camres/cam_rest_kb_clean_new.txt'
# cam_res=open(file_res_kb__path,'w')
# with open(file_path_kb, 'r') as f:
#     data = json.load(f)
#     for item in data:
#         address=item['address'].strip().replace(' ','_').lower()
#         area=item['area'].strip().replace(' ','_').lower()
#         food=''
#         if 'food' in item.keys():
#             food=item['food'].strip().replace(' ','_').lower()
#         location=item['location'].strip().replace(' ','_').lower()
#         phone=''
#         if 'phone' in item.keys():
#             phone=item['phone'].strip().replace(' ','_').lower()
#         pricerange=item['pricerange'].strip().replace(' ','_').lower()
#         postcode=item['postcode'].strip().replace(' ','_').lower()
#         type=item['type'].strip().replace(' ','_').lower()
#         id=item['id'].strip().replace(' ','_').lower()
#         name=item['name'].strip().replace(' ','_').lower()
#         line='1 '+name+' '+'address'+' '+address
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'area' + ' ' + area
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'food' + ' ' + food
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'location' + ' ' + location
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'phone' + ' ' + phone
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'pricerange' + ' ' + pricerange
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'postcode' + ' ' + postcode
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'type' + ' ' + type
#         cam_res.write(line)
#         cam_res.write('\n')
#         line = '1 '+name + ' ' + 'id' + ' ' + id
#         cam_res.write(line)
#         cam_res.write('\n')









