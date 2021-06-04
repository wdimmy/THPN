

def merge(file_name,res):
    with open(file_name) as fin:
        for line in fin:
            if line.strip():
                if '\t' in line:
                    continue
                else:
                    res.append('1 '+' '.join(line.split(' ')[1:]))
    return res

if __name__=='__main__':
    file_name = 'C:/Users/David_PC/Desktop/MNAG/data/dstc2/dstc2trn.txt'
    file_name1 = 'C:/Users/David_PC/Desktop/MNAG/data/dstc2/dstc2dev.txt'
    file_name2 = 'C:/Users/David_PC/Desktop/MNAG/data/dstc2/dstc2tst.txt'
    file_name2 = 'C:/Users/David_PC/Desktop/MNAG/data/dstc2/dstc2-kb.txt'
    res = []
    res=merge(file_name,res)
    res = merge(file_name1, res)
    res = merge(file_name2, res)
    res_file=open('C:/Users/David_PC/Desktop/MNAG/data/dstc2/dstc_tmp_kb.txt','w')
    res=list(set(res))
    for line in res:
        res_file.write(line)
    res_file.close()
