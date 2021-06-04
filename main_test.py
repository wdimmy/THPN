import numpy as np
import logging
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *
from models.KeyValModel import *
from models.copyseq2seq import *
from models.THPN import *

'''
python3 main_test.py -dec= -path= -bsz= -ds=
'''

BLEU = True

if (args['decoder'] == "Mem2Seq"):
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_mem2seq import *

        BLEU = True
    elif args['dataset'] == 'babi':
        from utils.utils_babi_mem2seq import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc_mem2seq import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camres_mem2seq import *
    else:
        print("You need to provide the --dataset information")
elif (args['decoder'] == "VanillaSeqToSeq"):
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_mem2seq import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi_mem2seq import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc_mem2seq import *
    else:
        print("You need to provide the --dataset information")
elif (args['decoder'] == "LuongSeqToSeq"):
    if args['dataset'] == 'camrest':
        from utils.utils_camrest import *
elif args['decoder'] == "PTRUNK":
    if args['dataset'] == 'camrest':
        from utils.utils_camrest import *
    if args['dataset'] == 'dstc2':
        from utils.utils_dstc2 import *
elif args['decoder'] == "KeyValSeqToSeq":
    if args['dataset'] == 'camrest':
        from utils.utils_camrest_keyvaluemodel import *
elif args['decoder'] == "CopySeqToSeq":
    if args['dataset'] == 'camrest':
        from utils.utils_camrest_copyseq2seq import *
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_copyseq2seq import *
elif args['decoder'] == 'THPN':
    if args['dataset'] == 'babi':
        from utils.utils_babi_THPN import *
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_THPN import *
    if args['dataset'] == 'dstc2':
        from utils.utils_dstc2_THPN import *
    if args['dataset'] == 'camrest':
        from utils.utils_camrest_THPN import *
else:
    print('111111111111111111111111111')
    if args['dataset'] == 'kvr':
        from utils.utils_kvr import *

        BLEU = True
    elif args['dataset'] == 'babi':
        from utils.utils_babi import *
    else:
        print("You need to provide the --dataset information")

# Configure models
# directory = args['path'].split("/")
task = args['task']
HDD = 128
Emb = 128
L = 3
if args['decoder'] == 'KeyValSeqToSeq':
    train, dev, test, lang, max_len, max_r, count_kb = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                        shuffle=True)
elif args['decoder'] == 'CopySeqToSeq':
    train, dev, test, lang, max_len, max_r, kb = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                  shuffle=True)
elif args['decoder'] == 'THPN':
    train, dev, test, lang, max_len, max_r, max_r_ans = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                         shuffle=True)
else:
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=int(args['batch']),
                                                                       shuffle=True)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](
        int(HDD), max_len, max_r, lang, args['path'], task, lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)
elif args['decoder'] == 'KeyValSeqToSeq':
    model = globals()[args['decoder']](int(HDD), int(Emb), max_len, max_r, len(count_kb.keys()), count_kb, lang,
                                       args['path'], task, int(args['batch']), lr=0.0, n_layers=int(L), dropout=0.0)
elif args['decoder'] == 'CopySeqToSeq':
    model = globals()[args['decoder']](int(HDD), int(Emb), max_len, max_r, kb, lang, args['path'], task, lr=0.0,
                                       n_layers=int(L), dropout=0.0)
elif args['decoder'] == 'THPN':
    model = globals()[args['decoder']](int(HDD), max_len, max_r, lang, max_r_ans, args['path'], task, lr=0.0,
                                       n_layers=3, dropout=0.0, unk_mask=0)
else:
    model = globals()[args['decoder']](
        int(HDD), max_len, max_r, lang, args['path'], task, lr=0.0, n_layers=int(L), dropout=0.0)

acc_test = model.evaluate(test, 1e6, BLEU)
print(acc_test)
# if testOOV!=[]:
#    acc_oov_test = model.evaluate(testOOV,1e6,BLEU) 
#    print(acc_oov_test)
