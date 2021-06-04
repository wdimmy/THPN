# -*- coding: utf-8 -*-
import numpy as np
import logging
from tqdm import tqdm
from utils.config import *
from models.Mem2Seq import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.copyseq2seq import *
from models.KeyValModel import *
from models.copynet import *
from models.MNAG import *
from models.THPN import *

BLEU = True
if args['decoder'] == "Mem2Seq":
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_mem2seq import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camres_mem2seq import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi_mem2seq import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc_mem2seq import *
    else:
        print("You need to provide the --dataset information")
elif args['decoder'] == "VanillaSeqToSeq":
    if args['dataset'] == 'kvr':
        from utils.utils_kvr import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2 import *
    else:
        print("You need to provide the --dataset information")
elif args['decoder'] == "LuongSeqToSeq":
    if args['dataset'] == 'kvr':
        from utils.utils_kvr import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2 import *
    else:
        print("You need to provide the --dataset information")

elif args['decoder'] == "PTRUNK":
    if args['dataset'] == 'kvr':
        from utils.utils_kvr import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2 import *
    else:
        print("You need to provide the --dataset information")
elif args['decoder'] == 'CopySeqToSeq':
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_copyseq2seq import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi_copyseq2seq import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest_copyseq2seq import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2_copyseq2seq import *
    else:
        print("You need to provide the --dataset information")
elif args['decoder'] == 'KeyValSeqToSeq':
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_keyvaluemodel import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi_keyvaluemodel import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest_keyvaluemodel import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2_keyvaluemodel import *
    else:
        print("You need to provide the --dataset information")
elif args['decoder'] == 'CopyNet':
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_copynet import *
    elif args['dataset'] == 'babi':
        from utils.utils_babi_copynet import *
    elif args['dataset'] == 'camrest':
        from utils.utils_camrest_copynet import *
    elif args['dataset'] == 'dstc2':
        from utils.utils_dstc2_copynet import *
    else:
        print("You need to provide the --dataset information")

elif args['decoder'] == 'MNAG':
    if args['dataset'] == 'babi':
        from utils.utils_babi_MNAG import *

elif args['decoder'] == 'THPN':
    if args['dataset'] == 'babi':
        from utils.utils_babi_THPN import *
    if args['dataset'] == 'kvr':
        from utils.utils_kvr_THPN import *
    if args['dataset'] == 'dstc2':
        from utils.utils_dstc2_THPN import *
    if args['dataset'] == 'camrest':
        from utils.utils_camrest_THPN import *

if args['decoder'] == "Mem2Seq":
    # Configure models
    avg_best, cnt, acc = 0.0, 0, 0.0
    cnt_1 = 0
    ### LOAD DATA
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                       shuffle=True)
    model = globals()[args['decoder']](int(args['hidden']),
                                       max_len, max_r, lang, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop']),
                                       unk_mask=bool(int(args['unk_mask']))
                                       )
    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            if args['decoder'] == "Mem2Seq":
                if args['dataset'] == 'kvr':
                    model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                      len(data[1]), 10.0, 0.5, data[-2], data[-1], i == 0)
                else:
                    model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                      len(data[1]), 10.0, 0.5, data[-4], data[-3], i == 0)
            else:
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, i == 0)
            pbar.set_description(model.print_loss())  # 显示每一个iteration/batch的各种average loss

        if (epoch + 1) % int(args['evalp']) == 0:
            acc = model.evaluate(dev, avg_best, BLEU)
            if 'Mem2Seq' in args['decoder']:
                model.scheduler.step(acc)
            if acc >= avg_best:
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 5): break
            if (acc == 1.0): break

elif args['decoder'] == 'CopySeqToSeq':
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, lang, max_len, max_r, kb = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                  shuffle=True)  # max_len single number for all dataset
    pbar = tqdm(enumerate(train), total=len(train))

    model = globals()[args['decoder']](int(args['hidden']), int(args['embed']),
                                       max_len, max_r, kb, lang, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop'])
                                       )
    for epoch in range(200):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data[0], data[1], data[2], data[3], data[4],
                              len(data[1]), 10.0, 0.5, i == 0)
            pbar.set_description(model.print_loss())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, BLEU)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 5): break
            if (acc == 1.0): break

elif args['decoder'] == 'KeyValSeqToSeq':
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, lang, max_len, max_r, count_kb = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                        shuffle=True)  # max_len single number for all dataset
    pbar = tqdm(enumerate(train), total=len(train))
    model = globals()[args['decoder']](int(args['hidden']), int(args['embed']),
                                       max_len, max_r, len(count_kb.keys()), count_kb, lang, args['path'], args['task'],
                                       int(args['batch']),
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop'])
                                       )
    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data[0], data[1], data[2], data[3], data[4], data[12], data[13],
                              len(data[1]), 10.0, 0.5, i == 0)
            pbar.set_description(model.print_loss())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, BLEU)
            # if 'Mem2Seq' in args['decoder']:
            # model.scheduler.step(acc)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 5): break
            if (acc == 1.0): break
elif args['decoder'] == 'CopyNet':
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, lang, max_len, max_r, kb = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                  shuffle=True)  # max_len single number for all dataset
    pbar = tqdm(enumerate(train), total=len(train))

    model = globals()[args['decoder']](int(args['hidden']), int(args['embed']),
                                       max_len, max_r, kb, lang, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop'])
                                       )
    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data[0], data[1], data[2], data[3], data[7],
                              len(data[1]), 10.0, 0.5, i == 0)
            pbar.set_description(model.print_loss())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, BLEU)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 5): break
            if (acc == 1.0): break
elif args['decoder'] == 'MNAG':
    avg_best, cnt, acc = 0.0, 0, 0.0
    cnt_1 = 0
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                       shuffle=True)

    pbar = tqdm(enumerate(train), total=len(train))

    model = globals()[args['decoder']](int(args['hidden']),
                                       max_len, max_r, lang, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop']),
                                       unk_mask=bool(int(args['unk_mask']))
                                       )
    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            if args['dataset'] == 'kvr':
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, data[-2], data[-1], i == 0, data[6])
            else:
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, data[-4], data[-3], i == 0, data[6])
            pbar.set_description(model.print_loss())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, BLEU)
            model.scheduler.step(acc)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 50): break
            if (acc == 1.0): break
elif args['decoder'] == 'THPN':
    avg_best, cnt, acc = 0.0, 0, 0.0
    cnt_1 = 0
    train, dev, test, lang, max_len, max_r, max_r_ans = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                         shuffle=True)
    pbar = tqdm(enumerate(train), total=len(train))

    model = globals()[args['decoder']](int(args['hidden']),
                                       max_len, max_r, lang, max_r_ans, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop']),
                                       unk_mask=bool(int(args['unk_mask']))
                                       )

    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            if args['dataset'] == 'kvr' or args['dataset'] == 'camrest':
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[9],
                                  len(data[1]), 10.0, 0.5, data[-6], data[-5], i == 0, data[5], data[12])
            else:
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[9],
                                  len(data[1]), 10.0, 0.5, data[-3], data[-2], i == 0, data[5], data[12])
            pbar.set_description(model.print_loss())

        if (epoch + 1) % int(args['evalp']) == 0:
            acc = model.evaluate(test, avg_best, BLEU)
            # model.scheduler.step(acc)
            if acc >= avg_best:
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if cnt == 50: break
            if acc == 1.0: break

else:
    # Configure models
    avg_best, cnt, acc = 0.0, 0, 0.0
    cnt_1 = 0
    ### LOAD DATA
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
                                                                       shuffle=True)
    model = globals()[args['decoder']](int(args['hidden']),
                                       max_len, max_r, lang, args['path'], args['task'],
                                       lr=float(args['learn']),
                                       n_layers=int(args['layer']),
                                       dropout=float(args['drop'])
                                       )
    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            if args['decoder'] == "Mem2Seq":
                if args['dataset'] == 'kvr':
                    model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                      len(data[1]), 10.0, 0.5, data[-2], data[-1], i == 0)
                else:
                    model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                      len(data[1]), 10.0, 0.5, data[-4], data[-3], i == 0)
            else:
                model.train_batch(data[0], data[1], data[2], data[3], data[4], data[5],
                                  len(data[1]), 10.0, 0.5, i == 0)
            pbar.set_description(model.print_loss())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, BLEU)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
            else:
                cnt += 1
            if (cnt == 5): break
            if (acc == 1.0): break
