import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging
import datetime
import numpy as np
import ast
from utils.until_temp import entityList
import sys
import codecs


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


MEM_TOKEN_SIZE = 3  # 用于表征一个token或者一个KB triple中有几个元素组成，这里都是3，如果不足MEM_TOKEN_SIZE则需要pad，但这里刚好都是3，无需pad


def read_lines(filepath1, filepath2):
    key = []
    ir_ans = []
    file_obj_key = codecs.open(filepath1, 'r', 'utf-8')
    line = file_obj_key.readlines()
    for i in line:
        i = i.strip('\r\n')
        key.append(i)

    file_obj_ans = codecs.open(filepath2, 'r', 'utf-8')
    line = file_obj_ans.readlines()
    for i in line:
        i = i.strip('\r\n')
        ir_ans.append(i)
    qa_dict = {}
    for index, query in enumerate(key):
        qa_dict[query] = ir_ans[index]
    return qa_dict


def read_lines_top3(filepath1, filepath2):
    key = []
    ir_ans = []
    ir_an = []
    file_obj_key = codecs.open(filepath1, 'r', 'utf-8')
    line = file_obj_key.readlines()
    for i in line:
        i = i.strip('\r\n')
        key.append(i)

    file_obj_ans = codecs.open(filepath2, 'r', 'utf-8')
    line = file_obj_ans.readlines()
    count = 0
    for i in line:
        if len(i) < 3:
            ir_ans.append(ir_an)
            ir_an = []
        else:
            i = i.strip('\r\n')
            ir_an.append(i)
    qa_dict = {}
    for index, query in enumerate(key):
        qa_dict[query] = ir_ans[index]
    return qa_dict


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.n_words = 4  # Count default tokens

    def index_words(self, story, trg=False):
        if trg:  # 一个词只用一个元素代表
            for word in story.split(' '):
                self.index_word(word)
        else:  # 一个词用多个元素代表，比如这里是一个三元组
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, src_word2id, trg_word2id, max_len, conv_seq, ent, ID, ans_seq,
                 ir_seq, max_r_ans):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq

        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.conv_seq = conv_seq
        self.ent = ent
        self.ID = ID
        self.ans_seq = ans_seq
        # print(self.ans_seq)
        self.ir_seq = ir_seq
        self.max_r_ans = max_r_ans

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        index_ans = self.ans_seq[index]
        src_plain = self.conv_seq[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        ir_seq = self.ir_seq[index]

        index_s = self.preprocess_inde(index_s, src_seq)
        index_ans = self.preprocess_inde(index_ans, ir_seq)

        conv_ir_seq = self.ir_seq[index]
        conv_ir_seq = self.preprocess(conv_ir_seq, self.src_word2id, trg=False)
        ID = self.ID[index]

        return src_seq, trg_seq, index_s, self.max_len, src_plain, self.trg_seqs[index], \
               self.ent[index], ID, index_ans, ir_seq, conv_ir_seq, self.max_r_ans  # ir_seq:word; conv_ir_seq:seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:  # 针对于target而言，每个单词不是三元组，而且需要加<EOS>；可能在dev和test里遇到UNK_token
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:  # 针对于source而言，每个单词是三元组，而且不需要加<EOS>；可能在dev和test里遇到UNK_token
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(sequence)
            print(story)
        return story

    def preprocess_inde(self, sequence, src_seq):
        # add sentinel 因为target中每个句子加上了<EOS>，所以现在比index_s/index_ans都多出一个数字2, 即response的结尾，这个时候
        # 需要用p_V来预测出<EOS>字符，所以需要index_s和index_ans都指向哨兵位置，从而能够用p_V的结果；这里index_s和index_ans的长
        # 度都应该和target的长度保持一致；这里所有的句子还未进行PAD操作，PAD操作会在dataloader的collate方法中进行；
        sequence = sequence + [len(src_seq) - 1]
        sequence = torch.Tensor(sequence)
        return sequence


def collate_fn(data):
    # 这里的输入的data是一个batch的data

    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]

        if (max_len):  # pad source
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end, :] = seq[:end]

        else:  # pad target
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_all(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end, :] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences;
    src_seqs, trg_seqs, ind_seqs, max_len, src_plain, trg_plain, ent, ID, index_ans, ir_seq, conv_ir_seq, max_r_ans = zip(
        *data)
    # print(src_seqs)

    # merge sequences (from tuple of 1D tensor to 2D tensor); pad sequences
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    index_ans, _ = merge(index_ans, None)
    conv_ir_seqs, conv_ir_lengths = merge_all(conv_ir_seq, max_r_ans)

    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    ind_seqs = Variable(ind_seqs).transpose(0, 1)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()

    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, src_plain, trg_plain, ent, ID, index_ans, ir_seq, conv_ir_seqs, conv_ir_lengths


def read_langs(file_name, qa_dict_trn, entity, max_line=None):
    logging.info(("Reading lines from {}".format(file_name)))
    data = []
    contex_arr = []
    conversation_arr = []
    u = None
    r = None
    user_counter = 0
    system_counter = 0
    system_res_counter = 0
    KB_counter = 0
    dialog_counter = 0
    with open(file_name) as fin:
        cnt_ptr = 0
        cnt_voc = 0
        max_r_len = 0
        max_ir_ans = 0
        cnt_lin = 1
        time_counter = 1

        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                ans_index = []
                ir_answer_arr = [[]]
                if '\t' in line:
                    u, r = line.split('\t')
                    if u != '<SILENCE>': user_counter += 1
                    system_counter += 1
                    gen_u = generate_memory(u, "$u", str(time_counter))
                    contex_arr += gen_u
                    conversation_arr += gen_u
                    if USE_IR:
                        tmp_u = nid + ' ' + u
                        if tmp_u in qa_dict_trn:
                            ir_answer = qa_dict_trn[tmp_u]
                            ir_answer_arr = generate_memory(ir_answer, "$a", str(time_counter))
                            # contex_arr = ir_answer_arr+contex_arr
                            for key in r.split(' '):
                                if ENTPTR:  # 用于做Masking NEW and EW
                                    if (key not in entity):  # 这里的entity包括整个KB中三元组的所有subject和object，但没有relation
                                        ir_index = [loc for loc, val in enumerate(ir_answer_arr) if (val[0] == key)]
                                        if (ir_index):
                                            ir_index = max(ir_index)
                                        else:
                                            ir_index = len(ir_answer_arr)
                                    else:
                                        ir_index = len(ir_answer_arr)
                                else:
                                    ir_index = [loc for loc, val in enumerate(ir_answer_arr) if (val[0] == key)]
                                    if (ir_index):
                                        ir_index = max(ir_index)
                                    else:
                                        ir_index = len(ir_answer_arr)
                                ans_index.append(ir_index)
                            ir_answer_arr = ir_answer_arr + [['$$$$'] * MEM_TOKEN_SIZE]
                            if len(ir_answer_arr) > max_ir_ans:
                                max_ir_ans = len(ir_answer_arr)

                    r_index = []
                    for key in r.split(' '):
                        if ENTPTR:  # 用于做Masking NEW and EW, 和上面一个Masking相反
                            if (key in entity):
                                index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                                # 为了取一个KB三元组的value和一个词的三元组的第一个content都用val[0], 所以开始对KB三元组reverse
                                if (index):
                                    index = max(index)
                                    cnt_ptr += 1
                                else:
                                    index = len(contex_arr)
                                    cnt_voc += 1
                            else:
                                index = len(contex_arr)
                                cnt_voc += 1
                        else:
                            index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                            if (index):
                                index = max(index)
                                cnt_ptr += 1
                            else:
                                index = len(contex_arr)
                                cnt_voc += 1
                        r_index.append(index)
                        system_res_counter += 1

                    if len(r_index) > max_r_len:
                        max_r_len = len(r_index)

                    contex_arr_temp = contex_arr + [['$$$$'] * MEM_TOKEN_SIZE]  # contex_arr_temp和contex_arr就是有没有加$$$$的区别
                    conversation_arr_tmp = conversation_arr + [['$$$$'] * MEM_TOKEN_SIZE]
                    ent = []  # 这里的ent就是用于计算Ent.F1 score值的,在y_ture和y_pred之间计算
                    for key in r.split(' '):
                        if (key in entity):
                            ent.append(key)

                    data.append([contex_arr_temp, r, r_index, list(conversation_arr_tmp), ent,
                                 dialog_counter, ans_index,
                                 list(ir_answer_arr)])  # r_index为reply与conversation的index gate为0，1对应于r_index
                    gen_r = generate_memory(r, "$s", str(time_counter))
                    contex_arr += gen_r
                    conversation_arr += gen_r
                    time_counter += 1
                else:
                    KB_counter += 1
                    r = line
                    if USEKB:
                        contex_arr += generate_memory(r, "", "")
                        conversation_arr += generate_memory(r, "", "")
                    else:
                        # 这里是contex_arr和conversation_arr唯一的区别，如果USEKB为false则一个加入KB一个不加入
                        # 因为每个pairs的input需要历史，所以contex_arr和conversation_arr都持续保存了一个dialogue
                        # 中的历史，以生成新的input
                        conversation_arr += generate_memory(r, "", "")
            else:  # 另外一段对话
                cnt_lin += 1
                if (max_line and cnt_lin >= max_line):
                    break
                contex_arr = []
                conversation_arr = []
                time_counter = 1
                dialog_counter += 1
    max_len = max([len(d[0]) for d in data])
    logging.info("Pointer percentace= {} ".format(cnt_ptr / (cnt_ptr + cnt_voc)))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info("Avg. User Utterances: {}".format(user_counter * 1.0 / dialog_counter))
    logging.info("Avg. Bot Utterances: {}".format(system_counter * 1.0 / dialog_counter))
    logging.info("Avg. KB results: {}".format(KB_counter * 1.0 / dialog_counter))
    logging.info("Avg. responce Len: {}".format(system_res_counter * 1.0 / system_counter))
    print('Sample: ', data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], data[0][5], data[0][6], data[0][7])
    return data, max_len, max_r_len, max_ir_ans


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s" or speaker == "$a":
        for word in sent_token:
            temp = [word, speaker, 't' + str(time)] + ["PAD"] * (MEM_TOKEN_SIZE - 3)
            sent_new.append(temp)
    else:
        if sent_token[1] == "R_rating":
            sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        else:  # reverse
            sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def get_seq(pairs, lang, batch_size, type, max_len, max_r_ans):
    x_seq = []
    y_seq = []
    ptr_seq = []
    conv_seq = []
    ent = []
    ID = []
    ans_seq = []
    ir_seq = []
    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        ptr_seq.append(pair[2])
        conv_seq.append(pair[3])
        ent.append(pair[4])
        ID.append(pair[5])
        ans_seq.append(pair[6])
        ir_seq.append(pair[7])
        if (type):
            lang.index_words(pair[0])
            lang.index_words(pair[1], trg=True)
    dataset = Dataset(x_seq, y_seq, ptr_seq, lang.word2index, lang.word2index, max_len, conv_seq, ent,
                      ID, ans_seq, ir_seq, max_r_ans)  # data[7]
    # You can specify how exactly the samples need to be batched using collate_fn
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(task, batch_size=100, shuffle=True):
    # test sample
    # file_train = 'C:/Users/David_PC/PycharmProjects/Mem2Seq-master/data/dialog-bAbI-tasks/dialog-babi-task{}trn.txt'.format(
    #     task)
    # file_dev = 'C:/Users/David_PC/PycharmProjects/Mem2Seq-master/data/dialog-bAbI-tasks/dialog-babi-task{}dev.txt'.format(
    #     task)
    # file_test = 'C:/Users/David_PC/PycharmProjects/Mem2Seq-master/data/dialog-bAbI-tasks/dialog-babi-task{}tst.txt'.format(
    #     task)
    # ent = entityList(
    #     'C:/Users/David_PC/PycharmProjects/Mem2Seq-master/data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(task))
    # real data
    file_train = 'data/babi/dialog-babi-task{}trn.txt'.format(
        task)
    file_dev = 'data/babi/dialog-babi-task{}dev.txt'.format(
        task)
    file_test = 'data/babi/dialog-babi-task{}tst.txt'.format(
        task)
    ent = entityList(
        'data/babi/dialog-babi-kb-all.txt', int(task))
    qa_dict = {}
    qa_dict_trn = {}
    if TOP_K == 1 or TOP_K == 2 or TOP_K == 3 or TOP_K == 4 or TOP_K == 5:
        if args['dataset'] == 'babi':
            qa_dict = read_lines(
                'data/ir_data/babi/task{}-1/tstQuestions.txt'.format(task),
                'data/ir_data/babi/task{}-1/ir_tst_TOP{}.txt'.format(task, TOP_K))
            qa_dict_trn = read_lines(
                'data/ir_data/babi/task{}-1/trnQuestions.txt'.format(task),
                'data/ir_data/babi/task{}-1/ir_trn_TOP{}.txt'.format(task, TOP_K))
    else:
        if args['dataset'] == 'babi':
            qa_dict = read_lines_top3(
                '/home/wdl/Code/czy/MNAG/data/ir_data/babi/task{}-1/tstQuestions.txt'.format(task),
                '/home/wdl/Code/czy/MNAG/data/ir_data/babi/task{}-1/ir_tst_TOP3.txt'.format(task))
            qa_dict_trn = read_lines_top3(
                '/home/wdl/Code/czy/MNAG/data/ir_data/babi/task{}-1/trnQuestions.txt'.format(task),
                '/home/wdl/Code/czy/MNAG/data/ir_data/babi/task{}-1/ir_trn_TOP3.txt'.format(task))

    pair_train, max_len_train, max_r_train, max_ir_ans_train = read_langs(file_train, qa_dict_trn, ent, max_line=None)

    logging.info(pair_train[1:2])
    logging.info(max_len_train)
    logging.info(max_r_train)

    pair_dev, max_len_dev, max_r_dev, max_ir_ans_dev = read_langs(file_dev, qa_dict, ent, max_line=None)
    pair_test, max_len_test, max_r_test, max_ir_ans_tst = read_langs(file_test, qa_dict, ent, max_line=None)

    max_len = max(max_len_train, max_len_dev, max_len_test) + 1  # 为了考虑之后每个句子需要加上<EOS>符号？
    max_r = max(max_r_train, max_r_dev, max_r_test) + 1
    max_r_ans = max(max_ir_ans_train, max_ir_ans_dev, max_ir_ans_tst) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True, max_len, max_r_ans)
    dev = get_seq(pair_dev, lang, batch_size, False, max_len, max_r_ans)
    test = get_seq(pair_test, lang, batch_size, False, max_len, max_r_ans)

    # logging.info(train[0])
    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, lang, max_len, max_r, max_r_ans
