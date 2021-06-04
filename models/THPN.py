import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn  as sns
import nltk
import os
from sklearn.metrics import f1_score
import codecs
import math
import logging.handlers

LOG_FILE = str(args['dataset']) + '_' + str(TOP_K) + '_' + str(args['task']) + '_' + str(args['hidden']) + '_' + str(
    args['batch']) + '_' + str(args['learn']) + '_' + str(args['drop']) + '_' + str(args['layer'])
logger = logging.getLogger(LOG_FILE)  # get the logger named tst
if not logger.handlers:
    # LOG_FILE = str(args['dataset']) + '_' + str(TOP_K) + '_' + args['task'] + '.log'
    handler = logging.handlers.RotatingFileHandler(LOG_FILE + '.log', maxBytes=1024 * 1024, backupCount=5)  # initialize the handler
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt)  # initialize the formatter
    handler.setFormatter(formatter)  # add the formatter to the handler
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)  # add handle to the logger
file_result = open('./analysis_camrest.txt', 'w')


class THPN(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, max_r_ans, path, task, lr, n_layers, dropout, unk_mask):
        super(THPN, self).__init__()
        self.name = "THPN"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len  # max input
        self.max_r = max_r  # max responce len
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers  # n_layers refers to the number of hops
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.newsoftmax = nn.Softmax(dim=2)
        self.max_r_ans = max_r_ans
        self.ir_dict(task)  # qa_dict
        self.count = 0
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask,
                                         self.max_r_ans)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1  # Print average loss for each iteration/batch
        self.batch_size = 0
        self.loss_ans = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        print_loss_ans = self.loss_ans / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f},AL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr,
                                                                 print_loss_ans)

    def save_model(self, dec_type):
        name_data = str(args['dataset'])
        directory = 'save/THPN' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args['batch']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, answer_index, batch_size, clip,
                    teacher_forcing_ratio, conv_seqs, conv_seqs_id, reset, src_plain, ir_ans_lengths):

        # IR
        use_teacher_forcing_answer = random.random() < IF_TA
        if use_teacher_forcing_answer:
            answer = target_batches  # [m,b]
            answer_len = target_lengths
            answer_index = [list(range(target_batches.size()[0])) for _ in range(target_batches.size()[1])]
            answer_index = torch.LongTensor(answer_index)
        else:
            answer, answer_len, _ = self.IR_module(src_plain, True)




        # print('===================================')
        # print(answer)

        # reset
        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.loss_ans = 0
            self.print_every = 1
        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab, loss_Ptr, loss_Ans = 0, 0, 0
        # Run words through encoder
        decoder_hidden, hop_inf = self.encoder(input_batches)  # [1,b,e]  #[h+1,b,e]
        decoder_hidden_init = decoder_hidden.unsqueeze(0)
        decoder_hidden = decoder_hidden.unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        max_target_length = max(answer_len)
        max_target_length_label = max(target_lengths)

        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length_label, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length_label, batch_size, input_batches.size(0)))
        all_decoder_outputs_ans = Variable(torch.zeros(max_target_length_label, batch_size, answer.size(0)))
        hop_inform = Variable(torch.zeros([len(hop_inf) - 1, batch_size, self.hidden_size]))

        embed_answer_sum = Variable(torch.zeros([max_target_length, batch_size, self.hidden_size]))  # [m,b,e]
        embed_answer_sum_3 = Variable(torch.zeros([3, max_target_length, batch_size, self.hidden_size]))  # [3,m,b,e]
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()
            embed_answer_sum = embed_answer_sum.cuda()
            answer = answer.cuda()
            embed_answer_sum_3 = embed_answer_sum_3.cuda()
            all_decoder_outputs_ans = all_decoder_outputs_ans.cuda()
            conv_seqs_id = conv_seqs_id.cuda()
            answer_index = answer_index.cuda()
            hop_inform = hop_inform.cuda()

        for index, i in enumerate(hop_inf):
            if index == 0:
                continue
            hop_inform[index - 1, :, :] = i
        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        self.decoder.load_answer_memory(answer.transpose(0, 1))

        # embedding recall answers

        for t in range(max_target_length):
            embed_word = self.decoder.embed_answer(answer[t])  # [b,h]
            embed_answer_sum[t, :, :] = embed_word
        decoder_hidden = self.decoder.attention_dec(decoder_hidden, embed_answer_sum)

        conv_seqs_id = conv_seqs_id.transpose(1, 0)

        gate_hidden = decoder_hidden  # [b,h]

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length_label):
                decoder_ptr, decoder_vacab, decoder_ans, decoder_hidden_init = self.decoder.ptrMemDecoder(decoder_input,
                                                                                                          decoder_hidden_init,
                                                                                                          gate_hidden,
                                                                                                          hop_inform)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                all_decoder_outputs_ans[t] = decoder_ans
                decoder_input = target_batches[t]  # Chosen word is next input 【batch1，batch2】
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length_label):
                decoder_ptr, decoder_vacab, decoder_ans, decoder_hidden_init = self.decoder.ptrMemDecoder(decoder_input,
                                                                                                          decoder_hidden_init,
                                                                                                          gate_hidden,
                                                                                                          hop_inform)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                _, topai = decoder_ans.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                all_decoder_outputs_ans[t] = decoder_ans
                ## get the correspective word in input

                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1)))
                top_ans_i = torch.gather(conv_seqs_id[:, :, 0], 0, Variable(topai.view(1, -1)))

                next_in = []
                for i in range(batch_size):
                    # print('#########################')
                    # print(toppi.squeeze()[i])
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # print( input_lengths[i] - 1)
                    if (toppi.squeeze()[i] < input_lengths[i] - 1):
                        next_in.append(top_ptr_i.squeeze()[i].item())
                    elif topai.squeeze()[i] < ir_ans_lengths[i] - 1:
                        next_in.append(top_ans_i.squeeze()[i].item())
                    else:
                        next_in.append(topvi.squeeze()[i])

                decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation

        # print('======================loss VOCAB================')
        # print(target_batches.transpose(0,1).shape)
        # print(target_lengths)
        # print(all_decoder_outputs_vocab.transpose(0,1).shape)

        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq *V
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        # print('======================loss PTR================')
        # print(target_index.transpose(0,1).shape)
        # print(target_lengths)
        # print(all_decoder_outputs_ptr.transpose(0,1).shape)

        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        # print('=====================loss Ans===============')
        # print(answer_index.shape)
        # print(ir_ans_lengths)
        # print(all_decoder_outputs_ans.transpose(0,1).shape)

        if USE_IR:
            loss_Ans = masked_cross_entropy(
                all_decoder_outputs_ans.transpose(0, 1).contiguous(),  # -> batch x seq
                answer_index.contiguous(),  # -> batch x seq
                target_lengths
            )
            loss = loss_Vocab + loss_Ptr + loss_Ans
        else:
            loss = loss_Vocab + loss_Ptr
        loss.backward()

        # Clip gradient norms
        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()
        self.loss_ans += loss_Ans.item()

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index,
                       src_plain, conv_seqs_id, ir_ans_lengths, conv_seqs):
        # gold
        answer = target_batches  # [m,b]
        answer_len = target_lengths

        # IR
        teacher_forcing_ratio = 0.5

        answer, answer_len, ir_src = self.IR_module(src_plain, False)

        # print('===================================')
        # print(answer)
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)

        # Run words through encoder
        decoder_hidden, hop_inf = self.encoder(input_batches)

        decoder_hidden_init = decoder_hidden.unsqueeze(0)

        decoder_hidden = decoder_hidden.unsqueeze(0)

        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoded_words = []
        max_target_length = 0
        max_target_length = max(answer_len)

        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))  # [r,b,m]
        all_decoder_outputs_ans = Variable(torch.zeros(self.max_r, batch_size, answer.size(0)))

        hop_inform = Variable(torch.zeros([len(hop_inf) - 1, batch_size, self.hidden_size]))
        embed_answer_sum = Variable(torch.zeros([max_target_length, batch_size, self.hidden_size]))  # [m,b,e]
        embed_answer_sum_3 = Variable(torch.zeros([3, max_target_length, batch_size, self.hidden_size]))  # [3,m,b,e]

        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()
            embed_answer_sum = embed_answer_sum.cuda()
            answer = answer.cuda()
            embed_answer_sum_3 = embed_answer_sum_3.cuda()
            all_decoder_outputs_ans = all_decoder_outputs_ans.cuda()
            conv_seqs_id = conv_seqs_id.cuda()
            hop_inform = hop_inform.cuda()
        for index, i in enumerate(hop_inf):
            if index == 0:
                continue
            hop_inform[index - 1, :, :] = i
        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]  # Choose the first value of the triplet
            p.append(elm_temp)
        ans = []
        for elm in conv_seqs:
            elm_temp = []
            for word_triple in elm:
                if len(word_triple) == 0:
                    continue
                else:
                    elm_temp.append(word_triple[0])
            ans.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time

        self.decoder.load_answer_memory(answer.transpose(0, 1))

        for t in range(max_target_length):
            embed_word = self.decoder.embed_answer(answer[t])  # [m,b]
            embed_answer_sum[t, :, :] = embed_word
        decoder_hidden = self.decoder.attention_dec(decoder_hidden, embed_answer_sum)

        conv_seqs_id = conv_seqs_id.transpose(1, 0)
        gate_hidden = decoder_hidden
        for t in range(self.max_r):
            decoder_ptr, decoder_vacab, decoder_ans, decoder_hidden_init = self.decoder.ptrMemDecoder(decoder_input,
                                                                                                      decoder_hidden_init,
                                                                                                      gate_hidden,
                                                                                                      hop_inform)
            all_decoder_outputs_vocab[t] = decoder_vacab
            all_decoder_outputs_ptr[t] = decoder_ptr
            all_decoder_outputs_ans[t] = decoder_ans

            topv, topvi = decoder_vacab.data.topk(1)
            topp, toppi = decoder_ptr.data.topk(1)
            topap, topai = decoder_ans.data.topk(1)

            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1)))  # [m,b,3]
            top_ans_i = torch.gather(conv_seqs_id[:, :, 0], 0, Variable(topai.view(1, -1)))

            next_in = []
            for i in range(batch_size):
                # if (toppi.squeeze()[i] < input_lengths[i] - 1):
                if topai.squeeze()[i] < ir_ans_lengths[i] - 1:
                    next_in.append(top_ans_i.squeeze()[i].item())
                elif toppi.squeeze()[i] < input_lengths[i] - 1:
                    # elif topai.squeeze()[i]<ir_ans_lengths[i]-1 and ans[i][topai.squeeze()[i]] != '$$$$':
                    next_in.append(top_ptr_i.squeeze()[i].item())
                else:
                    next_in.append(topvi.squeeze()[i])

            decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                # print('{},{}'.format(len(ans[i]),topai.squeeze()[i]))
                if (toppi.squeeze()[i] < len(p[i]) - 1):
                    temp.append(p[i][toppi.squeeze()[i]])
                    from_which.append('p')
                elif topai.squeeze()[i] < len(ans[i]) - 1:
                    # elif  topai.squeeze()[i]<ir_ans_lengths[i]-1 and ans[i][topai.squeeze()[i]] != '$$$$':
                    temp.append(ans[i][topai.squeeze()[i]])
                    from_which.append('a')
                else:
                    ind = topvi.squeeze()[i]
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind.item()])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)
        # self.save_response(src_plain, all_decoder_outputs_ptr,all_decoder_outputs_ans, decoded_words, target_batches,ir_src)
        # self.save_response_human(src_plain, all_decoder_outputs_ptr,all_decoder_outputs_ans, decoded_words, target_batches,ir_src)
        self.save_dialogue(src_plain, all_decoder_outputs_ptr, all_decoder_outputs_ans, decoded_words, target_batches,
                           ir_src)
        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def save_response(self, src_plain, all_decoder_outputs_ptr, all_decoder_outputs_ans, decoded_words, target_batches,
                      answer):
        file_result.write('******************START****************\n')
        src_plain_str = src_plain[0]
        file_result.write('content:\n')
        for sent in src_plain_str:
            file_result.write(sent[0])
            file_result.write(' ')
        file_result.write('\n\n')

        file_result.write('generation_ptr:\n')
        all_decoder_outputs_ptr_tmp = all_decoder_outputs_ptr.transpose(1, 0)  ## [b,r,,m]
        sent_dis = all_decoder_outputs_ptr_tmp[0]  # [m,v]
        for sen in sent_dis:
            sen_cpu = sen.cpu()
            file_result.write(str(list(sen_cpu.detach().numpy())))
            file_result.write('\n')

        file_result.write('generation_ans:\n')
        all_decoder_outputs_ans_tmp = all_decoder_outputs_ans.transpose(1, 0)  ## [b,r,,m]
        sent_dis = all_decoder_outputs_ans_tmp[0]  # [m,v]
        for sen in sent_dis:
            sen_cpu = sen.cpu()
            file_result.write(str(list(sen_cpu.detach().numpy())))
            file_result.write('\n')
        file_result.write('decode words:\n')
        for j in decoded_words:
            file_result.write(str(j[0]))
            file_result.write(' ')
        file_result.write('\n\n')

        file_result.write('ir_answer:\n')
        for ii in answer:
            file_result.write(str(ii))
            file_result.write('\n')

        file_result.write('\n\n')

        file_result.write('answer:\n')
        target_tmp = target_batches.transpose(1, 0)[0]
        for wi in target_tmp:
            file_result.write(self.lang.index2word[wi.item()])
            file_result.write(' ')
        file_result.write('\n')
        file_result.write('******************END****************\n')

    def save_dialogue(self, src_plain, all_decoder_outputs_ptr, all_decoder_outputs_ans, decoded_words, target_batches,
                      answer):
        file_result.write('#DIALOGUE#\n')
        file_result.write('ir_answer: ')
        for ii in answer:
            file_result.write(str(ii))
            file_result.write('\n')
        src_plain_str = src_plain[0]
        file_result.write('content: ')
        for sent in src_plain_str:
            file_result.write(sent[0])
            file_result.write(' ')
        file_result.write('\n')
        file_result.write('decode words: ')
        for j in decoded_words:
            file_result.write(str(j[0]))
            if str(j[0]) == '<EOS>':
                break
            file_result.write(' ')
        file_result.write('\n')
        file_result.write('gold answer: ')
        target_tmp = target_batches.transpose(1, 0)[0]
        for wi in target_tmp:
            file_result.write(self.lang.index2word[wi.item()])
            file_result.write(' ')
        file_result.write('\n')

    def save_response_human(self, src_plain, all_decoder_outputs_ptr, all_decoder_outputs_ans, decoded_words,
                            target_batches,
                            answer):
        file_result.write('#DIALOGUE#\n')
        src_plain_str = src_plain[0]
        file_result.write('content:')
        for sent in src_plain_str:
            file_result.write(sent[0])
            file_result.write(' ')
        file_result.write('\n')
        file_result.write('generation response:')
        for j in decoded_words:
            file_result.write(str(j[0]))
            if str(j[0]) == "<EOS>":
                break
            file_result.write(' ')
        file_result.write('\n')

    def evaluate(self, dev, avg_best, BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = [], [], [], []
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = [], [], [], []
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            if args['dataset'] == 'kvr' or args['dataset'] == 'camrest':
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[-5],
                                            data_dev[12], data_dev[-6])
            else:
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[-2],
                                            data_dev[12], data_dev[-3])
                # acc_P += acc_ptr
            # acc_V += acc_vac
            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<EOS>':
                        break
                    else:
                        st += e + ' '
                st = st.strip()
                temp_gen.append(st)
                correct = data_dev[6][i]
                ### compute F1 SCORE
                if args['dataset'] == 'kvr':
                    f1_true, f1_pred = computeF1(data_dev[7][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[13][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[14][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += f1_pred
                    f1_true, f1_pred = computeF1(data_dev[15][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += f1_pred
                elif args['dataset'] == 'dstc2' or args['dataset'] == 'camrest':
                    f1_true, f1_pred = computeF1(data_dev[7][i], st.lstrip().rstrip(), correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred

                if args['dataset'] == 'babi':
                    if data_dev[8][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[8][i]] = []
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1
                        dialog_acc_dict[data_dev[8][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[8][i]].append(0)
                else:
                    if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                        acc += 1

                w += wer(correct.lstrip().rstrip(), st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s += str(correct.lstrip().rstrip()) + "\n"
                hyp_s += str(st.lstrip().rstrip()) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg / float(len(dev)),  # len(dev) denotes how many batches
                                                            wer_avg / float(len(dev))))
        # dialog accuracy
        logger.info('================one res======================')
        logger.info("R:{:.4f}".format(acc_avg / float(len(dev))))
        logger.info(self.print_loss())

        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logger.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        if args['dataset'] == 'kvr':
            logger.info("F1 SCORE:\t" + str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))
            logger.info("F1 CAL:\t" + str(f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')))
            logger.info("F1 NAV:\t" + str(f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')))
            logger.info("F1 WEA:\t" + str(f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')))
        if args['dataset'] == 'dstc2' or args['dataset'] == 'camrest':
            logger.info("F1 SCORE:\t" + str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        logger.info("BLEU SCORE:" + str(bleu_score))
        if args['dataset'] != 'dstc2':
            if (BLEU):
                if (bleu_score >= avg_best):
                    self.save_model(str(self.name) + str(bleu_score))
                    logging.info("MODEL SAVED")
                return bleu_score
            else:
                acc_avg = acc_avg / float(len(dev))
                if (acc_avg >= avg_best):
                    self.save_model(str(self.name) + str(acc_avg))
                    logging.info("MODEL SAVED")
                return acc_avg
        else:
            acc_avg = acc_avg / float(len(dev))
            if acc_avg >= avg_best:
                self.save_model(str(self.name) + str(acc_avg))
                logging.info('DSTC2:MODEL SAVED')
            return acc_avg

    def IR_module(self, src_plain, train=True):
        # print('===================src_plain=============')
        # print(src_plain)
        user_query = self.src_plain2str(src_plain)
        if train:
            ir_target_batches, ir_target_lengths, res = self.ir_target(user_query, self.qa_dict_trn)
        else:
            ir_target_batches, ir_target_lengths, res = self.ir_target(user_query, self.qa_dict)
        ir_target_batches = ir_target_batches.long()
        # answer_gate = self.ir_target_gate(res, src_plain)
        return ir_target_batches, ir_target_lengths, res

    def src_plain2str(self, src_plain):
        res = []
        # print('=========================================')
        # print(src_plain)
        for sent in src_plain:
            sent1 = []
            for i in range(len(sent))[::-1]:
                sent1.append(sent[i])
            str1 = ''
            # print('########################################################')
            # print(sent1)
            # if args['dataset']=="babi" or args["dataset"]=="dstc2" or args["dataset"]=="kvr":
            tmp = sent1[1][2]
            turn = tmp[1:]
            # else:
            # tmp = sent1[0][2]
            # turn=tmp[1:]

            for word in sent1:
                if word[2] == '$$$$':
                    continue
                if word[2] == tmp:
                    str1 = word[0] + ' ' + str1
            KB_turn = 0
            if args['dataset'] == 'babi':
                kb_slot = ['R_cuisine', 'R_location', 'R_price', 'R_rating', 'R_phone', 'R_address', 'R_number']
            else:
                kb_slot = ['R_post_code', 'R_cuisine', 'R_location', 'R_phone', 'R_address', 'R_price', 'R_cuisine',
                           'R_address', 'R_rating']
            for word in sent1:
                if word[1] in kb_slot:
                    KB_turn = KB_turn + 1

            turn = int(turn) + KB_turn
            str1 = str(turn) + ' ' + str1
            res.append(str1)
        return res

    # read IR result of train data and dev data(test data)  and key:user query value:IR answer
    def read_lines(self, filepath1, filepath2):
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

    # use cuurent user query to get value from map,
    def ir_target(self, user_query, qa_dict):
        res = []
        keys = list(qa_dict.keys())
        for query in user_query:
            query = query.strip()
            if query in keys:
                res.append(qa_dict[query] + ' ' + '$$$$')
            else:
                logger.info('dict error!')
                logger.info(query)
                self.count = self.count + 1
        ir_target_lengths = []
        max_length = 0
        for target in res:
            target_len = len(target.split(' '))
            if (max_length < target_len):
                max_length = target_len
        res_num = []
        for target in res:
            # print(target)
            target = target.split(' ')
            ir_target_lengths.append(len(target))
            while (len(target) < max_length):
                target.append('PAD')
            sent = []
            for w in target:
                if w in self.lang.word2index:
                    sent.append(self.lang.word2index[w])
                elif w == 'PAD':
                    sent.append(1)
                elif w == 'EOS':
                    sent.append(2)
                else:
                    sent.append(0)
            res_num.append(sent)
        res_num = np.array(res_num)
        res_num = np.transpose(res_num)
        res_num = torch.Tensor(res_num)
        return res_num, ir_target_lengths, res

    def ir_dict(self, task):
        if args['dataset'] == 'babi':
            self.qa_dict = self.read_lines(
                'data/ir_data/babi/task{}-1/tstQuestions.txt'.format(task),
                'data/ir_data/babi/task{}-1/ir_tst_TOP{}.txt'.format(task, TOP_K))
            self.qa_dict_trn = self.read_lines(
                'data/ir_data/babi/task{}-1/trnQuestions.txt'.format(task),
                'data/ir_data/babi/task{}-1/ir_trn_TOP{}.txt'.format(task, TOP_K))
        if args['dataset'] == 'dstc2':
            self.qa_dict = self.read_lines(
                'data/ir_data/dstc2/dstc2-1/tstQuestions.txt',
                'data/ir_data/dstc2/dstc2-1/ir_tst_TOP{}.txt'.format(TOP_K))
            self.qa_dict_trn = self.read_lines(
                'data/ir_data/dstc2/dstc2-1/trnQuestions.txt',
                'data/ir_data/dstc2/dstc2-1/ir_trn_TOP{}.txt'.format(TOP_K))
        if args['dataset'] == 'kvr':
            self.qa_dict = self.read_lines(
                'data/ir_data/kvr/kvr-1/tstQuestions.txt',
                'data/ir_data/kvr/kvr-1/ir_tst_TOP{}.txt'.format(TOP_K))
            self.qa_dict_trn = self.read_lines(
                'data/ir_data/kvr/kvr-1/trnQuestions.txt',
                'data/ir_data/kvr/kvr-1/ir_trn_TOP{}.txt'.format(TOP_K))
        if args['dataset'] == 'camrest':
            self.qa_dict = self.read_lines(
                'data/ir_data/camrest/tstQuestions.txt',
                'data/ir_data/camrest/ir_tst_TOP1.txt')
            self.qa_dict_trn = self.read_lines(
                'data/ir_data/camrest/trnQuestions.txt',
                'data/ir_data/camrest/ir_trn_TOP1.txt')


def computeF1(entity, st, correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true, y_pred


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")  # C_1 C[1]
        self.softmax = nn.Softmax(dim=1)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, story):
        story = story.transpose(0, 1)  # [batch,length,word] [b,m,s]
        story_size = story.size()  # b * m * 3  [2,15,3]
        if self.unk_mask:  # dropout
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
                story = story * a.long()

        u = [self.get_state(story.size(0))]  # [b,e]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](
                story.contiguous().view(story.size(0), -1).long())  # [b , (m * s) , e]  [2,45,12] m=length
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # [b , m , s , e] []
            m_A = torch.sum(embed_A, 2).squeeze(2)  # [b , m , e]

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # [b , m , e]; 用u[-1]代表永远用最新生成的query
            prob = self.softmax(torch.sum(m_A * u_temp, 2))  # [b , m]
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)  # [b , m , e]

            prob = prob.unsqueeze(2).expand_as(m_C)  # [b , m , e]
            o_k = torch.sum(m_C * prob, 1)  # b e
            u_k = u[-1] + o_k  # b e
            u.append(u_k)
        return u_k, u


class DecoderrMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, max_r_ans):
        super(DecoderrMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.max_r_ans = max_r_ans
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        for hop in range(self.max_hops + 1):
            CA = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            CA.weight.data.normal_(0, 0.1)
            self.add_module("CA_{}".format(hop), CA)
        self.CA = AttrProxy(self, "CA_")

        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
        self.W3 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.WW = nn.Linear(2 * embedding_dim, embedding_dim)
        self.WW1 = nn.Linear(embedding_dim, self.max_r_ans)
        if IF_GATE and IF_HOP_ATT:
            self.concat = nn.Linear(3 * embedding_dim, embedding_dim)
        elif IF_GATE or IF_HOP_ATT:
            self.concat = nn.Linear(2 * embedding_dim, embedding_dim)
        else:
            self.concat = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.W2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.v = nn.Parameter(torch.rand(embedding_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

        self.Wg = nn.Linear(2 * embedding_dim, embedding_dim)
        self.Wg_g = nn.Linear(embedding_dim, embedding_dim)

    def load_memory(self, story):
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
                story = story * a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = embed_A  # b m e
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def load_answer_memory(self, story):
        story_size = story.size()
        self.a_story = []
        for hop in range(self.max_hops):
            embed_A = self.CA[hop](story.contiguous().view(story.size(0), -1))  # .long()) # b * m  * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m  * e
            self.a_story.append(embed_A)
            embed_C = self.CA[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = embed_C
        self.a_story.append(m_C)

    # def ir_recall(self,enc_query):
    def embed_answer(self, enc_query):
        embed_q = self.CA[0](enc_query)
        return embed_q.unsqueeze(0)

    # [1,b,e]     [m,b,e]
    def attention_dec(self, hidden, answer_emb):
        hidden = hidden.squeeze()  # [b,e]
        res = Variable(torch.zeros([answer_emb.size()[0], answer_emb.size()[1], answer_emb.size()[2]]))
        if USE_CUDA:
            res = res.cuda()
        for index, i in enumerate(answer_emb):
            if len(hidden.size()) != 2:
                hidden = hidden.unsqueeze(0)
            res[index, :, :] = self.softmax(self.W2(torch.cat((i, hidden), 1))) * hidden  # [m,b,e]

        e = torch.sum(res, 0)  # [b,e]
        return e.unsqueeze(0)

    # [1,b,e]  [3,m,b,e]
    def attention_dec_3(self, hidden, answer_emb):
        hidden = hidden.squeeze()  # [b,e] [e]
        res_final = Variable(torch.zeros([answer_emb.size()[1], answer_emb.size()[2], answer_emb.size()[3]]))
        res = Variable(torch.zeros([answer_emb.size()[1], answer_emb.size()[2], answer_emb.size()[3]]))
        if USE_CUDA:
            res = res.cuda()
            res_final = res_final.cuda()
        for ii in range(3):
            for index, i in enumerate(answer_emb[ii]):
                if len(hidden.size()) != 2:
                    hidden = hidden.unsqueeze(0)
                res[index, :, :] = self.softmax(self.W2(torch.cat((i, hidden), 1))) * hidden  # [m,b,e]
            res_final += res
        e = torch.sum(res_final, 0)  # [b,e]
        return e.unsqueeze(0)

    def ptrMemDecoder(self, enc_query, last_hidden, gate_hidden, hop_inf):
        embed_q = self.C[0](enc_query)  # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)  # (1,b,e)
        hidden_g = self.Wg_g(torch.tanh(self.Wg(torch.cat((hidden, gate_hidden), 2).squeeze(0))))  # [b,e]

        s_t = hidden[-1].unsqueeze(0)  # [b,e]
        H = s_t.repeat(hop_inf.size(0), 1, 1).transpose(0, 1)  # [b,hop,e]
        energy = torch.tanh(self.W3(torch.cat([H, hop_inf.transpose(1, 0)], 2)))  # [b,hop,e]
        energy = energy.transpose(2, 1)  # [b,e,hop]
        v = self.v.repeat(embed_q.size(0), 1).unsqueeze(1)  # [b,1,e]
        energy = torch.bmm(v, energy)  # [b,1,hop]
        a = F.softmax(energy, dim=2)
        context = a.bmm(hop_inf.transpose(1, 0)).squeeze(1)  # [b,h]

        if IF_GATE and IF_HOP_ATT:
            concat_input = torch.cat((hidden[-1], hidden_g, context), 1)
        elif IF_GATE:
            concat_input = torch.cat((hidden[-1], hidden_g), 1)
        elif IF_HOP_ATT:
            concat_input = torch.cat((hidden[-1], context), 1)
        else:
            concat_input = hidden[-1]
        # concat_input = torch.cat((hidden[-1],hidden_g, context), 1)
        if IF_GATE or IF_HOP_ATT:
            concat_output = torch.tanh(self.concat(concat_input))
        else:
            concat_output = concat_input

        hidden_res = self.out(concat_output)
        temp = []
        u = [hidden[0].squeeze()]  # [1,b,e]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]  # [m , b , e]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # [b.m,e]
            prob_lg = torch.sum(m_A * u_temp, 2)  # b*m
            prob_ = self.softmax(prob_lg)
            m_C = self.m_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)  # [b,m,e]
            o_k = torch.sum(m_C * prob, 1)  # [b,e]
            if hop == 0:
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg
        # p_ptr=prob_

        u = [hidden[0].squeeze()]  # [1, b, e]
        for hop in range(self.max_hops):
            m_A = self.a_story[hop]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used when the batch == 1
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)  # b*m

            prob_ = self.softmax(prob_lg)

            m_C = self.a_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)  # [b,e]
            # if hop == self.max_hops-1:
            #    p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)

        p_ans = prob_lg
        # p_ans=prob_
        return p_ptr, p_vocab, p_ans, hidden_res.unsqueeze(0)


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
