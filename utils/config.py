import os
import logging 
import argparse
from tqdm import tqdm

UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False, default="kvr")
parser.add_argument('-t','--task', help='Task Number', required=False,default="kvr")
parser.add_argument('-dec','--decoder', help='decoder model', required=False, default="VanillaSeqToSeq")
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False, default=128)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, default=1)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False, default=0.01)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, default=0.2)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', required=False, default=1)
parser.add_argument('-layer','--layer', help='Layer Number', required=False,default=1)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-test','--test', help='Testing mode', required=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-useKB','--useKB', help='Put KB in the input or not', required=False, default=1)
parser.add_argument('-ep','--entPtr', help='Restrict Ptr only point to entity', required=False, default=1)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=5)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-emb','--embed', help='Embed size', required=False)
parser.add_argument('-Bi','--Bi', help='Bi RNN', required=False,default=True)
parser.add_argument('-topk','--topk', help='IR topk', required=False,default=1)
parser.add_argument('-use_ir','--use_ir', help='ir answer', required=False,default=True)
parser.add_argument('-g','--gate', help='Gate', required=False,default=1)
parser.add_argument('-ha','--hop_att', help='hop attention', required=False,default=1)
parser.add_argument('-ac','--addCon', help='if encode ir answers', required=False,default=0)
parser.add_argument('-ta','--ta', help='if teacher forcing for retrived answers', required=False,default=0)

args = vars(parser.parse_args())
print(args)

name = str(args['task'])+str(args['decoder'])+str(args['hidden'])+str(args['batch'])+str(args['learn'])+str(args['drop'])+str(args['layer'])+str(args['limit'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

LIMIT = int(args["limit"]) 
USEKB = int(args["useKB"])
ENTPTR = int(args["entPtr"])
ADDCON=int(args["addCon"])
ADDNAME = args["addName"]
USE_BI=args['Bi']
TOP_K=int(args['topk'])
USE_IR=args['use_ir']
IF_GATE = int(args["gate"])
IF_HOP_ATT = int(args["hop_att"])
IF_TA = float(args["ta"])
