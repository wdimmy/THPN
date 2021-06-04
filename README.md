# THPN
This repository contains the code for our ACL 2021 Workshop Paper 
"A Template-guided Hybrid Pointer Network for Knowledge-basedTask-oriented Dialogue Systems".

<p align="center">
<img src="./model.jpg" width="100%" />
</p>

Some codes are adapted from this [repository](https://github.com/HLTCHKUST/Mem2Seq.git).

If the code is used in your research, please star our repo and cite our paper as follows:


# CMD for the THPN model training

python main_train.py -lr=0.001 -layer=1 -hdd=12 -dr=0.0 -dec=THPN -bsz=2 -ds=babi -t=1 -topk=1 -use_ir=True














