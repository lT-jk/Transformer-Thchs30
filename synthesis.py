import os
import cv2
import glob
import time
import random
import librosa
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models.transformer_tts import Transformer
from utils.optimizer import LRScheduler
from utils import utils

device = torch.device("cuda:0")

# 1:创建模型及加载权重
checkpoint_path = 'logs/transformer_tts/epoch-78.pt'
model = Transformer(33,80,1,4,256,1024,has_inputs=True,src_pad_idx=0,trg_pad_idx=0).to(device)
assert(os.path.exists(checkpoint_path))
packages = torch.load(checkpoint_path)
model.load_state_dict(packages['model'],strict=False)
model = model.train(False) #在测试阶段必须使用
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("loading pretrained weights from:",checkpoint_path)
print("# of parameters: {}".format(num_params))


#数据集
from pypinyin import pinyin,Style
self_symbols = ' 01234abcdefghijklmnopqrstuvwxyz'
self_char2index = {key: value + 1 for value, key in enumerate(self_symbols)}
self_index2char = {value + 1: key for value, key in enumerate(self_symbols)}


def text2id(self, text):
    text = text.strip(" \n")
    pyin = ' '.join([py[0] for py in pinyin(text, style=Style.TONE2, heteronym=False)])
    ids = np.array([self_char2index[c] for c in pyin if c in self_char2index], dtype='int32')
    return ids

#######################################################
#开始解码
model = model.train(False)
text = "生如夏花之绚烂"
inputs = torch.from_numpy(text2id(text)).long().unsqueeze(0).to(device)



max_length = 100
targets = torch.zeros([1, 1, 80]).to(device)
with torch.no_grad():
  for i in range(max_length):
    t_mask = torch.ones([1,len(targets)]).long().to(device)
    predMel,enc_dec_attn_list = model(inputs,targets,t_mask)
    targets = torch.cat([targets[:,:-1,:], predMel[:,-1:,:]], dim=1)
    targets = torch.cat([targets, torch.zeros([1, 1, 80]).to(device)], dim=1)


import matplotlib.pyplot as plt
bi = 0
plt.figure()
for layer_idx,enc_dec_attn in enumerate(enc_dec_attn_list):
  for head_idx,attn in enumerate(enc_dec_attn[bi].detach().cpu().numpy()):
   idx = layer_idx*8+head_idx+1
   plt.subplot(4,8,idx)
   plt.imshow(attn)


plt.show()


from utils.audio import melspectrogram,inv_mel_spectrogram,load_wav,save_wav
syn_melSpec = targets[0,:-1,:].contiguous().transpose(0,1).cpu().numpy()
syn_wav = inv_mel_spectrogram(syn_melSpec)
save_wav(syn_wav,"测试/syn.wav")
