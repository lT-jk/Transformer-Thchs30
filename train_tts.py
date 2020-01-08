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
from tensorboardX import SummaryWriter
logs_root = "logs/transformer_tts"
os.makedirs(logs_root,exist_ok=True)
logs_writer = SummaryWriter(logs_root)


# 1:创建模型
model = Transformer(33,80,1,4,256,1024,has_inputs=True,src_pad_idx=0,trg_pad_idx=0).to(device)
#加载权重
checkpoint_path = 'logs/transformer_tts/epoch-78.pt'
if os.path.exists(checkpoint_path):
    packages = torch.load(checkpoint_path)
    model.load_state_dict(packages['model'],strict=False)
    start_epoch = packages['epoch']
    global_step = packages['step']
    print("loading pretrained weights!")
else:
    start_epoch = 0
    global_step = 0


optimizer = LRScheduler(filter(lambda x: x.requires_grad, model.parameters()),512, 16000, step=global_step)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("# of parameters: {}".format(num_params))


#model.t_pos_embedding.weight

# 2:加载数据
from torch.utils.data import Dataset,DataLoader
from datasets.thchs30_tts_dataset import Thchs30Dataset,collate_fn
dataset_root = "path to thchs30 dataset"
train_dataset = Thchs30Dataset(dataset_root,'train')
dev_dataset = Thchs30Dataset(dataset_root,'dev')
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=4,collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset,batch_size=32,shuffle=True,num_workers=4,collate_fn=collate_fn)
train_iter = iter(train_loader)
batch = next(train_iter)


# 3:开始训练
epochs = 200

for epoch in range(start_epoch,epochs):
    #训练
    model = model.train(True)
    train_loss_list = []
    train_mel_loss_list = []
    train_stop_loss_list = []
    for i,batch in enumerate(train_loader):
        inputs,targets,t_mask = [data.to(device) for data in batch]
        predMel,enc_dec_attn_list = model(inputs,targets,t_mask)
        mel_loss = nn.L1Loss()(predMel,targets)
        #stop_loss = nn.L1Loss()(stopMel,t_mask.float().unsqueeze(-1))
        loss = mel_loss# + stop_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        logs_writer.add_scalar('train/loss',loss.item(),global_step)
        #logs_writer.add_scalar('train/mel_loss',mel_loss.item(),global_step)
        #logs_writer.add_scalar('train/stop_loss',stop_loss.item(),global_step)
        train_loss_list.append(loss.item())
        #train_mel_loss_list.append(mel_loss.item())
        #train_stop_loss_list.append(stop_loss.item())
        if global_step%50==0:
            avg_loss = np.mean(train_loss_list[-50:])
            #avg_mel_loss = np.mean(train_mel_loss_list[-50:])
            #avg_stop_loss = np.mean(train_stop_loss_list[-50:])
            print("{}/{} loss:{:.5f} avg_loss:{:.5f}"# avg_mel_loss:{:.3f} avg_stop_loss:{:.3f}"
                   .format(global_step,epoch,loss.item(),avg_loss))#,avg_mel_loss,avg_stop_loss))
            # print("pinyin:",train_dataset.id2pyin(inputs[0].cpu().numpy()))
        if global_step%1000==0:
            save_path = os.path.join(logs_root,"step-%d.pt" % global_step)
            packages = {
              'model':model.state_dict(),
              'epoch':epoch,
              'step':global_step,
              'train_avg_loss':avg_loss}
            torch.save(packages,save_path)
    start_epoch += 1
    #评估
    if epoch%3==0:
      dev_loss_list = []
      #dev_mel_loss_list = []
      #dev_stop_loss_list = []
      for index,batch in enumerate(dev_loader):
        with torch.no_grad():
          inputs,targets,t_mask = [data.to(device) for data in batch]
          predMel,enc_dec_attn_list = model(inputs,targets,t_mask)
          mel_loss = nn.L1Loss()(predMel,targets)
          #stop_loss = nn.L1Loss()(stopMel,t_mask.float().unsqueeze(-1))
          loss = mel_loss# + stop_loss
          dev_loss_list.append(loss.item())
          #dev_mel_loss_list.append(mel_loss.item())
          #dev_stop_loss_list.append(stop_loss.item())
          if index>50:break
      avg_loss = np.mean(dev_loss_list)
      #avg_mel_loss = np.mean(dev_mel_loss_list)
      #avg_stop_loss = np.mean(dev_stop_loss_list)
      logs_writer.add_scalar('dev/avg_loss',avg_loss,epoch)
      #logs_writer.add_scalar('dev/avg_mel_loss',avg_mel_loss,epoch)
      #logs_writer.add_scalar('dev/avg_stop_loss',avg_stop_loss,epoch)
      print("dev_epoch:{} loss:{:.5f} avg_loss:{:.5f}"# avg_mel_loss:{:.3f} avg_stop_loss:{:.3f}"
             .format(epoch,loss.item(),avg_loss))#,avg_mel_loss,avg_stop_loss))
      save_path = os.path.join(logs_root,"epoch-%d.pt" % epoch)
      packages = {
        'model':model.state_dict(),
        'epoch':epoch,
        'step':global_step,
        'dev_avg_loss':avg_loss}
      torch.save(packages,save_path)



# #查看注意力图
# import matplotlib.pyplot as plt
# bi = np.random.randint(0,2)
# plt.figure()
# for layer_idx,enc_dec_attn in enumerate(enc_dec_attn_list):
#   for head_idx,attn in enumerate(enc_dec_attn[bi].detach().cpu().numpy()):
#    idx = layer_idx*8+head_idx+1
#    plt.subplot(4,8,idx)
#    plt.imshow(attn)
#
#
# plt.show()



