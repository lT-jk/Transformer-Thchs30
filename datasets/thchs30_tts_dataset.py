import os
import glob
import torch
import numpy as np
from utils import audio
from pypinyin import pinyin,Style


class Thchs30Dataset:
    def __init__(self,dataset_root,dataset_type,outputs_per_step=1):
        assert dataset_type in ['train','dev','test']
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.wav_txt_list = []
        self.wav_path_list = glob.glob(os.path.join(dataset_root,dataset_type,'*.wav'))
        for wav_path in self.wav_path_list:
            wav_name = wav_path.split('/')[-1]
            txt_path = os.path.join(self.dataset_root,'data',wav_name+'.trn')
            with open(txt_path) as f:
                text = ''.join(f.readlines()[0].strip().split(' '))
            self.wav_txt_list.append([wav_path,text])
        self.symbols = ' 01234abcdefghijklmnopqrstuvwxyz'
        self.char2index = {key:value+1 for value,key in enumerate(self.symbols)}
        self.index2char = {value+1:key for value,key in enumerate(self.symbols)}
        self.outputs_per_step = outputs_per_step
    def __len__(self):
        return len(self.wav_txt_list)
    def __getitem__(self,index):
        wav_path,text = self.wav_txt_list[index]
        mel = self.wav2spec(wav_path)
        ids = self.text2id(text)
        mel_tensor = torch.from_numpy(mel)
        ids_tensor = torch.from_numpy(ids)
        return ids_tensor,mel_tensor
    def text2id(self,text):
        text = text.strip(" \n")
        pyin = ' '.join([py[0] for py in pinyin(text, style=Style.TONE2, heteronym=False)])
        ids = np.array([self.char2index[c] for c in pyin if c in self.char2index],dtype='int32')
        return ids
    def id2pyin(self,ids):
        pyin = ''.join([self.index2char[_id] for _id in list(ids) if _id in self.index2char])
        return pyin
    def wav2spec(self,wav_path):
        wav = audio.load_wav(wav_path)
        spec = audio.melspectrogram(wav).astype(np.float32)
        spec = spec.transpose()
        feat_size = spec.shape[1]
        pad_spec = np.zeros([(len(spec)+self.outputs_per_step-1)//self.outputs_per_step*self.outputs_per_step,feat_size],dtype='float32')
        pad_spec[:len(spec)] = spec 
        return pad_spec.reshape([-1,self.outputs_per_step*feat_size])


def collate_fn(batch):
    def func0(p):
        return p[0].size(0)
    def func1(p):
        return p[1].size(0)
    #batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    longest_input = max(batch,key=func0)[0]
    longest_target = max(batch,key=func1)[1]
    minibatch_size = len(batch)
    max_input_length = longest_input.size(0)
    max_mel_length = longest_target.size(0)
    mel_feat_size = longest_target.size(1)
    inputs = torch.zeros([minibatch_size,max_input_length],dtype=torch.int64)
    mels = torch.zeros([minibatch_size,max_mel_length,mel_feat_size],dtype=torch.float32)
    mel_mask = torch.zeros([minibatch_size,max_mel_length],dtype=torch.int64)
    for idx in range(minibatch_size):
        ids,mel = batch[idx]
        inputs[idx].narrow(0,0,len(ids)).copy_(ids)
        #inputs[idx][len(ids)] = 2
        mels[idx].narrow(0,0,len(mel)).copy_(mel)
        mel_mask[idx][:len(mel)] = 1
    return inputs,mels,mel_mask



#测试
if __name__ == "__main__":
    from torch.utils.data import Dataset,DataLoader
    dataset_root = "path to thchs30 dataset"
    dataset_type = "train"
    dataset = Thchs30Dataset(dataset_root,dataset_type)
    len(dataset)
    ids,mel = dataset.__getitem__(0)
    data_loader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=1,collate_fn=collate_fn)
    data_iter = iter(data_loader)
    inputs,mels,mel_mask = next(data_iter)
