'''
Author: sigmoid
Description: 基于传统注意力模块的测试脚本
Email: 595495856@qq.com
Date: 2021-03-06 19:38:18
LastEditTime: 2021-03-06 20:10:26
'''
import math
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

from dataset import MERData
from model_classic import Encoder, Decoder
from config import cfg
from utils.util import get_all_dist, load_dict, custom_dset, collate_fn_double
torch.backends.cudnn.benchmark = False

# 配置参数
valid_datasets = ['data/valid.pkl', 'data/CROHME2016/label/test_caption_2014.txt']
dictionaries = 'data/CROHME2016/label/dictionary.txt'
result_path = "results/recognition.txt"

Imagesize = 500000
batch_size_t = 2
maxlen = 70
maxImagesize = 100000
hidden_size = 256

worddicts = load_dict(dictionaries)  #token 2 id
worddicts_r = [None] * len(worddicts)   #id 2 token
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk 

# 数据读取处理
test, test_label, uidList = MERData(
                                valid_datasets[0], valid_datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

image_test = custom_dset(test, test_label)

test_loader = torch.utils.data.DataLoader(
    dataset = image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn_double,
    num_workers = cfg.num_workers,
)

# 1. 加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(112)

encoder = encoder.cuda()
decoder = decoder.cuda()

encoder.load_state_dict(torch.load('checkpoints/encoder_classic.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_classic.pkl'))

encoder.eval()
decoder.eval()

# 评估参数
total_dist = 0 # 统计所有的序列的总编辑距离
total_label = 0 # 统计所有序列的总长度
total_line = 0 # 统计一共有多少个序列
total_line_rec = 0 # 统计识别正确的序列
error1, error2, error3 = 0, 0, 0

fw = open(result_path, 'w') # 保存识别结果
# 2. 开始评估
for step_t, (x_t, y_t, batch_list) in enumerate(test_loader): 
    # abandon <batch data
    if x_t.size()[0]<batch_size_t:
        break
    x_t = x_t.cuda()
    y_t = y_t.cuda()
    feat_t = encoder(x_t) # (bs, c, h, w) c=684

    # 1.init input
    decoder_input_t = torch.LongTensor([111]*batch_size_t).view(-1, 1).cuda()
    decoder_hidden_t = decoder.init_hidden(batch_size_t).cuda()
    # 2.reset coverage
    decoder.reset(batch_size_t, feat_t.size()) 

    prediction = torch.zeros(batch_size_t, maxlen)
    prediction_sub = []
    prediction_real = []    

    label_sub = []
    label_real = []

    # 处理标签
    m = torch.nn.ZeroPad2d((0, maxlen-y_t.size()[1], 0, 0))
    y_t = m(y_t)

    for i in range(maxlen):
        decoder_output_t, decoder_hidden_t, _ = decoder(decoder_input_t, decoder_hidden_t, feat_t)
        
        topv, topi = torch.max(decoder_output_t, 1) 
        if torch.sum(topi)==0: # 一个批次中所有序列都预测完成
            break
        
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t, 1)
        
        # prediction
        prediction[:, i] = decoder_input_t.flatten()

    for i in range(batch_size_t):
        uid = uidList[batch_list[i]]
        for j in range(maxlen):
            if int(prediction[i][j]) == 0:
                break
            else:
                prediction_sub.append(int(prediction[i][j]))
                prediction_real.append(worddicts_r[int(prediction[i][j])])

        if len(prediction_sub) < maxlen: #不足后面填0
            prediction_sub.append(0)

        for k in range(y_t.size()[1]):
            if int(y_t[i][k]) == 0:
                break
            else:
                label_sub.append(int(y_t[i][k]))
                label_real.append(worddicts_r[int(y_t[i][k])])
        label_sub.append(0)

        # 评价指标
        dist, llen, sub, ins, dls = get_all_dist(label_sub, prediction_sub)  
        wer_step = float(dist) / llen

        if len(prediction_sub)==len(label_sub) : # 计算error1, error2
            e = 0
            for r, l in zip(prediction_sub, label_sub):
                if r!=l :
                    e += 1
                if e>3: break # 超过3个直接跳出
            if e==1:
                error1 += 1
            elif e==2:
                error2 += 1
            elif e==3:
                error3 += 1
        wer_step = float(dist) / llen

        total_dist += dist
        total_label += llen
        total_line += 1 

        if dist == 0:
            total_line_rec = total_line_rec + 1
    
        print('step is %d' % (step_t))
        print('prediction is ')
        print(prediction_real)
        print('the truth is')
        print(label_real)
        print('the wer is %.5f' % (wer_step))
        
        # save predict result
        fw.write(uid+'\t')
        fw.write(' '.join(prediction_real)+'\n')

        label_sub = []
        prediction_sub = []
        label_real = []
        prediction_real = []
fw.close()

wer = float(total_dist) / total_label
print("{}/{}".format(total_line_rec, total_line))

exprate = float(total_line_rec) / total_line
error1 += total_line_rec
error2 += error1
error3 += error2

# 打印评测结果
print('{}/{}'.format(total_line_rec, total_line))
print('ExpRate is {:.4f}'.format(exprate))

print('error1 nums: {}, error2 nums: {}, error3 nums: {}'.format(error1, error2, error3))
print('error1 is {:.4f}, error2 is {:.4f}, error3 is {:.4f}'.format((error1)/total_line, error2/total_line, error3/total_line))

print('wer is {:.4f}'.format(wer))
