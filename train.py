'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2021-02-21 14:12:39
LastEditTime: 2021-02-22 20:43:44
'''
import os, math, time
import random  
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from tensorboardX import SummaryWriter

from model import Encoder, Decoder
from dataset import MERData
from config import cfg  
from utils.util import collate_fn_double, custom_dset, cmp_result, load_dict

# set random seed
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

exprate = 0 # save model flag
best_wer = 2**31

logdir = 'logs/model' + str(cfg.model_idx)

# log
writer = SummaryWriter(logdir)

# load dictionary
worddicts = load_dict(cfg.dictionaries)
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

# load train data and test data
train, train_label, train_list = MERData(
                                cfg.datasets[0], cfg.datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=cfg.batch_Imagesize, maxlen=cfg.maxlen, maxImagesize=cfg.maxImagesize
                            )
len_train = len(train)

fw = open('valid_image_list.txt', 'w')
for image in train_list:
    fw.write(image)
    fw.write('\n')
fw.close()

test, test_label, test_list = MERData(
                                cfg.valid_datasets[0], cfg.valid_datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=cfg.batch_Imagesize, maxlen=cfg.maxlen, maxImagesize=cfg.maxImagesize
                          )
len_test = len(test)

image_train = custom_dset(train, train_label)
image_test = custom_dset(test, test_label)

train_loader = torch.utils.data.DataLoader(
    dataset = image_train,
    batch_size = cfg.batch_size,
    shuffle = True,
    collate_fn = collate_fn_double,
    num_workers = cfg.num_workers,
    )
    
test_loader = torch.utils.data.DataLoader(
    dataset = image_test,
    batch_size = cfg.batch_size_t,
    shuffle = True,
    collate_fn = collate_fn_double,
    num_workers = cfg.num_workers,
)

# 加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(cfg.num_class)

# load pre-train
# encoder_dict = torch.load('checkpoints/encoder_pre.pkl')
# encoder.load_state_dict(encoder_dict)
# decoder_dict = torch.load('checkpoints/attn_decoder_coverage_pre.pkl')
# decoder.load_state_dict(decoder_dict)

encoder = encoder.cuda()
decoder = decoder.cuda()

# loss, optimizer
criterion = nn.CrossEntropyLoss().cuda()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=10e-3)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=10e-3)
scheduler_encoder = optim.lr_scheduler.MultiStepLR(encoder_optimizer, [30, 50, 60], gamma=0.5)
scheduler_decoder = optim.lr_scheduler.MultiStepLR(encoder_optimizer, [30, 50, 60], gamma=0.5)

for epoch in range(1, cfg.num_epoch+1):
    ud_epoch = time.time()
    scheduler_encoder.step()
    scheduler_decoder.step()
    running_loss=0
    whole_loss = 0

    encoder.train(mode=True)
    decoder.train(mode=True)

    # 开始训练 
    for step, (x, y, _) in enumerate(train_loader):
        if x.size()[0] < cfg.batch_size:  
            break
        x = x.cuda()
        y = y.cuda()
        # ----encoder----
        feat = encoder(x)
        
        # ----decoder----
        decoder_input = torch.LongTensor([108]*cfg.batch_size).view(-1, 1).cuda()       
        decoder_hidden = decoder.init_hidden(cfg.batch_size).cuda()
        
        # reset coverage
        decoder.reset(cfg.batch_size, feat.size())

        target_length = y.size()[1]
        loss = 0

        # whether use tf 
        use_teacher_forcing = True if random.random() < cfg.teacher_forcing_ratio else False
        flag_z = [0]*cfg.batch_size
        
        if use_teacher_forcing:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, feat )
    
                y = y.unsqueeze(0)
                for i in range(cfg.batch_size):
                    if int(y[0][i][di]) == 0:
                        flag_z[i] = flag_z[i]+1
                        if flag_z[i] > 1:
                            continue
                        else:
                            loss += criterion(decoder_output[i].view(1, -1), y[:,i,di])
                    else:
                        loss += criterion(decoder_output[i].view(1, -1), y[:,i,di])

                if int(y[0][0][di]) == 0:
                    break
                decoder_input = y[:,:,di].transpose(1, 0)
                y = y.squeeze(0)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        else:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, feat)

                topv, topi = torch.max(decoder_output, 1)
                decoder_input = topi
                decoder_input = decoder_input.view(cfg.batch_size, 1)

                y = y.unsqueeze(0)
                for k in range(cfg.batch_size):
                    if int(y[0][k][di]) == 0:
                        flag_z[k] = flag_z[k]+1
                        if flag_z[k] > 1:
                            continue
                        else:
                            loss += criterion(decoder_output[k].view(1, -1), y[:,k,di])
                    else:
                        loss += criterion(decoder_output[k].view(1, -1), y[:,k,di])
                y = y.squeeze(0)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
    
        if step % 20 == 19:
            pre = ((step+1)/len_train)*100*cfg.batch_size
            whole_loss += running_loss
            running_loss = running_loss/(cfg.batch_size*20)
            print('epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch, scheduler_decoder.get_lr()[0], cfg.teacher_forcing_ratio, cfg.batch_size, pre, running_loss))
            running_loss = 0

    loss_all_out = whole_loss / len_train
    writer.add_scalar('loss', loss_all_out, epoch)
    ud_epoch = (time.time()-ud_epoch)/60
    print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))
    print("epoch cost time...", ud_epoch)

    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    whole_loss_t = 0

    # ------- valid ----------
    encoder.eval()
    decoder.eval()
    print('Now, begin testing!!')

    for step_t, (x_t, y_t, _) in enumerate(test_loader):
        if x_t.size()[0] < cfg.batch_size:  
            break
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]

        print('testing for %.3f%%'%(step_t*100*cfg.batch_size_t/len_test), end='\r')
        
        x_t = x_t.cuda()
        y_t = y_t.cuda()
        
        feat_t = encoder(x_t)
        # init input
        decoder_input_t = torch.LongTensor([108]*cfg.batch_size_t).view(-1, 1).cuda()
        decoder_hidden_t = decoder.init_hidden(cfg.batch_size_t).cuda()     
        # reset coverage  
        decoder.reset(cfg.batch_size_t, feat_t.size())

        prediction = torch.zeros(cfg.batch_size_t, cfg.maxlen)
        prediction_sub = []
        label_sub = []

        m = torch.nn.ZeroPad2d((0, cfg.maxlen-y_t.size()[1], 0, 0))
        y_t = m(y_t)
        for i in range(cfg.maxlen):
            decoder_output_t, decoder_hidden_t, _ = decoder(decoder_input_t, decoder_hidden_t, feat_t)
            topv, topi = torch.max(decoder_output_t, 1)
            if torch.sum(topi) == 0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(cfg.batch_size_t, 1)

            # prediction
            prediction[:, i] = decoder_input_t.flatten()

        for i in range(cfg.batch_size_t):
            for j in range(cfg.maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))

            if len(prediction_sub)<cfg.maxlen: #不足后面填0
                prediction_sub.append(0)

            for k in range(y_t.size()[1]):
                if int(y_t[i][k]) ==0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
            total_dist += dist
            total_label += llen
            total_line += 1
            
            if dist == 0:
                total_line_rec = total_line_rec+ 1

            label_sub = []
            prediction_sub = []

    print('total_line_rec is', total_line_rec)
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    print('wer is %.5f' % (wer))
    print('sacc is %.5f ' % (sacc)) # ExpRate
    writer.add_scalars('metric', {'ExpRate': sacc, 'WER': wer}, epoch)
    # save model
    if (sacc > exprate):
        exprate = sacc
        best_wer = wer
        print('currect ExpRate:{}'.format(exprate))
        print("saving the model....")
        torch.save(encoder.state_dict(), 'checkpoints/encoder_coverage.pkl')
        torch.save(decoder.state_dict(), 'checkpoints/attn_decoder_coverage.pkl')
        print("done")
    else:
        print('the best is %f' % (exprate))
        print('the wer is %f' % (best_wer))
        print('the loss is bigger than before,so do not save the model')

writer.close()
