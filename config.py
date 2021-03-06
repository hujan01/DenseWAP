'''
Author: sigmoid
Description: 配置文件
Email: 595495856@qq.com
Date: 2021-02-21 14:40:56
LastEditTime: 2021-02-21 15:07:15
'''
class Config():  
    seed = 2020
    
    model_idx = 0

    datasets = ['data/train.pkl', 'data/CROHME2016/label/train_caption.txt']
    valid_datasets = ['data/valid.pkl', 'data/CROHME2016/label/test_caption_2014.txt']
    dictionaries = 'data/CROHME2016/label/dictionary.txt'

    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000 
    maxImagesize = 100000

    maxlen = 70
    hidden_size = 256
    num_class = 112

    num_epoch = 80
    lr = 0.0001
    batch_size = 4
    batch_size_t = 4
    teacher_forcing_ratio = 0.8
 
    num_workers = 4

cfg = Config()