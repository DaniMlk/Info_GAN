# %matplotlib inline
#Import a ton of stuff
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from layers import Conv2d, ConvTranspose2d, Linear
from InfoGAN import InfoGAN
from logger import Logger
import os.path
import random
import scipy
import scipy.ndimage
import pdb

rnd = [20,30,40,50]
rnd_real = [70,80,90,100]
picture_size = 320
batch_size = 169
color = 3
in_size = 24
out_size = 32
iterate_size = 13
num_per_pic = 169
total_pic = 1
total_pic_test = 1
loss_train =[]
loss_test =[]
filter_k = 64
num_pic_show = 2
is_best = True
best_loss = 1000
root = "/home/dani/GAN_32/Dataset_small"
count = 0
logger = Logger('./logs')
loss_t_train = 0
epochSize = 50
idx = []
vis = 1
temp = 0
# b = 0 # baraye jelo raftan rooye axs-haye ye
# l = 0
###################################load_dataset#################################
def data_loader(b, l):
    cache_file_name = root+"/cache/batch_"+(str(b).rjust(4,'0'))+"seg"+(str(l+1).rjust(4,'0'))+".pickle"
    if os.path.isfile(cache_file_name):
        data_32, data_24, input_24, idx, idx_real= pickle.load(open(cache_file_name,'rb'))
    else :
        h = 1
        idx = []
        idx_real = []
        choice_rnd=random.choice(rnd)
        choice_rnd_real = random.choice(rnd_real)
        data_32 = scipy.ndimage.imread(root+"/32_32_"+str(choice_rnd)+"/pic"+(str(b).rjust(4,'0'))+"seg"+(str(l+1).rjust(4,'0'))+".jpg").reshape(1,32,32,color)
        comp_class = np.multiply(np.ones((1,32,32,1)),choice_rnd/100)
        input_24 = data_32[:,4:28,4:28,:]
        data_32 = np.concatenate((data_32,comp_class),axis=3)
        idx.append(int(choice_rnd/10-2))
        data_24 = scipy.ndimage.imread(root+"/24_24_"+str(choice_rnd_real)+"/pic"+(str(b).rjust(4,'0'))+"seg"+(str(l+1).rjust(4,'0'))+".jpg").reshape(1,24,24,color)
        idx_real.append(int(choice_rnd_real/10-7))
        x1 = int(batch_size / num_per_pic)
        for m in range(max(x1,1)):
            for k in range(min(batch_size,num_per_pic)-h):
                choice_rnd=random.choice(rnd)
                choice_rnd_real = random.choice(rnd_real)
                data_load_32 = scipy.ndimage.imread(root+"/32_32_"+str(choice_rnd)+"/pic"+(str(b+m).rjust(4,'0'))+"seg"+(str(l+k+1+h).rjust(4,'0'))+".jpg").reshape(1,32,32,color)
                comp_class = np.multiply(np.ones((1,32,32,1)),choice_rnd/100)
                data_load_24  = scipy.ndimage.imread(root+"/24_24_"+str(choice_rnd_real)+"/pic"+(str(b+m).rjust(4,'0'))+"seg"+(str(l+k+1+h).rjust(4,'0'))+".jpg").reshape(1,24,24,color)
                idx_real.append(int(choice_rnd_real/10-7))
                input_24 = np.concatenate((input_24,data_load_32[:,4:28,4:28,:]),axis=0)
                data_load_32 = np.concatenate((data_load_32,comp_class),axis=3)
                idx.append(int(choice_rnd/10-2))
                data_32 = np.concatenate((data_32,data_load_32),axis=0)
                data_24 = np.concatenate((data_24,data_load_24),axis=0)
            h = 0
        b = b + max(x1,1) - 1
        print("no cached file...")
        # pdb.set_trace()
        pickle.dump([data_32, data_24, input_24, idx, idx_real], open(cache_file_name,'wb'))

    # print(data_32.shape)
    # print(np.asarray(idx))

    ################imortant note: I used idx instead of idx_real, after creat dataset change it############
    y_train = np.zeros((data_32.shape[0],4),dtype=np.uint8)
    y_train[np.arange((np.asarray(idx_real)).shape[0]), np.asarray(idx_real)] = 1
    shuffle_train_fake = np.random.permutation(data_32.shape[0])
    shuffle_train_real = np.random.permutation(data_24.shape[0])
    data_32_th = torch.from_numpy(data_32[shuffle_train_fake]).permute(0,3,1,2)
    input_24_th = torch.from_numpy(input_24[shuffle_train_fake]).permute(0,3,1,2)
    y_train_th = torch.from_numpy(y_train[shuffle_train_real])
    data_24_th = torch.from_numpy(data_24[shuffle_train_real]).permute(0,3,1,2)
    return data_32_th, data_24_th, input_24_th, idx, idx_real, shuffle_train_fake, shuffle_train_real, y_train_th


data_32_read,data_24_read,input_24_read,idx_read,idx_real_read,shuffle_train_fake,shuffle_train_real,y_train_th=data_loader(b=0,l=0)
#################################end_of_load_dataset############################
################################creat_our_network###############################
from model import *
c1_len = 4
c2_len = 0
c3_len = 0
embedding_len = 128

gen = G().cuda()
FE = FrontEnd().cuda()
q = Q().cuda()
dis = D().cuda()
gan = InfoGAN(gen,FE,q,dis,embedding_len,c1_len,data_loader)
gan.train_all(total_pic)
