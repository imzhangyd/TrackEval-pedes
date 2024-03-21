'''
This script handles the training process.
'''
import glob
import argparse
import time

# import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os
import torch.nn as nn
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2

from transformer.Optim import ScheduledOptim


from Dataset_dropGT import func_getdataloader_dropGT
from Dataset import func_getdataloader
import pandas as pd
import numpy as np
import torch
from transformer.Models import Transformer,Transformer_sep_pred_wocusum
from Dataset_oridif_match import func_getdataloader_oridif_match
from Dataset_oridif import func_getdataloader_oridif
from torch import nn
import gurobipy as grb
import time
from treelib import Node, Tree

import numpy as np
import pandas as pd

import os
import shutil


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, total_loss_prob,total_loss_dist = 0, 0, 0
    total_accuracy = 0
    total_accuracy_5 = 0

    desc = '  - (Training)   '
    Loss_func = nn.CrossEntropyLoss()
    # Loss_func = nn.BCELoss()
    Loss_func_dist = nn.MSELoss()
    # for test in training_data:
    #     print(test)
    
    # first = 0
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False): #加载的时候就自动向量化了
        
        # prepare data 准备数据
        src_seq = batch[0].float().to(device)
        trg_seq = batch[1].float().to(device)
        label_shift = batch[2].float().to(device)#.cumsum(dim=-2)
        # label_shift[:,:,:-1] = 1
        label_prob = batch[3].to(device)
        passed_patches = batch[4].to(device).half()
        future_patches = batch[5].to(device).half()

        # forward
        optimizer.zero_grad()
        pred_shift,pred_prob = model(src_seq, trg_seq) #前向推理 希望输入前 和 候选 输出cost矩阵
        
        loss_prob = Loss_func(pred_prob,label_prob)
        loss_dist = Loss_func_dist(
            pred_shift[(label_shift[:,:,-1]>0)][:,:-1],
            label_shift[(label_shift[:,:,-1]>0)][:,:-1])  #修改计算loss的方式
        loss = loss_prob+loss_dist
        # loss = loss_prob
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss_prob += loss_prob.item()*pred_shift.shape[0]
        total_loss_dist += loss_dist.item()*pred_shift.shape[0]
        total_loss += loss.item()*pred_shift.shape[0]
        # 计算accuracy
        pred_num = torch.argmax(pred_prob,1)
        accracy = np.sum((pred_num==label_prob).cpu().numpy())
        total_accuracy += accracy

        up = (label_prob - label_prob%5)
        down = up+5
        accuracy5 = np.sum((((pred_num-up)>=0)&(((pred_num-down) <0))).cpu().numpy())
        total_accuracy_5 += accuracy5

    return total_loss_prob,total_loss_dist,total_loss,total_accuracy,total_accuracy_5


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, total_loss_prob,total_loss_dist = 0, 0, 0
    total_absdist = 0
    total_accuracy = 0
    total_accuracy_5 = 0

    desc = '  - (Validation) '
    Loss_func = nn.CrossEntropyLoss()
    # Loss_func = nn.BCELoss()
    Loss_func_dist = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = batch[0].float().to(device)
            trg_seq = batch[1].float().to(device)
            label_shift = batch[2].float().to(device)#.cumsum(dim=-2)
            # label_shift[:,:,-1] = 1
            label_prob = batch[3].to(device)
            

            # forward
            pred_shift,pred_prob = model(src_seq, trg_seq) #前向推理 希望输入前 和 候选 输出cost矩阵
            
            loss_prob = Loss_func(pred_prob,label_prob)
            loss_dist = Loss_func_dist(
                pred_shift[(label_shift[:,:,-1]>0)][:,:-1],
                label_shift[(label_shift[:,:,-1]>0)][:,:-1])
            loss = loss_prob+loss_dist
            # loss = loss_prob

            total_loss_prob += loss_prob.item()*pred_shift.shape[0]
            total_loss_dist += loss_dist.item()*pred_shift.shape[0]
            total_loss += loss.item()*pred_shift.shape[0]
            # 计算accuracy
            pred_num = torch.argmax(pred_prob,1)
            accracy = np.sum((pred_num==label_prob).cpu().numpy())
            total_accuracy += accracy

            up = (label_prob - label_prob%5)
            down = up+5
            accuracy5 = np.sum((((pred_num-up)>=0)&(((pred_num-down) <0))).cpu().numpy())
            total_accuracy_5 += accuracy5

    return total_loss_prob,total_loss_dist,total_loss,total_accuracy,total_accuracy_5

def eval_epoch_calabsdist(model, validation_data, device, opt,datamean,datastd):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, total_loss_prob,total_loss_dist = 0, 0, 0
    total_absdist = 0
    total_accuracy = 0
    total_accuracy_5 = 0

    desc = '  - (Validation) '
    Loss_func = nn.CrossEntropyLoss()
    # Loss_func = nn.BCELoss()
    Loss_func_dist = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = batch[0].float().to(device)
            trg_seq = batch[1].float().to(device)
            label_shift = batch[2].float().to(device)#.cumsum(dim=-2)
            # label_shift[:,:,-1] = 1
            label_prob = batch[3].to(device)
            

            # forward
            pred_shift,pred_prob = model(src_seq, trg_seq) #前向推理 希望输入前 和 候选 输出cost矩阵
            # replace the pred with last shift
            # pred_shift = torch.cat((src_seq[:,-1:,:],src_seq[:,-1:,:]),1)
            loss_prob = Loss_func(pred_prob,label_prob)
            loss_dist = Loss_func_dist(
                pred_shift[(label_shift[:,:,-1]>0)][:,:-1],
                label_shift[(label_shift[:,:,-1]>0)][:,:-1])
            loss = loss_prob+loss_dist
            # loss = loss_prob
            # if opt.cal_absdist:
            pastabsseq = batch[-1][:,-1:,:]
            predpos = pred_shift[:,:,:-1].cpu().numpy()*datastd.reshape(1,1,2)+ \
                datamean.reshape(1,1,2) + pastabsseq[:,0:,:-1].numpy()
            gtpos = label_shift[:,:,:-1].cpu().numpy()*datastd.reshape(1,1,2)+ \
                datamean.reshape(1,1,2) + pastabsseq[:,0:,:-1].numpy()

            absdist = Loss_func_dist(
                torch.Tensor(predpos[(label_shift.cpu().numpy()[:,:,-1]>0)]),
                torch.Tensor(gtpos[(label_shift.cpu().numpy()[:,:,-1]>0)])
            )
            total_absdist += absdist.item()*pred_shift.shape[0]
                
            total_loss_prob += loss_prob.item()*pred_shift.shape[0]
            total_loss_dist += loss_dist.item()*pred_shift.shape[0]
            total_loss += loss.item()*pred_shift.shape[0]
            # 计算accuracy
            pred_num = torch.argmax(pred_prob,1)
            accracy = np.sum((pred_num==label_prob).cpu().numpy())
            total_accuracy += accracy

            up = (label_prob - label_prob%5)
            down = up+5
            accuracy5 = np.sum((((pred_num-up)>=0)&(((pred_num-down) <0))).cpu().numpy())
            total_accuracy_5 += accuracy5

    return total_loss_prob,total_loss_dist,total_loss,total_accuracy,total_accuracy_5,total_absdist

def  train(model, training_data,traindata_len, validation_data,valdata_len, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")

        now = int(round(time.time()*1000))
        nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard/'+nowname))

        # 写下参数列表
        log_param = os.path.join(opt.output_dir, nowname+'.log')
        thisfold = open(log_param,'a')
        thisfold.write('n_layer:{}\n'.format(opt.n_layers))
        thisfold.write('n_head:{}\n'.format(opt.n_head))
        thisfold.write('d_k/v:{}\n'.format(opt.d_k))
        thisfold.write('ffn_inner_d:{}\n'.format(opt.d_inner_hid))
        thisfold.write('warmup:{}\n'.format(opt.n_warmup_steps))
        thisfold.write('batchsize:{}\n'.format(opt.batch_size))
        thisfold.close()


    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy\n')
        log_vf.write('epoch,loss,accuracy\n')

    def print_performances(header, loss,accu, start_time, lr):
        print('  - {header:12} loss: {loss:3.4f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))
    # 输出初始权重查看
    # printnum = 0
    # for name, parms in model.named_parameters(): 
    #         print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
    #         ' -->grad_value:',parms.grad)
    #         printnum += 1
    #         if printnum > 5:
    #             break

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss_prob,train_loss_dist,train_loss,train_accuracy,train_accuracy_5 = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)


        # 输出每个epoch的权重
        # printnum = 0
        # for name, parms in model.named_parameters(): 
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #     ' -->grad_value:',parms.grad,'-->weight_value:',parms.detach().cpu().numpy())
        #     printnum += 1
        #     if printnum > 5:
        #         break

        # print(len(training_data))
        train_loss /= traindata_len
        train_loss_prob /= traindata_len
        train_loss_dist /= traindata_len

        train_accuracy_5 /= traindata_len
        train_acc = train_accuracy/traindata_len
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        # print('Loss:{}'.format(train_loss))
        print_performances('Training', train_loss, train_acc, start, lr)

        start = time.time()
        valid_loss_prob,valid_loss_dist,valid_loss,valid_accuracy,valid_accuracy_5 \
            = eval_epoch(model, validation_data, device, opt)
        valid_loss_prob /= valdata_len
        valid_loss_dist /= valdata_len
        valid_loss /= valdata_len
        # valid_ppl = math.exp(min(valid_loss, 100))
        valid_accuracy_5 /= valdata_len
        valid_acc = valid_accuracy/valdata_len
        print_performances('Validation', valid_loss, valid_acc, start, lr)
        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*0)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = nowname+'.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                accu=100*train_acc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                accu=100*valid_acc))

        if opt.use_tb:
            tb_writer.add_scalar('loss/train',train_loss, epoch_i)
            tb_writer.add_scalar('loss/val',valid_loss, epoch_i)
            tb_writer.add_scalar('loss/train_prob',train_loss_prob, epoch_i)
            tb_writer.add_scalar('loss/val_prob',valid_loss_prob, epoch_i)
            tb_writer.add_scalar('loss/train_dist',train_loss_dist, epoch_i)
            tb_writer.add_scalar('loss/val_dist',valid_loss_dist, epoch_i)

            tb_writer.add_scalar('accuracy/train',train_acc, epoch_i)
            tb_writer.add_scalar('accuracy/val',valid_acc, epoch_i)
            tb_writer.add_scalar('accuracy/train_5',train_accuracy_5, epoch_i)
            tb_writer.add_scalar('accuracy/val_5',valid_accuracy_5, epoch_i)

            tb_writer.add_scalar('learning_rate', lr, epoch_i)

def eval(model, validation_data,valdata_len, device, opt,dmean=None,dstd=None):
    

    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    with open(log_valid_file, 'w') as log_vf:
        log_vf.write('epoch,loss,accuracy\n')

    def print_evalperformances(header, loss,distloss,valid_absdist,accu, start_time):
        print('  - {header:12} loss: {loss:3.4f}, distloss: {distloss:3.4f},valid_absdist:{valid_absdist:3.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss, distloss = distloss,valid_absdist=valid_absdist,
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    valid_losses = []
    start = time.time()
    valid_loss_prob,valid_loss_dist,valid_loss,valid_accuracy,valid_accuracy_5,valid_absdist = \
        eval_epoch_calabsdist(model, validation_data, device, opt, dmean,dstd)
    valid_loss_prob /= valdata_len
    valid_loss_dist /= valdata_len
    valid_loss /= valdata_len
    valid_absdist /= valdata_len
    # valid_ppl = math.exp(min(valid_loss, 100))
    valid_accuracy_5 /= valdata_len
    valid_acc = valid_accuracy/valdata_len
    print_evalperformances('Validation', valid_loss,valid_loss_dist,valid_absdist,valid_acc, start)
    valid_losses += [valid_loss]


    
    with open(log_valid_file, 'a') as log_vf:
        log_vf.write('{loss: 8.5f},{accu:3.3f}\n'.format(
            loss=valid_loss,
            accu=100*valid_acc))



def main1(
    n_layer_, n_head_, d_kv_, d_modle_, ffn_, warmup_,
    batch_,train_path,val_path,output_path,past,cand,near,
    traindatamean=None,traindatastd=None,inoutdim=3
    ):
    '''
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    # parser.add_argument(' -data_pkl', default='m30k_deen_shr.pkl')     # all-in-1 data pickle or bpe field
    parser.add_argument('-inoutdim', default=inoutdim)
    parser.add_argument('-train_path', default=train_path)   # bpe encoded data
    parser.add_argument('-val_path', default=val_path)     # bpe encoded data
    parser.add_argument('-epoch', type=int, default=60)
    parser.add_argument('-b', '--batch_size', type=int, default=batch_)
    parser.add_argument('-d_model', type=int, default=d_modle_) # 每个位置的向量
    ######################################################################
    parser.add_argument('-n_position',type=int,default=5000) # pass+cand的总长度 最长长度
    parser.add_argument('-len_established',type=int,default=past) # len established
    parser.add_argument('-len_future',type=int,default=cand) # len future
    parser.add_argument('-num_cand',type=int,default=near**cand) # len future
    ######################################################################
    parser.add_argument('-d_inner_hid', type=int, default=ffn_) #FFN的中间那层的全连接数目，据说与记忆容量相关
    parser.add_argument('-d_k', type=int, default=d_kv_) #k的维度 
    parser.add_argument('-d_v', type=int, default=d_kv_) #v的维度
    parser.add_argument('-n_head', type=int, default=n_head_) # 注意力头
    parser.add_argument('-n_layers', type=int, default=n_layer_) #encoder decoder数量
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=warmup_)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight',default = True)
    parser.add_argument('-proj_share_weight', default = True)
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
    parser.add_argument('-output_dir', type=str, default=output_path) #==========================输出文件
    parser.add_argument('-use_tb', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', default = False)
    parser.add_argument('-label_smoothing', default = True)
    parser.add_argument('-evalonly',default=False)
    parser.add_argument('-usedropGTdata',default=False)
    parser.add_argument('-checkpoint',
    default='./20221117_model_MOT17_auginput19dim/20221117_20_17_38.chkpt')
    # default='./20221115_model_MOT17_disturbpassed/20221115_09_11_11.chkpt')
    # parser.add_argument('-log',type=str,default='m30k_deen_shr')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model


    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        
    device = torch.device('cuda')

    #========= load model if evalonly =========#
    if opt.evalonly:
        loadmodelpa = {}
        loadmodelpa['model'] = opt.checkpoint
        transformer_ins = load_model(loadmodelpa,device)
        if opt.usedropGTdata:
            ins_loader_val,valdata = func_getdataloader_dropGT(
                txtfile=opt.val_path, batch_size=batch_,
                shuffle=True, num_workers=16,mean=traindatamean,std=traindatastd)
        else:
            ins_loader_val,valdata = func_getdataloader(
                txtfile=opt.val_path, batch_size=batch_, 
                shuffle=True, num_workers=16,mean=traindatamean,std=traindatastd)
        eval(transformer_ins, ins_loader_val,len(valdata), device, opt,valdata.mean,valdata.std)
    # =======train===========#
    else:
        #========= Loading Dataset =========#
        print('==>init dataLoader')
        batch_size = opt.batch_size
        ins_loader_train,traindata = func_getdataloader_oridif(
            txtfile=opt.train_path, batch_size=batch_, shuffle=True, \
                num_workers=16,mean=traindatamean,std=traindatastd)
        ins_loader_val,valdata = func_getdataloader_oridif(
            txtfile=opt.val_path, batch_size=batch_, shuffle=True, \
                num_workers=16,mean = traindatamean,std=traindatastd)

        testonedata = traindata[0]

        print('==>init transformer')
        transformer = Transformer_sep_pred_wocusum(
            n_passed = opt.len_established,
            n_future = opt.len_future,
            n_candi = opt.num_cand,
            n_position = opt.n_position,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout,
            inoutdim=opt.inoutdim)
        # transformer_ins = nn.DataParallel(transformer,device_ids=[0,1,2,3]).to(device)
        transformer_ins = transformer.to(device)
        print('==>init optimizer')
        optimizer = ScheduledOptim(
            optim.Adam(transformer_ins.parameters(), betas=(0.9, 0.98), eps=1e-09),
            opt.lr_mul, opt.d_model, opt.n_warmup_steps)
        print('==>start train')
        train(transformer_ins, ins_loader_train,len(traindata), ins_loader_val,len(valdata), optimizer, device, opt)
    
        traindatamean = traindata.mean
        traindatastd = traindata.std
    return traindatamean, traindatastd



def find_near(pdcontent,x,y):
# all_posi 已经找到了第 n 帧的所有的粒子位置，并记录了第几个粒子和序列的第几位
    # 下面就增加一列 是到目标点的距离
    # x = 226.508
    # y = 62.934
    # pdcontent = pdcontent.drop_duplicates(subset=['pos_x','pos_y'])
    # 不去重可能导致预测结果有两个一样的，但是是不同的det呢，会使得分掉一些概率，例如0.5 0.5 但是实际上两个是一样的，这会导致一些点没有用到，因为删掉了。
    # 理想的情况，就是对于两者的概率都是两者之和。
    # TODO 不去重
    all_posi = pdcontent.values.tolist()
    dis_all_posi = []
    for thisframepos in all_posi:
        dis = (thisframepos[0]-x)**2 +(thisframepos[1]-y)**2
        dis_all_posi.append(thisframepos+[dis])
    # dis_all_posi就是生成的带有目标点距离的list
    dis_all_posi_np = np.array(dis_all_posi).reshape(-1,len(all_posi[0])+1)
#     print(dis_all_posi_np)
    a_arg = np.argsort(dis_all_posi_np[:,-1]) #按第'-1'列排序
    sortnp = dis_all_posi_np[a_arg.tolist()]

    # return sortnp[1:1+n] #return n*5 array
    # 或许这样不太明智，有的可能是最后一个位置，不太行，我们还没考虑到缺失位置的情况
    return sortnp #返回整条序列


def load_model(g_opt, device):

    checkpoint = torch.load(g_opt['model']) #, map_location={'cuda:0': 'cuda:2'}
    opt = checkpoint['settings']
    transformer = Transformer_sep_pred_wocusum(
        n_passed = opt.len_established,
        n_future = opt.len_future,
        n_candi = opt.num_cand,
        n_position = opt.n_position,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        inoutdim = opt.inoutdim).to(device)
    transformer.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return transformer 

def getIdx(a):
    co = a.unsqueeze(0)-a.unsqueeze(1)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    return out





#################################################################
#################################################################
#################################################################
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########
                            #########

################################################################################
def main2(
    input_detxml,output_trackcsv,model_path,fract,
    Past,Cand,Near,mean_=None,std_=None,vis=False,
    track_high_thresh=0.6,track_buffer=30,track_low_thresh=0.1,new_track_thresh=0.7):

    print('===>>>START {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

    opt = {}
    opt['model'] = model_path
    # device = 'cpu'
    device = 'cuda'
    transformer = load_model(opt, device)
    transformer.eval()

    # Past = 7
    # 待预测的数据——这里直接用不同SNR的同样密度的
    test_track_xml = input_detxml
    # pos_list_all = readXML(test_track_xml) # 读入 list格式

    # 所有检测的df
    # detection_total = pd.read_csv(test_track_xml,index_col=0).iloc[:,[7,8,0]]
    # 过滤掉低置信度的检测坐标
    detection_total_ori = pd.read_csv(test_track_xml,index_col=0).iloc[:,[6,7,8,2,3,4,5,0]]
    detection_total = detection_total_ori[detection_total_ori['conf']>track_high_thresh]
    detection_total = detection_total[[
        'pos_x','pos_y','bb_left','bb_top','bb_width','bb_height','frame']]
    detection_total['bb_right'] = detection_total['bb_left'] + detection_total['bb_width']
    detection_total['bb_bottom'] = detection_total['bb_top'] + detection_total['bb_height']
    detection_total = detection_total[['pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
    # 随机丢失一些行
    # detection_total = detection_total.sample(frac=fract,replace=False,random_state=1,axis=0)
    # detection_total.reset_index(drop=True,inplace=True)
    detection_total['det_id'] = detection_total.index
    imgx = detection_total['pos_x'].max()
    imgy = detection_total['pos_y'].max()
    # detection_total = detection_total[detection_total['frame']>97]
    start_frame = min(list(detection_total['frame']))
    end_frame = max(list(detection_total['frame']))

    # 建立 live track df
    established_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame'])   
    keep_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame'])

    
    print('===>>>Finish prepare total det')
    this_frame = start_frame
    while(this_frame<end_frame):
        print('===>>>Process Frame {}-{}/{}'.format(this_frame,this_frame+1,end_frame))
        this_det = detection_total[detection_total['frame'] == this_frame]
        # 候选的detection 和 id
        next_det = detection_total[detection_total['frame']==this_frame+1]

        if this_frame == start_frame or (len(established_track) == 0 and len(this_det)>0): # 初始化established track
            # established_track 是维护的live track
            established_track = this_det[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
            established_track = established_track.rename(columns={'det_id':'trackid'})
            temp = np.zeros([len(this_det),2])
            temp[:,0] = this_det['det_id']
            established_track_HOLDnum = pd.DataFrame(temp)
            established_track_HOLDnum.columns = ['trackid','HOLDnum']
        t_trackid = set(established_track['trackid'])

        if len(established_track) == 0:
            this_frame += 1
            continue

        n_near = Near

        # 对每个track生成sample
        # 对所有的生成sample
        
        one_frame_match_list = []
        for one_trackid in t_trackid:

            thistrack_dic = {}
            one_track = established_track[established_track['trackid']==one_trackid]

            # 如果需要，进行padding
            one_track_length = len(one_track)
            one_track_list = one_track.values.tolist()
            p_ind = one_track_list[0][0] # 该track id号
            padding_before = []
            if one_track_length < Past:
                for a in range(Past-one_track_length):
                    padding_before.append([p_ind]+[-1]*len(one_track_list[0])) #多加最后一列，0表示该数据不是padding的-1，-1表示是padding的
            convert_one_track = []
            for b in one_track_list:
                convert_one_track.append(b+[0])
            pad_paticle_poslist = padding_before+convert_one_track

            # 对于非结尾处的sample进行生成
            if convert_one_track[-1][-2]<end_frame:
                pastposlist = []
                # =============pasted部分[]==============
                for i in range(-Past,0): # 后7个作为一条数据
                    pastposlist.append([
                        pad_paticle_poslist[i][1],
                        pad_paticle_poslist[i][2],
                        pad_paticle_poslist[i][3],
                        pad_paticle_poslist[i][4],
                        pad_paticle_poslist[i][5],
                        pad_paticle_poslist[i][6],
                        pad_paticle_poslist[i][8]
                        ])
                tree = Tree()
                tree.create_node(tag='ext', identifier='ext', data=pastposlist[-1])
                
                frame_ind = this_frame+1 #frame_ind是候选的第一个位置是哪一帧
                frame_indnext_list = [frame_ind+t for t in range(Cand)]

                nodenamelist = ['ext']
                SIG = True
                for frame_ind__ in frame_indnext_list:
                    newnamelist = []
                    for tobe_extlabel in nodenamelist: #对于这一层的每个节点
                        
                        thisnodedata = tree.get_node(tobe_extlabel).data
                        # 判断该节点是否为空，准备查找的obj
                        if thisnodedata[-1] == -1:
                            parentnodelabel = tobe_extlabel
                            for _ in range(Cand):
                                parentnodelabel = tree.parent(parentnodelabel).tag
                                parentnode = tree.get_node(parentnodelabel)
                                if parentnode.data[-1] != -1:
                                    near_objpos = parentnode.data.copy()
                                    break
                        else:
                            near_objpos = thisnodedata.copy()
                        if frame_ind__ > end_frame  or len(detection_total[detection_total['frame']==frame_ind__]) == 0:
                            # np_re = (-np.ones([n_near,3])).tolist()
                            np_re= []
                        else:
                            np_re = find_near(  #pos_x, pos_y, left, right, top, bottom, frame, det_id, dis
                                pdcontent=detection_total[detection_total['frame']==frame_ind__],
                                x=near_objpos[0],
                                y=near_objpos[1])
                            # if len(np_re) < n_near:
                                # np_re = np.concatenate([np_re,-np.ones([n_near-len(np_re),5])])
                        # 判断其子集是否为空，如果为空则生成n_near个+空，如果不为空，则看项是不是空，如果是空则生成n_near-1个子集+null，如果是空，则生成4个子集
                        # if tree.children(tobe_extlabel) == []:
                        numb = 0
                        neednull = 1
                        notequalGT = 0

                        det_id_4cand = []
                        for ppos in np_re: #对于每一个找到的位置,都要生成候选
                            det_id_4cand.append(int(ppos[-2]))
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename,
                                parent=tobe_extlabel, 
                                data=[ppos[0],ppos[1],ppos[2],ppos[3],ppos[4],ppos[5],0])
                            if numb == n_near-neednull:
                                break
                        if numb < n_near-neednull:
                            neednull = n_near-numb               
                        for _i in range(neednull):
                            det_id_4cand.append(-1)
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename, 
                                parent=tobe_extlabel, 
                                data=[-1]*len(pastposlist[0]))
                        if SIG:
                            det_id_4cand_reserve = det_id_4cand.copy()
                            SIG = False
                    nodenamelist = newnamelist.copy()
                # 转为list
                all_candidate = []
                paths_leaves = [path_[1:] for path_ in tree.paths_to_leaves()]
                for onepath in paths_leaves:
                    onepath_data = []
                    for onepos in onepath:
                        onepath_data.append(tree.get_node(onepos).data)
                    all_candidate.append(onepath_data)
                
            thistrack_dic['trackid'] = one_trackid
            thistrack_dic['cand5_id'] = det_id_4cand_reserve
            thistrack_dic['pastpos'] = pastposlist
            thistrack_dic['cand25'] = all_candidate
            
            one_frame_match_list.append(thistrack_dic)

        print('===>>>Finish construct samples')

        # if this_frame == 473:
        #     holdhold = 1
        this_frame_dataloader,this_frame_data = func_getdataloader_oridif_match(
            one_frame_match_list,
            batch_size=len(one_frame_match_list), 
            shuffle=False, 
            num_workers=1,
            mean=mean_,
            std=std_)
        bbat = this_frame_data[0]
        for batch in this_frame_dataloader:
            src_seq = batch[0].float().to(device) # 这里就会转换成19维
            trg_seq = batch[1].float().to(device)
            trackid_batch = batch[2].tolist()
            cand5id_batch = batch[3].tolist()
            pred_shift,pred_prob = transformer(src_seq, trg_seq)
            src_seq_abpos = batch[-1].float()
        
        # 记录轨迹的最后一个绝对位置  x y
        lastab = src_seq_abpos[:,-1,:2].numpy()
        

        # 找到det_id是一样的项，然后将算出来的soft_pred_prob5 加一起
        shrink = nn.MaxPool1d(kernel_size=near**(Cand-1), stride=near**(Cand-1))
        soft = nn.Softmax(dim=-1)
        pred_prob5 = shrink(pred_prob.unsqueeze(0)).squeeze(0)
        soft_pred_prob5 = soft(pred_prob5).detach().cpu().numpy().tolist()
        norm_pred_prob5 = (pred_prob5-pred_prob5.min())+0.0001
        norm_pred_prob5 = (norm_pred_prob5/norm_pred_prob5.max()).detach().cpu().numpy().tolist()
        # 应该记录一个分类的number
        pred_value,pred_clsnum = torch.max(pred_prob,1) #value是那个小类相似度值，clsnum是预测小类
        pred_nextone_detid  = torch.Tensor(cand5id_batch)[ #预测的下一个位置的detid，for后面判断连接
            [range(len(cand5id_batch))],pred_clsnum // near**(Cand-1)][0].detach().cpu().numpy().reshape(-1,1)
        # 建立对这batch数据预测位移的dataframe ，用于HOLD
        pred_shift_next1 = pred_shift[:,0,:-1].detach().cpu().numpy()*this_frame_data.std +this_frame_data.mean
        pred_shift_exist_flag = pred_shift[:,0,-1].detach().cpu().numpy().reshape(-1,1)
        pred_shift_id_np = np.concatenate([
            np.array(trackid_batch).reshape(-1,1),
            pred_shift_next1,pred_shift_exist_flag,
            pred_nextone_detid
            ],-1)
        pred_shift_id_pd = pd.DataFrame(pred_shift_id_np)
        pred_shift_id_pd.columns = [
            'trackid',
            'shift_x',
            'shift_y',
            'shift_left',
            'shift_right',
            'shift_top',
            'shift_bottom',
            'shift_width',
            'shift_height',
            'abs_x',
            'abs_y',
            'abs_left',
            'abs_right',
            'abs_top',
            'abs_botoom',
            'abs_width',
            'abs_height',
            'exist_flag',
            'p_detid']
        print('===>>>Finish Predicting probability')
        
        ######################################################
        # vis
        if vis:
            imagefolder = 'dataset/MOT17/image/'
            thisframeimage = cv2.imread(
                imagefolder+input_detxml.split('/')[-1].split('.')[0] \
                    +'/img1/{:06d}.jpg'.format(int(this_frame)))
            for indd,tkid in enumerate(trackid_batch):
                onetracklet = src_seq_abpos[indd]
                predthisdetid = pred_nextone_detid[indd]
                realtraj = onetracklet[onetracklet[:,-1] != -1][:,:2] \
                    .reshape(-1,2).unsqueeze(0).numpy()

                thisfrmtkid_image = np.zeros(
                    [thisframeimage.shape[0]+200,thisframeimage.shape[1],thisframeimage.shape[2]])
                thisfrmtkid_image[:-200,:,:] = thisframeimage
                
                

                # 绘制所有下一位置的候选位置和id，以及亲和度
                # -1 就是不动，先只绘制一个候选位置吧
                nextindhere = range(0,trg_seq.shape[1],n_near**(Cand-1))
                nextindhere = list(nextindhere)
                # tm = range(trg_seq.shape[0])
                # tm = list(tm)
                nextoneshifts = trg_seq[indd][nextindhere][:,0].detach().cpu().numpy()
                nullflag = np.sum(nextoneshifts == 0,1)<3
                nextoneabsshifts = nextoneshifts[:,:2]*this_frame_data.std[:2] +this_frame_data.mean[:2]
                nextonepos = lastab[indd].reshape(-1,2) + nextoneabsshifts

                for jh in range(len(nextonepos)):
                    if predthisdetid == cand5id_batch[indd][jh]:
                        colortuple = (255,0,255) # 对于最高预测的用紫色写文字和圈出
                    else:
                        colortuple = (0,255,0) #对于普通预测的用绿色写文字和圈出
                    if nullflag[jh]:
                        thisfrmtkid_image = cv2.polylines(
                            thisfrmtkid_image,
                            np.concatenate(
                                [lastab[indd,:][np.newaxis,:],nextonepos[jh,:][np.newaxis,:]],0)[np.newaxis,:].astype(int), 
                            False, colortuple, 1
                        )
                        thisfrmtkid_image = cv2.putText(
                            img       = thisfrmtkid_image, 
                            text      = str(cand5id_batch[indd][jh]), 
                            org       = (int(nextonepos[jh,0]),int(nextonepos[jh,1])), 
                            fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                            fontScale = 1,
                            color     = colortuple,
                            thickness = 1)

                        thisfrmtkid_image = cv2.putText(
                            img       = thisfrmtkid_image, 
                            text      = '[Detection]'+str(cand5id_batch[indd][jh])+'_{:.1f}'.format(pred_prob5[indd][jh]), 
                            org       = (50,thisframeimage.shape[0]+40+jh*40), 
                            fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                            fontScale = 2,
                            color     = colortuple,
                            thickness = 2)

                        x_ = int(nextonepos[jh,0])
                        y_ = int(nextonepos[jh,1])
                        pt = (x_,y_)
                        thisfrmtkid_image = cv2.circle(
                            thisfrmtkid_image, pt, 2, colortuple, 2)
                    else:
                        thisfrmtkid_image = cv2.putText(
                            img       = thisfrmtkid_image, 
                            text      = '[Detection]'+str(cand5id_batch[indd][jh])+'_{:.1f}'.format(pred_prob5[indd][jh]), 
                            org       = (50,thisframeimage.shape[0]+40+jh*40), 
                            fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                            fontScale = 2,
                            color     = colortuple,
                            thickness = 2)

                # 绘制单个轨迹的历史 历史轨迹是白色
                thisfrmtkid_image = cv2.polylines(
                    thisfrmtkid_image,
                    realtraj.astype(int), False, (255,255,255), 1)
                for i in range(len(realtraj[0])):
                    x_ = int(realtraj[0,i,0])
                    y_= int(realtraj[0,i,1])
                    pt = (x_,y_)
                    thisfrmtkid_image = cv2.circle(
                        thisfrmtkid_image, pt, 1, (255,255,255), 1)

                cv2.imwrite(
                    output_trackcsv.replace('link.csv','/')+"frame{:06d}_track{:06d}_.jpg".format(int(this_frame), tkid),
                    thisfrmtkid_image
                )
        ######################################################
              
        costlist = []
        # 生成优化矩阵
        for it in range(len(one_frame_match_list)):
            for m in range(near):
                costlist.append([norm_pred_prob5[it][m], trackid_batch[it], cand5id_batch[it][m]])

        # gurobi 求解
        # 开始创建模型
        # print('===>>>start construct model {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))
        print('===>>>start construct model')
        costs = costlist.copy()
        # Create a new model
        model = grb.Model("mip1")
        model.Params.outputflag = 0

        # Create variables 变量，其实也只有0-1变量，决定选不选那个连线
        ro = [] # ro就是0-1变量，代表每个连线，选还是不选。名称是z_0_1 格式，代表第0个passed 和 第1个candidate连接，对应的0-1变量
        for i in range(len(costs)):
            ro.append(model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY,'z_' + str(costs[i][1]) + '_' + str(costs[i][2]))) #名字
        model.update()

        # Set objective 目标函数
        expr = grb.LinExpr()
        for i in range(len(ro)):
            expr.add(ro[i], costs[i][0]) # 第一个位置是cost，这一步 就是将 0-1变量与对应项的cost相乘。
        model.setObjective(expr, grb.GRB.MAXIMIZE) # 最大化

        # Add constraint 约束
        nrConstraint = 0
        exprcs = []
        # 这个循环，是以passed为主体，就是连到同一个passed上的线，只能选一条
        for j in trackid_batch: #遍历所有的passed
            exprc = grb.LinExpr()
            flag = False
            for cc in range(len(costs)): 
                if costs[cc][1] == j: 
                    exprc.add(ro[cc], 1.0)
                    flag = True
            nrConstraint += 1 # 约束的数目+1
            exprcs.append(exprc)
            if flag: #若用LESS_EQUAL就可以解了 #其实就是less_equal吧.因为这里还没把出现\消失作为一个新的choice.如果作为新的choice就是必须都=1
                model.addConstr(exprc, grb.GRB.EQUAL, 1.0, "c_" + str(nrConstraint)) #这里用=1约束，意思是每个passed一定要有一个匹配的。
        # print('constrained num:{}'.format(nrConstraint))        
        # print('--------------------------------------------------------------')      
        # 这个循环，是以candidate为主体，就是连到同一个candidate上的线，要么没有要么只有1个，毕竟候选数量是很多的。
        for j in list(next_det['det_id']): 
            exprc = grb.LinExpr()
            flag = False
            for cc in range(len(costs)):
                if costs[cc][2] == j:
                    exprc.add(ro[cc], 1.0)
                    flag = True
            nrConstraint += 1
            exprcs.append(exprc)
            if flag:
                model.addConstr(exprc,grb.GRB.LESS_EQUAL,1.0, "c_" + str(nrConstraint)) #这里用<=1约束，意思是每个candidate最多有一个匹配的，可能无
        
        # print('===>>>Finish construct model {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))
        print('===>>>Finish construct model {}')
        model.optimize()
        # print('===>>>Finish optimize {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))
        print('===>>>Finish optimize')
        assert model.status == grb.GRB.Status.OPTIMAL

        # 取所有的解
        solutionIndex = []
        solutions = []
        # 找出所有的解的line
        for i in range(len(ro)):
            if ro[i].Xn > 0.5:
                solutionIndex.append(i)
                solutions.append(ro[i].VarName)
        # print('find all solution')
        
        # 对于从live track df中删除的，就整条加入keep result
        # 对于连接的，就更新live df
        # 对于没有连接的det id 将其变成trackid 加入live track df

        # print(pred_shift_id_pd['exist_flag'].mean(),pred_shift_id_pd['exist_flag'].std(),pred_shift_id_pd['exist_flag'].median())
        linked_det_id = []
        for so in solutions:
            link_track_id = int(so.split('_')[1])
            link_cand_id = int(so.split('_')[2])
            
            # 再读入对应track 和对应 frame的图，在上面标注gurobi匹配后的结果
            p_detid = pred_shift_id_pd[pred_shift_id_pd['trackid'] == link_track_id]['p_detid'].values[0]
            if link_cand_id != p_detid:
                dif_cls_gurobi_changeto_1 = True # 只记录，但不更改
            # read image
            ######################################################
            # vis
            if vis:
                gurobiimage = cv2.imread(
                    output_trackcsv.replace('link.csv','/')+"frame{:06d}_track{:06d}_.jpg".format(int(this_frame), link_track_id))
                if link_cand_id == p_detid:
                    colortuple = (255,0,255) # 如果gurobi匹配结果是分类结果，用紫色写gurobi结果，不用圈了
                else:
                    colortuple = (10,215,255) # 如果gurobi不一致，用金色写结果，空心圈出gurobi预测的位置，这里是以gurobi预测的位置确实连接的
                gurobiimage = cv2.putText(
                            img       = gurobiimage, 
                            text      = '[Gurobi]'+str(link_cand_id), 
                            org       = (450,thisframeimage.shape[0]+40), 
                            fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                            fontScale = 2,
                            color     = colortuple,
                            thickness = 2)
                if link_cand_id != p_detid and link_cand_id > 0:
                    pt = (int(next_det[next_det['det_id'] == link_cand_id].iloc[0,0]), int(next_det[next_det['det_id'] == link_cand_id].iloc[0,1]))
                    gurobiimage = cv2.circle(
                        gurobiimage, pt, 5, colortuple, 2)

                # cv2.imwrite(
                #     output_trackcsv.replace('link.csv','/')+"frame{:06d}_track{:06d}_gurobi.jpg".format(int(this_frame), link_track_id),
                #     gurobiimage
                # )
                # cv2.imwrite(
                #     output_trackcsv.replace('link.csv','/')+"track{:06d}_frame{:06d}_gurobi.jpg".format(link_track_id, int(this_frame) ),
                #     gurobiimage
                # )
            # 到此可以看到预测的准确性，以及gurobi后的准确性
            # 下面会想要看到经过轨迹管理，最终的决策
            use_guribimatchdet = False
            use_fillup = False
            dist_changeto_1 = False
            dif_cls_gurobi_changeto_1 = False
            if link_cand_id != -1: # 若是不消失，有后继
                # 判断后继是否是第一个且过长超过30/50
                # 判断是否出现异常行为，

                thisid_track = established_track[established_track.trackid==link_track_id]
                # print(thisid_track)
                last_x = thisid_track.iloc[-1,1]
                last_y = thisid_track.iloc[-1,2]
                # print(last_x,last_y)
                ext = next_det[next_det['det_id'] == link_cand_id]
                # print(ext)
                next_x = ext.iloc[0,0]
                next_y = ext.iloc[0,1]
                # print(next_x,next_y)

                dist = np.sqrt((next_x-last_x)**2 + (next_y-last_y)**2)
                # print(dist)
                # if dist > 30:
                #     link_cand_id = -1
                #     dist_changeto_1 = True
                # else:
                ext = ext[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
                ext = ext.rename(columns={'det_id':'trackid'})
                ext['trackid'] = link_track_id
                established_track = established_track.append(ext)
                linked_det_id.append(link_cand_id)
                established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = 0
                use_guribimatchdet = True
            # use_guribimatchnull = False
            stop_4_outview = False
            stop_4_overhold = False
            stop_4_predstop = False
            stop_4_moviestop = False
            
            if link_cand_id == -1: # 若消失,则保持一帧，用预测填充，establish记录保持的数量。保持如果超过HOLDNUM，HOLDNUM是最后连续HOLD的情况。
                # 从established中删除，整条加入keep
                thisid_HOLDnum = established_track_HOLDnum[established_track_HOLDnum.trackid == link_track_id].iloc[0,1]
                thisid_pred_shift = pred_shift_id_pd[pred_shift_id_pd.trackid == link_track_id]
                
                # print(thisid_pred_shift.iloc[0,3])
                # 这里判断是否继续的标志可能因为det的缺失判定为停止
                thisid_track = established_track[established_track.trackid==link_track_id]
                # if  (thisid_HOLDnum <10) and (this_frame < end_frame-1) and len(thisid_track)>3: # and (thisid_pred_shift.iloc[0,3]>pred_shift_id_pd['exist_flag'].mean()): #and thisid_pred_shift.iloc[0,3] > pred_shift_id_pd['exist_flag'].median()
                    # (thisid_HOLDnum <10) and
                    # 沿着上一步的扩展
                if  (thisid_HOLDnum < track_buffer) and (this_frame < end_frame-1):# and len(thisid_track)>1: #不再增加扩展长度的约束，但是也得保证扩展的对才行啊
                    # 给十帧，如果没连上，就删掉填充的，如果扩展中途出去了，就结束但保留。如果有
                    last_frame = thisid_track.iloc[-1,-1]
                    last_x = thisid_track.iloc[-1,1]
                    last_y = thisid_track.iloc[-1,2]
                    last_left = thisid_track.iloc[-1,3]
                    last_right = thisid_track.iloc[-1,4]
                    last_top = thisid_track.iloc[-1,5]
                    last_bottom = thisid_track.iloc[-1,6]

                    # last_shift_x = 0
                    # last_shift_y = 0
                    if len(thisid_track) < 2:
                        last_shift_x = 0
                        last_shift_y = 0
                        last_shift_left   = 0
                        last_shift_right  = 0
                        last_shift_top    = 0
                        last_shift_bottom = 0
                    else:
                        last_shift_x = thisid_pred_shift.iloc[0,1]
                        last_shift_y = thisid_pred_shift.iloc[0,2]
                        last_shift_left   = thisid_pred_shift.iloc[0,3]
                        last_shift_right  = thisid_pred_shift.iloc[0,4]
                        last_shift_top    = thisid_pred_shift.iloc[0,5]
                        last_shift_bottom = thisid_pred_shift.iloc[0,6]

                    #预测的位移
                    # pred_shift_x = thisid_pred_shift.iloc[0,1]
                    # pred_shift_y = thisid_pred_shift.iloc[0,2]

                    # 不使用预测的，使用历史帧5帧的平均，如果没有建立起5帧就需要减小位移*0.5这样子希望其尽量保持较小的变化
                    shift_x = last_shift_x
                    shift_y = last_shift_y
                    shift_left  =last_shift_left  
                    shift_right =last_shift_right 
                    shift_top   =last_shift_top   
                    shift_bottom=last_shift_bottom

                    # if pred_shift_x**2 + pred_shift_y**2 > 4*(last_shift_x**2+last_shift_y**2): # 判断预测的是否过长
                    #     shift_x = last_shift_x
                    #     shift_y = last_shift_y
                    # else:
                    #     shift_x = pred_shift_x
                    #     shift_y = pred_shift_y
                    

                    if last_x+shift_x < imgx and last_y+shift_y < imgy and last_x+shift_x >0 and last_y+shift_y>0:
                        temp_dic = {
                            'trackid'  :[link_track_id],
                            'pos_x'    :[last_x+shift_x],
                            'pos_y'    :[last_y+shift_y],
                            'bb_left'  :[last_left  +shift_left  ],
                            'bb_right' :[last_right +shift_right ],
                            'bb_top'   :[last_top   +shift_top   ],
                            'bb_bottom':[last_bottom+shift_bottom],
                            'frame':[last_frame+1]
                            }
                        ext  = pd.DataFrame(temp_dic)
                        established_track = established_track.append(ext)
                        established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = thisid_HOLDnum+1
                        use_fillup = True
                    else:#走出去了也去掉
                        tobeapp = established_track[established_track['trackid']==link_track_id]
                        keep_track = keep_track.append(tobeapp)
                        established_track = established_track[established_track['trackid'] != link_track_id] # 这样就相当于去掉了那个
                        stop_4_outview = True
                else: # 因为过长或者movie结束，会去掉最后几个
                    stop_4_overhold = True if thisid_HOLDnum >=30 else False
                    # stop_4_predstop = True if p_longhold != 1 else False
                    stop_4_moviestop = True if this_frame >= (end_frame-1) else False
                    
                    thisholdnum = thisid_HOLDnum
                    if thisholdnum > 0:
                        # 去掉最后的几个
                        tobeapp = established_track[
                            established_track['trackid']==link_track_id].iloc[:-int(thisholdnum),:]
                    else: #之前的这一步就错过掉了，所以如果不固定的去用hold1结束，就会删除了一些轨迹。
                        tobeapp = established_track[
                            established_track['trackid']==link_track_id]

                    keep_track = keep_track.append(tobeapp)
                    established_track = established_track[established_track['trackid'] != link_track_id]
            
            if vis:
                conclusion = []
                if use_guribimatchdet:
                    conclusion.append('Use gurobi matching result')
                    if not dif_cls_gurobi_changeto_1:
                        conclusion.append('Same as cls pred')
                    else:
                        conclusion.append('Different from cls pred')
                if dist_changeto_1:
                    conclusion.append('Gurobi det is too far, not link it')
                
                if use_fillup:
                    conclusion.append('Fill up a prediction pos')
                if stop_4_moviestop:
                    conclusion.append('Stop for movie end')
                if stop_4_outview:
                    conclusion.append('Stop for out of image')
                if stop_4_predstop:
                    conclusion.append('Stop for cls of strong stop')
                if stop_4_overhold:
                    conclusion.append('Stop for hold long without re-link')

                if not use_fillup:
                    colortuple = (255,255,255) #不填充的话用白色写总结
                else:
                    colortuple = (255,255,0) #填充的话，用亮蓝色写总结，并空心圆圈出fill的位置

                for m in range(len(conclusion)):
                    gurobiimage = cv2.putText(
                        img       = gurobiimage, 
                        text      = conclusion[m], 
                        org       = (700,thisframeimage.shape[0]+50+m*50), 
                        fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                        fontScale = 3,
                        color     = colortuple,
                        thickness = 2)

                if use_fillup:
                    pt = (int(temp_dic['pos_x'][0]),int(temp_dic['pos_y'][0]))
                    gurobiimage = cv2.circle(
                        gurobiimage, pt, 5, colortuple, 2)

                cv2.imwrite(
                    output_trackcsv.replace('link.csv','/')+"frame{:06d}_track{:06d}_mangement.jpg".format(int(this_frame), link_track_id),
                    gurobiimage
                )
                cv2.imwrite(
                    output_trackcsv.replace('link.csv','/')+"track{:06d}_frame{:06d}_mangement.jpg".format(link_track_id, int(this_frame) ),
                    gurobiimage
                )
        # break
        for to_belinkid in list(next_det['det_id']): #对于没有连接的det，加入established
            if to_belinkid not in linked_det_id:
                itsconf = detection_total_ori.iloc[to_belinkid]['conf']
                if itsconf > new_track_thresh:
                    ext = next_det[next_det['det_id'] == to_belinkid]
                    ext = ext[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
                    ext = ext.rename(columns={'det_id':'trackid'})
                    established_track = established_track.append(ext)

                    temp_dic = {'trackid':[ext.iloc[0,0]],'HOLDnum':[0]}  
                    temp_pd = pd.DataFrame(temp_dic)
                    established_track_HOLDnum = established_track_HOLDnum.append(temp_pd)


        print('===>>>Finish process!!! {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

        this_frame += 1


    # 最后把live df 中所有的保存到keep
    keep_track = keep_track.append(established_track)
    keep_track.to_csv(output_trackcsv)
    # return(keep_track)



########################################################################

        #############                        ############# 
        ##############                      ############## 
        ###############                    ############### 
        ################                  ################ 
        ######## ########                ######## ######## 
        ########  ########              ########  ######## 
        ########   ########            ########   ######## 
        ########    ########          ########    ########
        ########     ########        ########     ########
        ########      ########      ########      ########
        ########       ########    ########       ########
        ########        ########  ########        ########
        ########         ################         ########
        ########          ##############          ########
        ########           ############           ########
        ########            ##########            ########
        ########             ########             ########


if __name__ == '__main__':

    pyfilepath = os.path.abspath(__file__)
    # =================train=========================
    # MOT17
    # traindatamean = [-3.8447242, -7.2373133, -2.3897965, -5.299531, -3.6751747, -10.802332, -2.909537, -7.12589, \
    #     803.7549, 403.8249, 774.43823, 836.82025, 314.45084, 494.85004,  62.86108, 179.01665  ]
    # traindatastd = [110.6746 ,67.4468, 112.93308, 112.18339, 72.53153, 95.47296, 41.04576, 102.74431, \
    #     455.76172, 218.71327, 454.32388, 459.0247, 217.7225, 238.48856, 49.815372, 130.16364 ]
    traindatamean = None
    traindatastd = None

    # data param
    inoutdim = 17
    past = 7
    cand=1
    near = 5
    # network param
    n_layer_ = 1
    n_head_ = 6
    d_kv_ = 96
    d_model_ = n_head_*d_kv_
    ffn_ = 2*d_model_
    # optim param
    warmup_ = 1000
    batch_ = 64
    # datapath outputpath
    
    # traindata_path = 'dataset/MOT17/train_dataset/merge_train.txt'
    # valdata_path = 'dataset/MOT17/train_dataset/merge_val.txt'
    
    traindata_path = 'dataset/MOT17/20221127boxgt_onefuture/merge_train.txt'
    valdata_path = 'dataset/MOT17/20221127boxgt_onefuture/merge_val.txt'  #dropGTdet_val
    outputmodel_path = './20221127_model_MOT17_auginput17dim_onefuture/'


    if not os.path.exists(outputmodel_path):
        os.makedirs(outputmodel_path)
    # train function
    shutil.copy(pyfilepath,outputmodel_path+pyfilepath.split('/')[-1])
    traindatamean,traindatastd = main1(
        n_layer_, n_head_, d_kv_, d_model_, ffn_, 
        warmup_, batch_,
        traindata_path,valdata_path,outputmodel_path,
        past,cand,near,
        traindatamean,traindatastd,inoutdim,
        )

    # print(traindatamean)
    # print(traindatastd)
    with open(outputmodel_path+'meanstd.txt','w') as f:
        f.write(str(traindatamean))
        f.write(str(traindatastd))
    f.close()
    #==================test=====================
    #  test link
    visrpocess = False
    output_path = './20221127_result_auginput17dim_finethresh_wodist_onefut/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    shutil.copy(pyfilepath,output_path+pyfilepath.split('/')[-1])

    fract_ = 1.0
    model_p = glob.glob(outputmodel_path+'/**.chkpt')[-1].replace('\\','/')

    # test_det_pa = 'dataset/MOT17/prediction/det.csv'
    test_det_palist = glob.glob('./dataset/MOT17/yolox_det_all/**.csv') #dataset/MOT17/yolox_det_all
    test_det_palist.sort()
    # test_det_palist = [ test_det_palist[_] for _ in [0,4,8,11] ]
    for test_det_pa in test_det_palist:
        seq = test_det_pa.split('/')[-1].split('.')[0]
        track_high_thresh = 0.6
        track_low_thresh = 0.1
        track_buffer = 30
        new_track_thresh = track_high_thresh
        
        if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
            track_buffer = 14
        elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
            track_buffer = 25
        else:
            track_buffer = 30

        if seq == 'MOT17-01-FRCNN':
            track_high_thresh = 0.65
        elif seq == 'MOT17-06-FRCNN':
            track_high_thresh = 0.65
        elif seq == 'MOT17-12-FRCNN':
            track_high_thresh = 0.7
        elif seq == 'MOT17-14-FRCNN':
            track_high_thresh = 0.67
        elif seq in ['MOT20-06', 'MOT20-08']:
            track_high_thresh = 0.3
            # exp.test_size = (736, 1920)

        new_track_thresh = track_high_thresh + 0.1

        output_csv_pa = output_path+seq+'link.csv'
        if visrpocess:
            if not os.path.exists(output_path+seq):
                os.makedirs(output_path+seq)
        keep_track = main2(
            input_detxml=test_det_pa,  # GT det
            output_trackcsv=output_csv_pa,
            model_path = model_p,
            fract=fract_,
            Past = past,
            Cand=cand,
            Near=near,
            mean_ = traindatamean,
            std_ = traindatastd,
            vis = visrpocess,
            track_high_thresh=track_high_thresh,
            track_buffer     = track_buffer,
            track_low_thresh = track_low_thresh,
            new_track_thresh = new_track_thresh
            )
