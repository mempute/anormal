#-*- coding: utf-8 -*-
import mempute as mp
from morphic import *
import numpy as np
import  sys
from operator import eq
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import math
import datetime
import os
import re
import shutil

def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

def reverse_standardization(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * org_x_np.std() + org_x_np.mean())

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()



encoding = 'euc-kr' # 문자 인코딩
pred_size = 15#31#3#7
infer_size = 1999
test_size = infer_size * 2  #*********
interval = 1
seq_len = 31#63#7#15
n_inp = seq_len - pred_size
latent_sz = 64              #*********
deat = 1
n_epair = 2                 #*********
spk = 50                  #*********
rv = 1.0#0.0001
spk_lr = -0.001
rspk_lr = -1.01
glam = 1
batch_size = 128
gid = int(sys.argv[2])
strait = int(sys.argv[3])
gpt = 0

if gpt:
    pos_encoding = 1
    nblock = 6
    n_heads = 4
    d_ff = 512
else:
    pos_encoding = 0
    nblock = 1              #*********
    n_heads = 8
    d_ff = 0

model_name = f"{sys.argv[4]}_seq_len_{seq_len}_pred_size_{pred_size}_latent_sz_{latent_sz}\
                _deat_{deat}_n_epair_{n_epair}_gpt_{gpt}\
                _nblock_{nblock}_n_heads_{n_heads}\
                _d_ff_{d_ff}_spk_{spk}_rv_{rv}_glam_{glam}\
                _spk_lr_{spk_lr}_rspk_lr_{rspk_lr}_strait_{strait}"
model_name = re.sub(' +', '', model_name)

count = 20#22#72 #나눠서 추론할때 pred_size단위 반복 횟수 #*********
icnt = 0

#python train.py 1 0 1 train 
#python train.py 2 0 1 train -2
#python train.py 3 0 1 train
#python train.py 5 0 1 train

lane_strait = pd.read_csv('./train_data/lane_data_s.csv', encoding=encoding)
#lane_strait.info()
del lane_strait['Distance']
lane_strait = lane_strait.values.astype(np.float32) 
#print("lane straite shape: %d", lane_strait.shape)

#print('-----------a-----------')
globals()['lane_{}'.format('strait')] = data_standardization(lane_strait)


lane_curve = pd.read_csv('./train_data/lane_data_c.csv', encoding=encoding)
#lane_curve.info()
del lane_curve['Distance']
lane_curve = lane_curve.values.astype(np.float32) 
#print("lane curve shape: %d", lane_curve.shape)

#print('-----------b-----------')
globals()['lane_{}'.format('curve')] = data_standardization(lane_curve)



globals()['yd{}'.format(30)] = np.full((lane_curve.shape[0], 1), 0.3)
globals()['yd{}'.format(40)] = np.full((lane_curve.shape[0], 1), 0.4)
globals()['yd{}'.format(50)] = np.full((lane_curve.shape[0], 1), 0.5)
globals()['yd{}'.format(70)] = np.full((lane_curve.shape[0], 1), 0.7)
globals()['yd{}'.format(100)] = np.full((lane_curve.shape[0], 1), 1.0)


def pre_job(dname, district, yi):
    globals()['drive{}_{}'.format(yi,district)] = pd.read_csv(f'./train_data/{dname}{yi}.csv', encoding=encoding)
    #globals()['drive{}_{}'.format(yi,district)].info()
        
    del globals()['drive{}_{}'.format(yi,district)]['Distance']
    globals()['drive{}_{}'.format(yi,district)] = globals()['drive{}_{}'.format(yi,district)].values.astype(np.float32) 
    #print(f"drive{yi} {district} shape: %d", globals()['drive{}_{}'.format(yi,district)].shape)

    #print('-----------1-----------')
    globals()['drive{}_{}_src'.format(yi,district)] = globals()['drive{}_{}'.format(yi,district)][:, :-4]

    #print('-----------2-----------')

    globals()['drive{}_{}_src'.format(yi,district)] = data_standardization(globals()['drive{}_{}_src'.format(yi,district)])

    globals()['drive{}_{}'.format(yi,district)] = np.concatenate((globals()['yd{}'.format(yi)], 
                                                                    globals()['lane_{}'.format(district)], 
                                                                    globals()['drive{}_{}_src'.format(yi,district)], 
                                                                    globals()['drive{}_{}'.format(yi,district)][:, -4:]), axis=1)
    #print('-----------3-----------')

    globals()['drive{}_{}_train'.format(yi,district)] = []
    data_len = globals()['drive{}_{}'.format(yi,district)].shape[0]
    train_size = data_len - test_size
    for i in range(0, train_size - seq_len +1, interval):
        globals()['drive{}_{}_train'.format(yi,district)].append(globals()['drive{}_{}'.format(yi,district)][i:i+seq_len])
    globals()['drive{}_{}_train'.format(yi,district)] = np.array(globals()['drive{}_{}_train'.format(yi,district)], dtype=np.float32)
    #print('-----------4-----------')
    #print(f"drive{yi} {district} train shape: %d", globals()['drive{}_{}_train'.format(yi,district)].shape)

    globals()['drive{}_{}_test'.format(yi,district)] = np.array(globals()['drive{}_{}'.format(yi,district)][train_size-n_inp:train_size+infer_size], dtype=np.float32)
    globals()['drive{}_{}_test'.format(yi,district)] = np.expand_dims(globals()['drive{}_{}_test'.format(yi,district)], 0)
    #print('-----------5-----------')

    #print('-----------6-----------')
    #print(f"drive{yi} {district} test shape: %d", globals()['drive{}_{}_test'.format(yi,district)].shape)



if strait:
    pre_job('data_s', 'strait', 30)
    pre_job('data_s', 'strait', 40)
    pre_job('data_s', 'strait', 50)
    pre_job('data_s', 'strait', 70)
    pre_job('data_s', 'strait', 100)
    train_data = np.concatenate((drive30_strait_train,
                                    drive40_strait_train,
                                    drive50_strait_train,
                                    drive70_strait_train,
                                    drive100_strait_train
                                    ),  axis=0)

    test_data = np.concatenate((drive30_strait_test,
                                    drive40_strait_test,
                                    drive50_strait_test,
                                    drive70_strait_test,
                                    drive100_strait_test),  axis=0)
else:
    pre_job('data_c', 'curve', 30)
    pre_job('data_c', 'curve', 40)
    pre_job('data_c', 'curve', 50)
    pre_job('data_c', 'curve', 70)
    pre_job('data_c', 'curve', 100)
    train_data = np.concatenate((drive30_curve_train,
                                    drive40_curve_train,
                                    drive50_curve_train,
                                    drive70_curve_train,
                                    drive100_curve_train),  axis=0)
    

    test_data = np.concatenate((drive30_curve_test,
                                    drive40_curve_test,
                                    drive50_curve_test,
                                    drive70_curve_test,
                                    drive100_curve_test),  axis=0)

print("train data shape: %d", train_data.shape)
print("test data shape: %d", test_data.shape)

arr = np.split(train_data, [train_data.shape[2]-4], axis=2)
x_train = arr[0]
y_train = arr[1]
#print("x_train shape: %d", x_train.shape)
#print("y_train shape: %d", y_train.shape)
bos = np.zeros((y_train.shape[0], 1, y_train.shape[2]), dtype = y_train.dtype)
y_train = np.concatenate((bos, y_train), axis=1) #go mark
eos = np.zeros((x_train.shape[0], 1, x_train.shape[2]), dtype = x_train.dtype)
x_train = np.concatenate((x_train, eos), axis=1) #end mark
x_train = np.concatenate((x_train, y_train), axis=2)



xtrain_other = arr[1] #only_target 타겟만 오차 학습
#xtrain_other = train_data #입력 전체 오차 학습

eos = np.zeros((xtrain_other.shape[0], 1, xtrain_other.shape[2]), dtype = train_data.dtype)
xtrain_other = np.concatenate((xtrain_other, eos), axis=1) #end mark

def write_csv(preds):       
    
    index = []# 인덱스와 배열 초기화
    # CSV 파일 열기
    with open('./train_data/answer_sample.csv', 'r') as f:
        # CSV 리더 객체 생성
        reader = csv.reader(f)
        # 헤더 읽기
        header = next(reader)
        #array = []
        # 각 행에 대해 반복
        for row in reader:
            # 첫 번째 열을 인덱스로 추가
            index.append(row)
            # 나머지 열을 배열로 추가
            #array.append(row[1:])

    os.remove('./train_data/answer_sample.csv')

    preds = preds.transpose((1,0,2))

    seq = preds.shape[0]
    preds = preds.reshape(seq,-1)
    # 새로운 CSV 파일 생성
    with open('./train_data/answer_sample.csv', 'w', newline="") as f:
        # CSV 라이터 객체 생성
        writer = csv.writer(f)
        # 헤더 쓰기
        writer.writerow(header)
        # 각 행에 대해 반복
        for i, row in enumerate(preds):
            new_row = []
            prow = index[i]
            if strait:
                new_row.append(prow[0])
            else:
                new_row.extend(prow[0:21])
            for x in row:
                s = str(x)
                new_row.append(s)
            if strait:
                new_row.extend(prow[21:])
            writer.writerow(new_row)
        if seq < len(index):
            n = len(index)
            while seq < n:
                writer.writerow(index[seq])
                seq += 1

    shutil.copyfile('./train_data/answer_sample.csv', "./" + model_name + "/answer_sample.csv")

def inference(xinp):

    eos = np.zeros((xinp.shape[0], 1, xinp.shape[2]), dtype = xinp.dtype)

    assert (xinp.shape[1] - n_inp) % pred_size == 0
    #pad = np.zeros((xinp.shape[0], xinp.shape[1] % seq_len, xinp.shape[2]), dtype = xinp.dtype)
    #xinp = np.concatenate((xinp, pad), axis=1)

    for i in range(0, xinp.shape[1] - seq_len +1, pred_size):
        global count
        global icnt
        if count:
            icnt += 1
            if icnt > count: break
        x = np.concatenate((xinp[:,i:i+seq_len], eos), axis=1) #end mark
        x[:,n_inp:n_inp+1,-4:] = x[:,n_inp-1:n_inp,-4:]
        x = x.copy()
        p = mp.xpredict(net, x, 1, n_inp+1)
        xinp[:,i+n_inp:i+n_inp+pred_size:,-4:] = p[:,n_inp+1:,-4:]
        
            
        #break #one predict
    return xinp

def logw(fp, s):
    fp.write(s + '\n')
    print(s)


     
if sys.argv[1] == '1':# command gid dist model-name

    train_params = dict(
        input_size=x_train.shape[1],
        hidden_sz=latent_sz,
        learn_rate=1e-4,
        drop_rate=0.1,
        signid_mse = 1,
        wgt_save = 0,
        layer_norm = 1,
        n_epair = n_epair, 
        residual = 1,
        on_schedule = False,
        dtype = mp.tfloat,
        levelhold = 0.7,
        tunehold = 0.98,
        seed = 777, #-1,
        decay = 0.0001,
        decode_active = 1,
        fast_once = 0,
        nblock = nblock,
        nontis = -1,
        gpt_model = gpt,
        pos_encoding = pos_encoding,
        input_feature = x_train.shape[2],
        n_heads = n_heads,
        d_ff = d_ff,
        regression = 1,
        dec_lev_learn = 1,
        decode_infeat = xtrain_other.shape[2],
        size_embed = 0,
        boost_dim = x_train.shape[2] -4, 
        batch_size=batch_size)

    train_params = default_param(**train_params)
    train_params['aaaaa11'] = deat
    train_params['aaaaa16'] = spk
    train_params['aaaaa17'] = spk_lr
    train_params['aaaaa18'] = rv
    train_params['aaaaa19'] = glam
    train_params['aaaaaa10'] = rspk_lr
    param = param2array(**train_params)
    if param is None:
        exit()
    net = mp.neuronet(param, gid, model_name)
    #mp.neurogate(net, stmt, model_name, sensor, x_train, y_train, 1)
    mp.close_net(net)

    spec = f"pred_size: {pred_size} seq_len: {seq_len} latent_sz: {latent_sz} \n\
                deat: {deat} n_epair: {train_params['n_epair']} \n\
                gpt:{gpt} nblock: {train_params['nblock']} \
                n_heads: {train_params['n_heads']} d_ff: {train_params['d_ff']} \
                strait: {strait}\n"
    spec = re.sub(' +', ' ', spec)
    fp = open("./" + model_name + "/history.txt", 'w')
    fp.write(spec)
    fp.close()
else:
    #param = mp.load_param(model_name)
    #train_params = arr2py_param(param)

    #param = param2array(**train_params)
    #mp.update_param(param, model_name)
    
    net = mp.loadnet(gid, model_name)
    fp = open("./" + model_name + "/history.txt", 'a')

    if sys.argv[1] == '2':# command gid dist model-name epoch, 학습
        fp.write('\n' + " ".join(sys.argv)+'\n')
        fp.close()
        epoch = int(sys.argv[5]) 

        x_train = x_train.copy()
        xtrain_other = xtrain_other.copy()
        mp.xtrain(net, x_train, xtrain_other, 1, epoch, 0, 1)
        mp.regist(net)


    elif sys.argv[1] == '3':# command gid dist model-name epoch, 추론 및 정확도 계산
        
        if pred_size < infer_size:
            if infer_size % pred_size:
                npad = pred_size - (infer_size % pred_size)
                zpad = np.zeros((test_data.shape[0], npad, test_data.shape[2]), dtype = train_data.dtype)
                x_test = np.concatenate((test_data, zpad), axis=1)
            else:
                npad = 0
                x_test = test_data
            x_test = x_test.copy()
            x_test = inference(x_test)
            if count:
                preds = x_test[:,n_inp:n_inp+count*pred_size]
                rights = test_data[:,n_inp:n_inp+count*pred_size]
            else:
                preds = x_test[:,n_inp:n_inp+infer_size]
                rights = test_data[:,n_inp:]
            #preds = x_test[:,n_inp:n_inp+pred_size] #one predict
        else:
            eos = np.zeros((test_data.shape[0], 1, test_data.shape[2]), dtype = train_data.dtype)
            x_test = np.concatenate((test_data, eos), axis=1) #end mark
            x_test[:,n_inp:n_inp+1,-4:] = x_test[:,n_inp-1:n_inp,-4:]
            #z = np.zeros((x_test.shape[0], pred_size, x_test.shape[2]), dtype = x_test.dtype)
            #x_preds = np.concatenate((x_test[:, :n_inp], z), axis=1)
            #print(x_preds.shape)


            x_test2 = x_test.copy()
            x_preds = mp.xpredict(net, x_test2, 1, n_inp+1) 
            preds = x_preds[:,n_inp+1:]
            rights = test_data[:,n_inp:]
        #rights = test_data[:,n_inp:seq_len] #one predict

        preds = preds[:,:,-4:]
        rights = rights[:,:,-4:]

        logw(fp, f"------------ pred seq len: {preds.shape[1]} -----------")

        logw(fp, f'-------- total: {rights.size} -----------')
        a = np.abs(rights - preds)
        a = a.reshape(a.size).tolist()
        i = 0
        for x in a:
            if x > 0.001:
                i+=1
                #print(x, " ", end='')
        logw(fp, f'\n--------------- 0.001 over: {i} -----------')
        logw(fp, f'\n--------- MSE error-----------')
        u = np.sum((np.abs(np.subtract(rights, preds))) / np.abs(rights))
        logw(fp, f'MAPE SUM: {u}')
        logw(fp, f'MAPE: {u/rights.size}')
        s = np.sum(np.sqrt(np.square(np.subtract(rights, preds))))
        logw(fp, f'MSE SUM: {s}')
        error = np.sqrt(np.square(np.subtract(rights, preds))).mean()
        logw(fp, f'MSE: {error}')
        logw(fp, '-------------------')
        # Weighted MAPE 계산
        if pred_size < infer_size:
            if count:
                nw = count * pred_size
            else:
                nw = infer_size
        else:
            nw = pred_size
        weights = np.arange(1.0001, 1.0001 + 0.0001 * nw, 0.0001)
        if weights.shape[0] != rights.shape[1]:
            nw -= 1 #소수점 백단위 이하에서는 끝수도 포함되어 -1
            weights = np.arange(1.0001, 1.0001 + 0.0001 * nw, 0.0001)
        weights = weights.reshape(1,weights.shape[0],1)
        #print(np.sum((np.abs(rights - preds)) / rights))
        #print(np.sum((np.abs(rights - preds) * weights) / rights))
        #print((1/rights.size * 100))
        wmape = np.sum((np.abs(rights - preds) * weights) / np.abs(rights)) * (1/rights.size * 100)
        #wmape = np.sum(np.abs(rights - preds) / rights * weights) / np.sum(weights) * 100
        #wmape = np.mean(np.abs((rights - preds) / rights)) * 100
        logw(fp, f"MAPE: {wmape}")

        write_csv(preds)

    elif sys.argv[1] == '5':# command gid dist model-name epoch, 추론만 수행
        
        if pred_size < infer_size:
            if infer_size % pred_size:
                npad = pred_size - (infer_size % pred_size)
                zpad = np.zeros((test_data.shape[0], npad, test_data.shape[2]), dtype = train_data.dtype)
                x_test = np.concatenate((test_data, zpad), axis=1)
            else:
                npad = 0
                x_test = test_data
            x_test = x_test.copy()
            x_test = inference(x_test)
            if count:
                preds = x_test[:,n_inp:n_inp+count*pred_size]
            else:
                preds = x_test[:,n_inp:n_inp+infer_size]
        else:
            eos = np.zeros((test_data.shape[0], 1, test_data.shape[2]), dtype = train_data.dtype)
            x_test = np.concatenate((test_data, eos), axis=1) #end mark
            x_test[:,n_inp:n_inp+1,-4:] = x_test[:,n_inp-1:n_inp,-4:]
            #z = np.zeros((x_test.shape[0], pred_size, x_test.shape[2]), dtype = x_test.dtype)
            #x_preds = np.concatenate((x_test[:, :n_inp], z), axis=1)
            #print(x_preds.shape)


            x_test2 = x_test.copy() 
            x_preds = mp.xpredict(net, x_test2, 1, n_inp+1)
            preds = x_preds[:,n_inp+1:]

        preds = preds[:,:,-4:]


        write_csv(preds)
  