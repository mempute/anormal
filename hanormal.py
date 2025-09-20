
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import mempute as mp
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import  sys
import time
import json
import os

from morphic import *

bos = 6
pos = 5

def print_array(a, fp_name=None, limit=0):
    if fp_name is None: fp = None
    else: fp = open(fp_name, 'w')
    if len(a.shape) == 2:
        batch = a.shape[0]
        seq = a.shape[1]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[", end='')
            if fp: fp.write("[")
            while j < seq:
                s = str(a[i, j])
                print(s + " ", end='')
                if fp: fp.write(s + " ")
                j = j + 1
            print("]")
            if fp: fp.write("]\n")
            i = i + 1

    elif len(a.shape) == 3:
        batch = a.shape[0]
        seq = a.shape[1]
        dim = a.shape[2]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while j < seq:
                k = 0
                print("[", end='')
                if fp: fp.write("[")
                while k < dim:
                    s = str(a[i, j, k])
                    print(s + " ", end='')
                    if fp: fp.write(s + " ")
                    k = k + 1
                print("]", end='')
                if fp: fp.write("]")
                j = j + 1
            print("   ]\n")
            if fp: fp.write("   ]\n")
            i = i + 1
    else:
        batch = a.shape[0]
        bind = a.shape[1]
        seq = a.shape[2]
        dim = a.shape[3]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            b = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while b < bind:
                j = 0
                print("[  ", end='')
                if fp: fp.write("[  ")
                while j < seq:
                    k = 0
                    print("[", end='')
                    if fp: fp.write("[")
                    while k < dim:
                        s = str(a[i, b, j, k])
                        print(s + " ", end='')
                        if fp: fp.write(s + " ")
                        k = k + 1
                    print("]", end='')
                    if fp: fp.write("]")
                    j = j + 1
                print("   ]\n")
                if fp: fp.write("   ]\n")
                b = b + 1
            print("   ]batch\n")
            if fp: fp.write("   ]batch\n")
            i = i + 1
    if fp: fp.close()


def convert_row_first(d_type, aarray):
    n_input_row = aarray.shape[0]
    n_input_col = aarray.shape[1]
    output_arr = np.full((int(n_input_row), n_input_col), pos, dtype = d_type)
    in_row = 0
    while in_row < n_input_row:
        output_arr[in_row,:] = aarray[in_row,:]
        in_row += 1
    return output_arr

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

#get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

np.set_printoptions(precision=4, suppress=True)

df = pd.read_csv("anormdata/creditcard.csv")

df.isnull().values.any()

from sklearn.preprocessing import StandardScaler

#print(df.describe())

data = df.drop(['Time'], axis=1)
#data['Class'] = 1

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))



deatt = 0
nblock = 1
n_epair = 2
hidden_sz = 32
gpt_model = -1
pos_enc = 1
n_heads = 4
ffn_hidden = 256
qa_kernel = 0#8
precision = 0.2
qlearn_lev = 0
qgen_lev = 0
cut_over = 0.8
infini_kernel = 0

batch_size = 32
auto_regress = 1
decord_optimize = 1
small_test = 0
test_predict = 1
auto_morphic = False
decode_learn = 1
inner_mul = 5
outer_mul = 13
jit_count = 60#27
dec_lev_opt = 1
embedding = False
decode_xother = False
idpred_print = 0
dense_data = 0
xdisc_adopt = 0

if embedding:
    decode_xother = True
dx_mul = 10
if decode_xother:
    tot = data.values
    minv = np.min(tot)
    maxv = np.max(tot)
    print(minv)
    print(maxv)
    minv *= dx_mul
    maxv *= dx_mul
    print(minv)
    print(maxv)
    minv = int(minv)
    maxv = int(maxv)
    dec_infeat_other = maxv - minv + 1
    print(dec_infeat_other)
else:
    dec_infeat_other = 0

x_train, x_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)


#pick out -> target 학습(케이스 b) 설정 요소
sparse_dup = 1 #pick out -> target 학습때 pick out된 데이터가 중복이 거의 없어 데이터를 중복 입력시킬 경우
just_pick = 0 #pick out를 소수점 일정이하 절삭하여 선형이 아닌 이산값으로 pick out -> target 학습시킬 경우

pickid = []
pick_out = 1
set_nan = 1
nan_val = 0


if pick_out: pickid_num = 15#28
else: pickid_num = 5

if small_test == 1:
    x_train = x_train.iloc[:64]#3000 이면 u0.004d, 1000 이면 u0.02d
    x_test = x_test.iloc[:64]
    sensor = 2.7
else:
    sensor = 4.24
    if small_test == 2:
        x_train = x_train.iloc[:20000]
        x_test = x_test.iloc[:20000]

if idpred_print: 
    x_train = x_train.iloc[:1000]
    x_test = x_test.iloc[:1000]


"""
df['salary'] = 0
df['salary'] = np.where(df['job'] != 'student', 'yes', 'no')
for x in range(150, 200, 10):
    start = x
    end = x+10
    temp = df[(df["height"] >= start) & (df["height"] < end)]
    print("{}이상 {}미만 : {}".format(start, end, temp["height"].mean()))
"""


if (decord_optimize and int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 6) or (decord_optimize == 0 and int(sys.argv[1]) >= 5 and int(sys.argv[1]) <= 6):
    x_train = x_train[x_train.Class == 0]
    set_nan = 0
    print('only normal case training')
else: print('mix training')

y_train = x_train['Class']
x_train = x_train.drop(['Class'], axis=1)


Y_test = x_test['Class']
x_test = x_test.drop(['Class'], axis=1)


x_train = x_train.values
x_test = x_test.values

y_train = y_train.values
y_test = Y_test.values
"""
#데이터를 양수 값을 만들기 위해 최소값을 더해 준다. 안해도 된다.
if sys.argv[1] is not '5':
    x_train = x_train - minv
    x_test = x_test - minv
"""
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

x_train = convert_row_first("f", x_train)
x_test = convert_row_first("f", x_test)
y_train = convert_row_first("f", y_train)
y_test = convert_row_first("f", y_test)

#x_train = min_max_scaling(x_train)
#x_test = min_max_scaling(x_test)

#make 31 seq
z = np.full((x_train.shape[0], 31 - x_train.shape[1]), pos, dtype = x_train.dtype)
x_train = np.concatenate((x_train, z), axis=1) 
z = np.full((x_test.shape[0], 31 - x_test.shape[1]), pos, dtype = x_test.dtype)
x_test = np.concatenate((x_test, z), axis=1) 

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train)
#print(y_train)

if set_nan and nan_val == 0: #신경망 학습을 위해 난값을 0로 대체하는 경우 타겟값이 0과 1인데
    y_train = y_train + 1       #0값이 난값 처리되므로 이를 피하기위해 +1을 한다.
if test_predict:
    if set_nan and nan_val == 0: 
        y_test = y_test + 1
else:
    x_test = x_train
    y_test = y_train

"""
print('TRAIN DATA')
for i, v in enumerate(y_train):
    if v == 1: print("index : {}, value: {}".format(i,v))
print()

print('TEST DATA')
for i, v in enumerate(y_test):
    if v == 1: print("index : {}, value: {}".format(i,v))
print()
"""
#x_train = np.transpose(x_train)
#y_train = np.transpose(y_train)

#predictions = autoencoder.predict(x_test)

if auto_morphic: #뉴로모픽 타겟 연결 학습할때만 의미있고 입력을 타겟으로 설정한다.
    y_train = x_train
    decode_learn = 0

import math
def truncate(num, r) -> float:
    d = 10.0 ** r
    return math.trunc(num * d) / d

def drop_nan(arr, data_x, data_y, r):
    nmax = 0
    ra, rx, ry = [], [], []
    for a, x, y in zip(arr, data_x, data_y):
        i = 0
        for j in range(arr.shape[1]):
            if np.isnan(a[j]) == 0:
                if r > 0: a[i] = truncate(a[j], r)
                else: a[i] = a[j]
                i += 1
            a[j] = np.nan if set_nan == 0 else nan_val #pick out을 신경망에 입력으로 학습시킬려면 난값은 에러나므로
        if i:
            ra.append(a)
            rx.append(x)
            ry.append(y)
            if i > nmax: nmax = i
    ra = np.array(ra)
    rx = np.array(rx)
    ry = np.array(ry)
    return nmax, ra, rx, ry
"""
def drop_nan(arr, r):
    nmax = 0
    for row in arr:
        i = 0
        for j in range(arr.shape[1]):
            if np.isnan(row[j]) == 0:
                if r > 0: row[i] = truncate(row[j], r)
                else: row[i] = row[j]
                i += 1
            row[j] = np.nan
        if i > nmax: nmax = i
    return nmax
"""


def get_rxid(data_x, figure): #외부 입력 데이터의 차원 축소 아이디 리턴
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    s = "execute mempute('discriminate', 0, {})".format(data_x.shape[1])
    rv = mp.mempute(stmt, s) #최종레벨 추출 패턴 시퀀스 길이 설정

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'disc_output 0 1 0 0 0 0')")

    if figure:
        s = "execute mempute('discriminate', 'eval_input', 100, {}, 'disc_output')".format(figure)
        r = mp.mempute(stmt, s)
        r = r[0]
        return r[0]
    else:
        r = mp.mempute(stmt, "execute mempute('discriminate', 'eval_input', 101, 0, 'disc_output')")
        r = r.astype(np.float32)
        return r

def sizeof_rxid():
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    r = mp.mempute(stmt, "execute mempute('discriminate', 102, -1)")
    
    r = r[0]
    return r[0]

def sizeof_signiden():
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    r = mp.mempute(stmt, "execute mempute('discriminate', 2, -1)")
    
    #print(r)#anormal iden size print
    r = r[0]
    return r[0]

def get_signiden(data_x):

    assert pick_out == 0

    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    rv = mp.direct(stmt, "execute mempute('phyper', 'revise_infer 1')")

    rv = mp.mempute(stmt, f"execute mempute('discriminate', 0, {pickid_num})") #최종레벨 추출 패턴 시퀀스 길이 설정

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'disc_output 0 1 0 0 0 0')")

    r = mp.mempute(stmt, "execute mempute('discriminate', 'eval_input', 1, 0, 'disc_output')")
    print('get anormal iden shape: ', r.shape)
    r = r.astype(np.float32)
    return r

def inference_signiden(signid, init):

    assert pick_out == 0

    if init:
        rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    #else:#위 함수 호출후에 세션 종료없이 현 퍼셉션에서 수행할때
    mp.mempute(stmt, f"execute mempute('discriminate', 1, {pickid_num})") #입력정보 변경, 타겟 채널 정보 복원(위 함수 
                                                                          #호출후 라면, 아니면 의미 없음)
    rv = mp.array(stmt, "execute mempute('array', 'signid_input 1 1 0 0 0 0')")
    mp.inarray(stmt, signid, 1)

    rv = mp.array(stmt, "execute mempute('array', 'eval_output 0 1 0 0 0 0')")

    r = mp.mempute(stmt, "execute mempute('predict', 'signid_input', 'eval_output')")

    return r

def get_pickout(data_x, data_y): #pick out 수행

    assert pick_out

    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    rv = mp.direct(stmt, "execute mempute('phyper', 'regularize_rate 1000')")

    rv = mp.array(stmt, "execute mempute('array', 'anormal_input')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'anormal_target')")
    mp.inarray(stmt, data_y, 1)

    rv = mp.array(stmt, "execute mempute('array', 'anormal_output 1 1 0 0 0 0')")

    #rv = mp.mempute(stmt, "execute mempute('discriminate', 4, 2)")

    pickout = mp.mempute(stmt, "execute mempute('discriminate', 'anormal_input', 103, 1, 'anormal_output', 'anormal_target')")
    #pickout = mp.mempute(stmt, "execute mempute('discriminate', 'anormal_input', 103, 1, 'anormal_output')")
    print(pickout.shape)

    #print_array(pickout, 'bb')
    print('pick out src shape: ', data_x.shape)
    nmax_col, pickout, data_x, data_y = drop_nan(pickout, data_x, data_y, 3 if just_pick else 0)

    if nmax_col > pickid_num: nmax_col = pickid_num

    pickout = pickout[:,:nmax_col]
    #print_array(pickout, 'aa')
    print('pick out max col: ', nmax_col, 'shape: ', pickout.shape)

    return nmax_col, pickout, data_x, data_y

def inference_reinforce(pick_id, init):

    assert pick_out

    if init:
        rv = mp.direct(stmt, "execute mempute('perception', 'reinforce')")

    rv = mp.array(stmt, "execute mempute('array', 'reinforce_input 1 1 0 0 0 0')")
    mp.inarray(stmt, pick_id, 1)
    
    rv = mp.array(stmt, "execute mempute('array', 'reinforce_output 0 1 0 0 0 0')")
    #rv = mp.direct(stmt, "execute mempute('display', -3)")
    r = mp.mempute(stmt, "execute mempute('predict', 'reinforce_input', 'reinforce_output')")

    return r

def predict_present(v_right, predictions, net, present, mse=None, ppmax=0):
    threshold = 1000000
    if set_nan and nan_val == 0: 
        v_right = v_right - 1
        predictions = predictions -1

    nan_cnt = norm_norm = norm_proud = proud_norm = proud_proud = 0
    prev_v = np.zeros((predictions.shape[0], predictions.shape[1]), dtype = 'i')
    for i, (y, p) in enumerate(zip(v_right, predictions)):
        if y == 0:
            if p > 0.5 or np.isnan(p): 
                prev_v[i] = 1
                norm_proud += 1
                if np.isnan(p): nan_cnt += 1
                if present: print("index : {} normal:proud nan: {}".format(i, np.isnan(p)))
            else: norm_norm += 1
        else:
            if p > 0.5 or np.isnan(p): 
                prev_v[i] = 1
                proud_proud += 1
                if np.isnan(p): nan_cnt += 1
                if present: print("index : {} proud:proud nan: {}".format(i, np.isnan(p)))
                if mse is not None and threshold > mse[i]:
                    threshold = mse[i] - 1e-7
            else:
                proud_norm += 1
                if present: print("index : {} proud:normal".format(i))

    s = "normal-normal : {} normal-proud : {} proud:normal : {} proud:proud : {} nan cnt : {}".format(norm_norm, norm_proud, proud_norm, proud_proud, nan_cnt)
    print(s)
    if net is not None: mp.print_log(net, s)
    
    accuracy = (norm_norm + proud_proud) / (norm_norm + proud_proud + norm_proud + proud_norm)
    precision = norm_norm / (norm_norm + proud_norm + 1e-7)
    recall = norm_norm / (norm_norm + norm_proud + 1e-7)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
    s = f"accuracy: {accuracy} precision: {precision} recall: {recall} f1 score: {f1}"
    print(s)
    if net is not None: mp.print_log(net, s)

    precision = proud_proud / (proud_proud + proud_norm + 1e-7)
    recall = proud_proud / (proud_proud + norm_proud + 1e-7)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
    s = f"reverse precision: {precision} recall: {recall} f1 score: {f1}"
    print(s)
    if net is not None: mp.print_log(net, s)

    if mse is None or proud_proud < ppmax:
        threshold = 0

    if present < 0:
        return threshold
    #predictions = predictions.astype('int32')
    predictions = prev_v

    #for i, v in enumerate(predictions):
    #    if v > 0.5: predictions[i] = 1
    #    else: predictions[i] = 0
    conf_matrix = confusion_matrix(v_right, predictions)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    return threshold

def discriminator(ytag, vright, predictions, mul, mse = None, threshold = 0):
    if mse is None:
        mse = np.mean(np.power(vright - predictions, 2), axis=1)#[batch, seq]
    if threshold == 0:
        rcount = 0
        for r in ytag:
            if r[0] == 1: rcount += 1
        rcount = int(rcount * mul)
        mse2 = sorted(mse, reverse=True)
        threshold = mse2[rcount]
            
    y_pred = [1 if e > threshold else 0 for e in mse]
    y_pred = np.array(y_pred)
    y_pred = np.expand_dims(y_pred, axis=-1)

    return y_pred, threshold, mse

def reconst_present(x_test, v_right, predictions):

    if set_nan and nan_val == 0: 
        v_right = v_right - 1
        predictions = predictions -1

    mse = np.mean(np.power(x_test - predictions, 2), axis=1)

    y_pred = [1 if e > threshold else 0 for e in mse]
    y_pred = np.array(y_pred)
    y_pred = np.expand_dims(y_pred, -1)
    conf_matrix = confusion_matrix(v_right, y_pred)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

xtrain_other = None
seq_len = x_train.shape[1]
if auto_regress or decode_xother is False:
    seq_len = seq_len + 1
    xtrain_other = x_train
    z = np.full((x_train.shape[0], 1), bos, dtype = x_train.dtype)
    x_train = np.concatenate((z, x_train), axis=1) #go mark
    print(x_train.shape)
    xtrain_other = np.concatenate((xtrain_other, z), axis=1) #end mark
    xtrain_other = np.expand_dims(xtrain_other, -1)
    dec_infeat_other = 1

if decode_xother:
    xtrain_other = x_train * dx_mul
    xtest_other = x_test * dx_mul
    xtrain_other = xtrain_other.astype(np.int32)
    xtest_other = xtest_other.astype(np.int32)
    xtrain_other = xtrain_other.astype(x_train.dtype)
    xtest_other = xtest_other.astype(x_train.dtype)
    #print_array(x_train, None, 10)
    #print_array(xtrain_other, None, 10)
    xtrain_other = xtrain_other - minv
    xtest_other = xtest_other - minv
    #print_array(xtrain_other, None, 10)
    if embedding:
        x_train = xtrain_other
        x_train = x_train.astype(np.int64)
        x_test = xtest_other
        x_test = x_test.astype(np.int64)
    xtrain_other = np.expand_dims(xtrain_other, -1)

inp_size = x_train.shape[-1]
gid = int(sys.argv[2])
model_name = sys.argv[4]



def make_mname(model_name):
    model_name = f"{model_name}_isz_{inp_size}_hsz_{hidden_sz}\
                _npair_{n_epair}\
                _ar_{auto_regress}_nblock_{nblock}\
                _nhead_{n_heads}\
                _dff_{ffn_hidden}_gpt_{gpt_model}\
                _posenc_{pos_enc}\
                _deatt_{deatt}_dec_infeat_other_{dec_infeat_other}\
                _precision_{precision}_cut_over_{cut_over}\
                _qlearn_lev_{qlearn_lev}_qgen_lev_{qgen_lev}\
                _qak_{qa_kernel}_ifk_{infini_kernel}"
    model_name = model_name.replace(' ', '')
    print(model_name)
    return model_name

model_name = make_mname(model_name)

#시나리오
#python anormal.py 1 0 0 anormal 1 1
#python anormal.py 2 0 0 anormal 2 1 1      0,1 두개레벨 학습
#python anormal.py 9 0 0 anormal 1 0 0      1레벨 이상 삭제, 0레벨부터 오버랩 학습이므로 이전레벨없어 복원없음
#python anormal.py 2 0 0 anormal 2 1 1      0레벨 오버랩 학습, 1레벨 다시 학습
#python anormal.py 5 0 0 anormal 0 2

if sys.argv[1] == '1':# command gid port perception-name decode_lev_opt mse, ex. 1 0 0 anormal 1 1
    
    if len(sys.argv) >= 6:
        dec_lev_opt = int(sys.argv[5]) #음수이면 인코더 튜닝학습까지 선행한 후에 -1이면 뉴로모픽 타겟으로
        #디코더 레벨학습, -2이면 인코더 추론을 타겟으로 디코더 레벨학습, 0이면 인코더 레벨 학습후에 디코더 레벨 학습
        #안함, 1이면 인코더 레벨 학습후 바로 뉴로모픽 타겟으로 디코더 레벨학습, 2이면 인코더 레벨 학습후 바로 
        #인코더 추론을 타겟으로으로 디코더 레벨학습


    if len(sys.argv) == 7: mse = int(sys.argv[6])
    else: mse = 0

    train_params = dict(
        input_size=inp_size,
        hidden_sz=hidden_sz,
        learn_rate=1e-4,
        drop_rate=0.1,
        signid_mse = mse,
        wgt_save = 0,
        layer_norm = 1,
        n_epair = n_epair,#3, #encode depth 개수, 1부터 시작
        residual = 1,
        on_schedule = False,
        dtype = mp.tfloat,
        levelhold = 0.7,
        tunehold = 0.98,
        seed = 777, #-1,
        decay = 0.0001,
        decode_active = 1,
        fast_once = 1,
        nblock = nblock,
        nontis = -1,
        gpt_model = gpt_model,
        pos_encoding = pos_enc,
        n_heads = n_heads,
        d_ff = ffn_hidden,
        regression = auto_regress,
        dec_lev_learn = dec_lev_opt, #0 if auto_regress is True else dec_lev_opt,
        decode_infeat = dec_infeat_other,
        size_embed = dec_infeat_other + 1 if embedding else 0, #+1 은 패딩 추가
        batch_size=batch_size)

    train_params = default_param(**train_params)
    #train_params['aaaa11'] = 0.5 #rfilter
    #train_params['aaaa12'] = True 
    #train_params['least_accuracy'] = 0.0
    #train_params['aaa8'] = 0 #approximate
    #train_params['aaaa13'] = 1 
    #train_params['aaaa15'] =-1#0#-1
    if auto_regress == 0:
        train_params['aaaa16'] = 64 #차원 고정 안되게
    #train_params['aaaa18'] = 8 
    #train_params['aaaa19'] = 4 
    #train_params['aaaaa10'] = 1
    train_params['aaaaa11'] = deatt #deatten
    #train_params['aaaaa12'] = 2 #incre_lev
    train_params['aaaa17'] = 0 #oneway
    #train_params['aaaaa13'] = 1
    #train_params['aaaaa14'] = 4
    #train_params['aaaaa15'] = 1
    train_params['aaaaa16'] = 0#-10 #npk
    #train_params['aaaaa17'] = 1e-3#0.001 #spk_lr
    train_params['aaaaa18'] = 1.0 #reduce_svar
    #train_params['aaaaa19'] = 4 #glam
    train_params['aaaaaa10'] = 0.5 #rspk_lr
    #train_params['aaaaaa11'] = 1 #monitor
    #train_params['aaaaa15'] = 1
    #train_params['aaaaaa13'] = 8
    #train_params['aaaaaa14'] = 50 #sampling

    train_params['aaaaaa15'] = qa_kernel
    train_params['aaaaaa16'] = precision
    train_params['aaaaaa17'] = cut_over
    train_params['aaaaaa18'] = 48000
    train_params['aaaaaa19'] = 32
    #train_params['aaaaa17'] = 1e-7 #바닥만 양자 연산
    train_params['aaaaa19'] = qlearn_lev #양자 정보 학습 레벨 설정, 음수이면 모든 레벨 양자학습
    train_params['aaaaaa14'] = qgen_lev #0이면 양자 학습 설정 레벨에서 양자 정보 생성, 1이면 바닥 레벨에서 양자 정보 생성, 음수이면 디코더에서 생성
                                                                        #양자 학습은 위 설정된 레벨에서 학습
                                 #위 함께 -1, 0 이면 이면 모든 레벨에서 양자 정보 생성하고 양자학습
    train_params['aaaaaa11'] = 1 #monitor

    train_params['aaaaa12'] = infini_kernel

    param = param2array(**train_params)
    if param is None:
        exit()
    net = mp.neuronet(param, gid, model_name)
    #mp.neurogate(net, stmt, model_name, sensor, x_train, y_train, 1)
    mp.close_net(net)
else:
    param = mp.load_param(model_name)
    train_params = arr2py_param(param)
    #train_params['drop_rate'] = 0.0
    #train_params['aaaa13'] = 0 #prePooling
    #train_params['aaa7'] = 0 #multis
    #train_params['seed'] = -1
    #train_params['aaaa12'] =-1
    #train_params['hidden_sz'] = 1
    #train_params['aaaaa17'] = -1#1e-7 #양자 어텐션 어텐션만 수행
    param = param2array(**train_params)
    mp.update_param(param, model_name)
    
    net = mp.loadnet(gid, model_name)

    #mp.neurogate(net, stmt, model_name, sensor, x_train, y_train, 0)

    if sys.argv[1] == '2':# command gid port perception-name level epoch relay, trainnig

        #relay 0 - not, ex. 2 0 0 anormal 3 2 , 레벨 3개 까지 학습, 즉 6레벨까지
        #1 - 레벨단위 이어 학습, ex. 2 0 0 anormal 1 2 1, 연결 학습하지 않고 종료
        #2 - 레벨내에서 이어 학습, ex. 2 0 0 anormal 1 2 2, 연결 학습하지 않고 종료
        #3 - 레벨을 증가하여 이어 학습하여 연결 학습까지 완결, ex. 2 0 0 anormal 1 2 3, 레벨을 0주면 아래 튜닝학습과 동일 3은 무시
        #4 - 현 레벨부터 이어 학습하여 연결 학습까지 완결, ex. 2 0 0 anormal 0 1 2 4, 레벨을 0주면 아래 튜닝학습과 동일 4은 무시
        #-1 - 이어서 학습이 아니면서 지정 레벨까지 한번에 레벨단위 학습한후 다음 튜닝 학습을 위해 
        #       타겟 연결 학습하지 않고 종료 ex. 2 0 0 anormal 2 2 -1
        #튜닝학습은 레벨을 0로 수행 - 2 0 0 anormal 0 2 
        #far_tuning학습은 마지막에 반드시 튜닝 학습을 호춣해야 한다. 이유는 morphic_to의 xtrain주석 참조
        #시나리오 
        #python hanormal.py 1 0 0 anormal 1 1 - 초기화
        #python hanormal.py 2 0 0 anormal 1 2 2 - 이어서 현 레벨 학습 실행
        #python hanormal.py 2 0 0 anormal 1 2 2 - 이어서 현 레벨 반복 학습 실행, 뉴로모픽 레벨 증가하지 않음
        #python hanormal.py 2 0 0 anormal 1 2 1 - 이어서 다음 상위 레벨 학습 실행, 뉴로모픽 레벨 증가
        #python hanormal.py 2 0 0 anormal 0 2 -1 - 최초 입력 레벨에서 위에서 학습된 최종 레벨까지 튜닝학습후 연결 학습 없이 종료
        #                                       반복하면 튜닝학습만 이어서 학습
        #python hanormal.py 2 0 0 anormal 0 2   - 이어서 튜닝학습후 연결학습까지 수행 or
        #python hanormal.py 5 0 0 anormal 0 2   - 튜닝학습 않고 연결학습만 수행

        level = int(sys.argv[5]) #레벨 학습 단계 설정, 1부터 시작
        epoch = int(sys.argv[6])
        if len(sys.argv) >= 8: relay = int(sys.argv[7])
        else: relay = 0

        if len(sys.argv) >= 9: tune_level = int(sys.argv[8])
        else: tune_level = 1

        if len(sys.argv) == 10: repeat = int(sys.argv[9])
        else: repeat = 1

        if level: tuning = 0 #레벨 단위 학습의 반복 레밸 횟수가 주어졋으면 레벨단위 학습, 1이면 한개 레벨만 학습
        else: 
            tuning = 1
            level = 1
        #print_array(x_train, "train")
        s = f"\nhanormal train:  EPOCH {epoch} RELAY {relay} LEVEL {level} REPEAT {repeat} TUNING {tuning} DECODE_LEV_LEARN {train_params.get('dec_lev_learn')}\n"
        print(s)
        mp.print_log(net, s)
        count = 0
        if repeat > 1: relay = 0
        while count < repeat:
            present = 10 if count + 1 == repeat else -1 #마지막 반복에서만 프린트
            reent = np.expand_dims(x_train, -1)
            for i in range(level):
                name = f'aaa_{i}'
                reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay, 0, name, present)

            x_test1 = np.expand_dims(x_test, -1)
            predictions = mp.xpredict(net, x_test1, 0, -1, 'bbb', present)

            print(x_test1.shape)
            print(y_test.shape)
            print(predictions.shape)
        
            if auto_morphic:
                y_pred, threshold, mse = discriminator(y_test, x_test, predictions, inner_mul)
                s = f'{repeat} REPEAT: {count} THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
            else:
               mse = None
            threshold = predict_present(y_test, y_pred, net, present, mse, jit_count)
            if auto_morphic and threshold > 0:
                y_pred, _, _ = discriminator(y_test, x_test, predictions, inner_mul, mse, threshold)
                s = f'optim THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

                y_pred, threshold, _ = discriminator(y_test, x_test, predictions, outer_mul, mse)
                s = f'outer THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                threshold = predict_present(y_test, y_pred, net, present, mse)

                y_pred, _, _ = discriminator(y_test, x_test, predictions, outer_mul, mse, threshold)
                s = f'outer optim THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

            count += 1
            if present == 0:#반복 중간이면 최종 레벨 이상 삭제
                mp.truncate(net, tune_level, 0)

    elif sys.argv[1] == '3':# command gid port perception-name, prediction

        x_test = np.expand_dims(x_test, -1)
        predictions = mp.xpredict(net, x_test, 0, -1, 'bbb', 100)

        print(x_test.shape)
        print(y_test.shape)
        print(predictions.shape)
                
        predict_present(y_test, predictions, net, 1)

    elif sys.argv[1] == '5':#anormal.py 5 0 0 anormal 0 10
        assert train_params.get('decode_active') > 0, "decode active setting lack"
        level = int(sys.argv[5]) #레벨 학습 단계 설정, 1부터 시작
        epoch = int(sys.argv[6])
        if len(sys.argv) >= 8: repeat = int(sys.argv[7])
        else: repeat = 1

        if len(sys.argv) >= 9: relay = int(sys.argv[8])
        else: relay = 0 

        if len(sys.argv) == 10: decord_mode = int(sys.argv[9])
        else: decord_mode = 1 #현수행이 디코드 레벨학습이면 true(1,2,3) or false(0)만 의미있다

        if level: 
            tuning = 0 #레벨 단위 학습의 반복 레밸 횟수가 주어졋으면 레벨단위 학습, 1이면 한개 레벨만 학습
            if (auto_regress or decode_xother is False) and (relay == 3 or relay == 4):
                z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
                x_test = np.concatenate((z, x_test), axis=1) #go mark
        else: 
            tuning = 1
            level = 1
            if auto_regress or decode_xother is False:
                z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
                x_test = np.concatenate((z, x_test), axis=1) #go mark

        s = f"\nhanormal decord train:  EPOCH {epoch} RELAY {relay} DECODE {decord_mode} LEVEL {level} REPEAT {repeat} TUNING {tuning}  DECODE_LEV_LEARN {train_params.get('dec_lev_learn')}\n"
        print(s)
        mp.print_log(net, s)
        #print_array(x_train, "train")
        count = 0
        if auto_regress:
            n_input = int(x_test.shape[1] / 2)
        else: n_input = -1
        if repeat > 1: relay = 0
        while count < repeat:
            present = 10 if repeat == 1 or count + 1 == repeat else -1 #마지막 반복에서만 프린트
            reent = np.expand_dims(x_train, -1)
            for i in range(level):
                name = f'aaa_{i}'
                reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay, decord_mode)#, name, present)

            x_test1 = np.expand_dims(x_test, -1)
            predictions = mp.xpredict(net, x_test1, decord_mode, n_input)#, 'bbb', present)
            print(x_test.shape)
            print(predictions.shape)

            if False:#tuning and auto_regress:
                x_test2 = x_test[:,1:]
                predictions = predictions[:,:-1]
            elif decode_xother:
                x_test2 = xtest_other
            else:
                x_test2 = x_test

            y_pred, threshold, mse = discriminator(y_test, x_test2, predictions, inner_mul)
            s = f'{repeat} REPEAT: {count} THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
            print(s)
            mp.print_log(net, s)
            threshold = predict_present(y_test, y_pred, net, present, mse, jit_count)

            if threshold > 0:#위 27개 이상의 정답을 맞추면 위 과정에서 획득된 최적 임계치로 다시 정답획득하여 오류개수를 줄인다.
                y_pred, _, _ = discriminator(y_test, x_test2, predictions, inner_mul, mse, threshold)
                s = f'optim THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)
                #정답의 11배 아웃터로 임계치를 획득하고 최적임계치를 획득하여
                y_pred, threshold, _ = discriminator(y_test, x_test2, predictions, outer_mul, mse)
                s = f'outer THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                threshold = predict_present(y_test, y_pred, net, present, mse)#최적임계치 획득
                #최적임계치로 오류를 줄인다.
                y_pred, _, _ = discriminator(y_test, x_test2, predictions, outer_mul, mse, threshold)
                s = f'outer optim THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

            count += 1
            """
            error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': Y_test})
            y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
            conf_matrix = confusion_matrix(error_df.true_class, y_pred)
            plt.figure(figsize=(12, 12))
            sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
            plt.title("Confusion matrix")
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.show()
            """
       

    elif sys.argv[1] == '6':
        if len(sys.argv) >= 6:
            decode = int(sys.argv[5])
        else:
            decode = 1
        if len(sys.argv) >= 7:
            inner_mul = int(sys.argv[6])

        if len(sys.argv) >= 8:
            jit_count = int(sys.argv[7])
    

        if auto_regress or decode_xother is False:
            z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
            x_test = np.concatenate((z, x_test), axis=1) #go mark
            n_input = int(x_test.shape[1] / 2)
        else: n_input = -1
        x_test1 = np.expand_dims(x_test, -1)
        predictions = mp.xpredict(net, x_test1, decode, n_input)#, 'bbb', 100) #decord
        print(x_test.shape)
        print(predictions.shape)

        if False:
            x_test = x_test[:,1:]
            predictions = predictions[:,:-1]
        elif decode_xother:
            x_test = xtest_other

        y_pred, threshold, mse = discriminator(y_test, x_test, predictions, inner_mul)
        print('THRESHOLD: ', threshold, 'DEOCDE MODE: ', decode)
        #threshold = 0.03 #decord 2
        #threshold = 23.5 #decord 3
        threshold = predict_present(y_test, y_pred, net, 1, mse, jit_count)

        if threshold > 0:
            y_pred, _, _ = discriminator(y_test, x_test, predictions, inner_mul, mse, threshold)
            s = f'optim THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            predict_present(y_test, y_pred, net, 1)

            y_pred, threshold, _ = discriminator(y_test, x_test, predictions, outer_mul, mse)
            s = f'outer THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            threshold = predict_present(y_test, y_pred, net, 1, mse)

            y_pred, _, _ = discriminator(y_test, x_test, predictions, outer_mul, mse, threshold)
            s = f'outer optim THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            predict_present(y_test, y_pred, net, 1)

        """
        error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': Y_test})
        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
        """

#auto encoder 방식 
# 4레벨 10번 반복 학습 & 튜닝 학습, 리지주얼 false
# python hanormal_to.py 5 2 4647 anormal 4 10 3; python hanormal_to.py 5 2 4647 anormal 0 10
#False_3_16_False_0.1_0.0001, 디코더 튜닝학습 순서 2, 3, 2, threshold = 2.9
#normal-normal : 19375 normal-proud : 591 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9702 precision: 0.9997420020639834 recall: 0.9703996794550737 f1 score: 0.984852335688507
#reverse precision: 0.8529411764705882 recall: 0.0467741935483871 f1 score: 0.08868501529051988
#python repeat.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10
#lev: 3 epoch: 100 train mean loss: 0.7188213655772882 accuracy: 649.8771362304688 threshold: 0.99
#normal-normal : 19505 normal-proud : 461 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9767 precision: 0.9997437211686314 recall: 0.9769107482720625 f1 score: 0.9881953592055933
#reverse precision: 0.8529411764705882 recall: 0.05918367346938776 f1 score: 0.11068702290076335
#대략 900번 반복 학습
#lev: 3 epoch: 100 train mean loss: 0.7264487208464206 accuracy: 642.620849609375 threshold: 0.99
#normal-normal : 19567 normal-proud : 399 proud:normal : 4 proud:proud : 30 nan cnt : 0
#accuracy: 0.97985 precision: 0.9997956159623933 recall: 0.9800160272463188 f1 score: 0.9898070162126615
#reverse precision: 0.8823529411764706 recall: 0.06993006993006994 f1 score: 0.12958963282937366

#False_3_16_False_0.1_0.0001, 디코더 튜닝학습 2 모드  ,4레벨 500번 학습, 60번 튜닝 반복 학습
#lev: 3 epoch: 50 train mean loss: 1.0012260127143982 error: 761.515869140625
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411764705882 recall: 0.04084507042253521 f1 score: 0.07795698924731183

#False_3_16_False_0.1_0.0001, python repeat2.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 500 200
#디코더 튜닝학습 2 모드 20번 반복, 계속 normal-proud 계속 조금씩 적어짐
#lev: 3 epoch: 100 train mean loss: 0.8867864813178014 accuracy: 711.3467407226562 threshold: 0.99
#normal-normal : 19481 normal-proud : 485 proud:normal : 6 proud:proud : 28 nan cnt : 0
#accuracy: 0.97545 precision: 0.9996921024272591 recall: 0.9757087047981569 f1 score: 0.9875548120548501
#reverse precision: 0.8235294117647058 recall: 0.05458089668615984 f1 score: 0.10237659963436929

#False_3_16_False_0.1_0.0001, python repeat2.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 500 200
#디코더 튜닝학습 1 모드 4번 반복,
#lev: 3 epoch: 100 train mean loss: 0.7715653893657219 error: 658.2728271484375 threshold: 0.99
#normal-normal : 19523 normal-proud : 443 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9776 precision: 0.9997439573945105 recall: 0.9778122808774917 f1 score: 0.988656504785537
#reverse precision: 0.8529411764705882 recall: 0.0614406779661017 f1 score: 0.11462450592885376

#False_3_16_False_0.3_0.0001, python repeat2.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 100 100
#디코더 튜닝학습 1 모드, 반복해도 그대로,
#lev: 3 epoch: 100 train mean loss: 1.0009864177077243 error: 761.520751953125 threshold: 0.99
#normal-normal : 19286 normal-proud : 680 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.96575 precision: 0.9997408117775128 recall: 0.9659421015726736 f1 score: 0.9825508826451335
#reverse precision: 0.8529411764705882 recall: 0.04090267983074753 f1 score: 0.07806191117092866

#False_3_32_False_0.3_0.0001, python repeat2.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 500 200
#디코더 튜닝학습 1 모드, 반복해도 그대로,
#lev: 3 epoch: 100 train mean loss: 1.0009911043139605 error: 761.5249633789062 threshold: 0.99
#normal-normal : 19286 normal-proud : 680 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.96575 precision: 0.9997408117775128 recall: 0.9659421015726736 f1 score: 0.9825508826451335
#reverse precision: 0.8529411739619377 recall: 0.040902679824978465 f1 score: 0.07806191114991605
#-----------------------------------------------------------------------------
#False_3_16_False_0.1_0.0001, python repeat2.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 5 5
#normal-normal : 19286 normal-proud : 680 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.96575 precision: 0.9997408117775128 recall: 0.9659421015726736 f1 score: 0.9825508826451335
#reverse precision: 0.8529411739619377 recall: 0.040902679824978465 f1 score: 0.07806191114991605

#False_6_16_False_0.1_0.0001, 반복해도 그대로, python repeat2.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 5 5
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411739619377 recall: 0.040845070416782384 f1 score: 0.07795698922635565
#lev: 3 epoch: 10 train mean loss: 1.0013631843985655 error: 761.516845703125
#python repeat2.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 20 20, 위와 동일
#------------------------------------------------------------------------------------
#False_3_16_0_False_0.1_0.0001, False_3_32_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 1
#lev: 3 epoch: 10 train mean loss: 1.001331278719963 error: 761.5134887695312
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411739619377 recall: 0.040845070416782384 f1 score: 0.07795698922635565

#False_3_16_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 100 0 4 10 1
#lev: 3 epoch: 100 train mean loss: 0.7707902386975594 error: 698.9960327148438 threshold: 0.99
#normal-normal : 19535 normal-proud : 431 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9782 precision: 0.9997441146366428 recall: 0.9784133026144446 f1 score: 0.988963701716195
#reverse precision: 0.8529411739619377 recall: 0.06304347824716446 f1 score: 0.1174088940172775

#False_3_32_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 100 0 4 10 1
#lev: 3 epoch: 100 train mean loss: 1.000958967094238 error: 761.5145263671875 threshold: 0.99
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411739619377 recall: 0.040845070416782384 f1 score: 0.07795698050425

#False_3_32_0_False_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 100 0 4 10 1
#lev: 3 epoch: 100 train mean loss: 1.0009889667614913 error: 761.5250854492188 threshold: 0.99
#normal-normal : 19286 normal-proud : 680 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.96575 precision: 0.9997408117775128 recall: 0.9659421015726736 f1 score: 0.9825508826451335
#reverse precision: 0.8529411739619377 recall: 0.040902679824978465 f1 score: 0.07806190241663427

#False_3_16_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 100 0 4 10 2
#ev: 3 epoch: 100 train mean loss: 1.0009683288442783 error: 761.5147094726562 threshold: 0.99
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411739619377 recall: 0.040845070416782384 f1 score: 0.07795698050425

#False_3_16_2_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 1
#lev: 3 epoch: 10 train mean loss: 1.0013045156613374 error: 761.5133666992188
#normal-normal : 19285 normal-proud : 681 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.9657 precision: 0.9997407983411094 recall: 0.9658920164279274 f1 score: 0.9825249643366619
#reverse precision: 0.8529411739619377 recall: 0.040845070416782384 f1 score: 0.07795698050425

#False_3_16_2_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2
#lev: 3 epoch: 10 train mean loss: 0.8625035922114666 error: 704.631103515625
#normal-normal : 19489 normal-proud : 477 proud:normal : 6 proud:proud : 28 nan cnt : 0
#accuracy: 0.97585 precision: 0.9996922287766093 recall: 0.9761093859561254 f1 score: 0.9877600669014978
#reverse precision: 0.8235294093425606 recall: 0.055445544543476125 f1 score: 0.10389609203740997

#=======================================================================
#False_3_16_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 1 10 10 10
#python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 3
#False_3_16_0_False_0.3_0.0001
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0
#accuracy: 0.9914 precision: 0.9990922844125109 recall: 0.9922868877041355 f1 score: 0.9956779075792592
#reverse precision: 0.47058823391003457 recall: 0.09411764700346022 f1 score: 0.15686271716647932

#False_3_16_2_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 10 10 10
# python repeat.py hanormal_to.py 5 2 4647 anormal 0 100 0 4 10 2
#lev: 3 epoch: 100 train mean loss: 0.8663769903091284 error: 700.6859741210938 threshold: 0.99
#normal-normal : 19815 normal-proud : 151 proud:normal : 15 proud:proud : 19 nan cnt : 0
#accuracy: 0.9917 precision: 0.9992435703429186 recall: 0.9924371431383731 f1 score: 0.9958286765007582
#reverse precision: 0.5588235277681661 recall: 0.111764705816609 f1 score: 0.18627448184352582

#False_3_16_2_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 1 10 10 10
#python repeat.py hanormal_to.py 5 2 4647 anormal 0 100 0 4 10 1
#lev: 3 epoch: 100 train mean loss: 1.0012097236437676 error: 761.5154418945312 threshold: 0.99
#THRESHOLD: 10.43299388885498 DECODE MODE: 1
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0
#accuracy: 0.9914 precision: 0.9990922844125109 recall: 0.9922868877041355 f1 score: 0.9956779075792592
#reverse precision: 0.47058823391003457 recall: 0.09411764700346022 f1 score: 0.15686271716647932

#False_3_16_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#dec_lev_learn = 0, 로스 줄지 않음, bad
#lev: 3 epoch: 50 train mean loss: 1.0011826327595956 error: 761.5150146484375
#THRESHOLD: 10.433550834655762 DECODE MODE: 2
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0

#False_3_16_0_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 10 10 10
#dec_lev_learn = 0, 디코드 레벨학습 3, 로스 줄지 않음, bad
#lev: 3 epoch: 50 train mean loss: 1.0011767385861812 error: 761.5151977539062
#THRESHOLD: 10.433455467224121 DECODE MODE: 2
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0























#auto encoder 방식 , 디폴트 디코드 튜닝 학습 모드 2
# 4레벨 10번 반복 학습 & 튜닝 학습, 리지주얼 true
# python hanormal_to.py 5 2 4647 anormal 4 10 3; python hanormal_to.py 5 2 4647 anormal 0 10
#False_3_16_True_0.1_0.0001, 디코더 튜닝학습 순서 2, 3, 2, threshold = 0.08
#normal-normal : 19815 normal-proud : 151 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9921 precision: 0.9996468570275452 recall: 0.9924371431433436 f1 score: 0.9960289534533024
#reverse precision: 0.7941176470588235 recall: 0.15168539325842698 f1 score: 0.25471698113207547
# 디코더 1 튜닝 추가 학습, threshold = 0.01
#normal-normal : 19790 normal-proud : 176 proud:normal : 8 proud:proud : 26 nan cnt : 0
#디코더 튜닝학습 순서 3, 2, 2, threshold = 0.08
#normal-normal : 19802 normal-proud : 164 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9914 precision: 0.9995961635537607 recall: 0.9917860362616447 f1 score: 0.9956757843925986
#reverse precision: 0.7647058823529411 recall: 0.1368421052631579 f1 score: 0.23214285714285718
# 디코더 1 튜닝 추가 학습, threshold = 0.08
#normal-normal : 19953 normal-proud : 13 proud:normal : 23 proud:proud : 11 nan cnt : 0
#accuracy: 0.9982 precision: 0.9988486183420104 recall: 0.9993488931183011 f1 score: 0.9990986931050022
#reverse precision: 0.3235294117647059 recall: 0.4583333333333333 f1 score: 0.3793103448275862
#threshold = 0.008
#normal-normal : 19817 normal-proud : 149 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.99215 precision: 0.9995964691046658 recall: 0.9925373134328358 f1 score: 0.9960543841572215
#reverse precision: 0.7647058823529411 recall: 0.14857142857142858 f1 score: 0.24880382775119617
#디코더 튜닝학습 순서 2, 2, 2, threshold = 0.08
#normal-normal : 19827 normal-proud : 139 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9926 precision: 0.9995462794918331 recall: 0.9930381648802965 f1 score: 0.9962815938897542
#reverse precision: 0.7352941176470589 recall: 0.1524390243902439 f1 score: 0.2525252525252525
# 디코더 2 튜닝 추가 학습, threshold = 0.08
#normal-normal : 19859 normal-proud : 107 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9942 precision: 0.9995470102677673 recall: 0.9946408895121707 f1 score: 0.9970879148466134
#reverse precision: 0.7352941176470589 recall: 0.1893939393939394 f1 score: 0.30120481927710846
#대략 6번 추가 반복 학습
#lev: 3 epoch: 10 train mean loss: 0.00727341588264188 error: 53.38076400756836
#normal-normal : 19877 normal-proud : 89 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9951 precision: 0.9995474202956854 recall: 0.9955424221175999 f1 score: 0.9975409013349392
#reverse precision: 0.7352941176470589 recall: 0.21929824561403508 f1 score: 0.33783783783783783
#False_3_16_True_0.3_0.0001 8번 repeat, 드롭아웃 증대
#lev: 3 epoch: 10 train mean loss: 0.02298430479071939 error: 70.86676788330078
#normal-normal : 19819 normal-proud : 147 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9922 precision: 0.9995460964292919 recall: 0.992637483722328 f1 score: 0.9960798110267879
#reverse precision: 0.7352941176470589 recall: 0.14534883720930233 f1 score: 0.24271844660194175
#False_6_16_True_0.1_0.0001 5번 튜닝 반복
#lev: 3 epoch: 10 train mean loss: 0.008556514756025698 error: 57.037723541259766
#normal-normal : 19828 normal-proud : 138 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.99265 precision: 0.9995463023642688 recall: 0.9930882500250425 f1 score: 0.9963068110443936
#reverse precision: 0.7352941176470589 recall: 0.15337423312883436 f1 score: 0.2538071065989848

#False_6_16_True_0.2_0.0001 5번 튜닝 반복
#lev: 3 epoch: 10 train mean loss: 0.00821291346502944 error: 54.33340835571289
#normal-normal : 19862 normal-proud : 104 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.99435 precision: 0.9995470786573398 recall: 0.9947911449464089 f1 score: 0.9971634410221653
#reverse precision: 0.7352941176470589 recall: 0.1937984496124031 f1 score: 0.3067484662576687


#False_6_16_True_0.3_0.0001, 9번 튜닝 반복, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10
#lev: 3 epoch: 10 train mean loss: 0.007116427582402069 error: 66.51753997802734
#normal-normal : 19880 normal-proud : 86 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.99525 precision: 0.9995474885615164 recall: 0.9956926775518381 f1 score: 0.9976163593024715
#reverse precision: 0.7352941176470589 recall: 0.22522522522522523 f1 score: 0.3448275862068966


#False_6_16_True_0.4_0.0001, 10번 튜닝 반복, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10
#lev: 3 epoch: 10 train mean loss: 0.006304458321000521 error: 81.81111145019531
#normal-normal : 19871 normal-proud : 95 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.99485 precision: 0.9995975652698827 recall: 0.9952419112491235 f1 score: 0.997414983059354
#reverse precision: 0.7647058823529411 recall: 0.21487603305785125 f1 score: 0.33548387096774196777

#False_6_16_True_0.4_0.0001, 8번 튜닝 반복, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 100 50
#lev: 3 epoch: 10 train mean loss: 0.010913052957337827 error: 66.01915740966797
#normal-normal : 19814 normal-proud : 152 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.992 precision: 0.9995964080314802 recall: 0.9923870579985976 f1 score: 0.995978687041319
#reverse precision: 0.7647058823529411 recall: 0.14606741573033707 f1 score: 0.2452830188679245

#False_6_16_True_0.4_0.0001, 4번 튜닝 반복, 디코더 튜닝학습 1 모드, python repeat.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 3
#lev: 3 epoch: 10 train mean loss: 0.00017014328365733562 error: 131.80918884277344
#normal-normal : 19966 normal-proud : 0 proud:normal : 34 proud:proud : 0 nan cnt : 0
#accuracy: 0.9983 precision: 0.9983 recall: 1.0 f1 score: 0.9991492768853525

#False_3_32_True_0.4_0.0001, 디코더 튜닝학습 1 모드, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 500 200
#결과 bad

#False_3_32_True_0.4_0.0001, 9번 튜닝 반복 , python repeat2.py hanormal_to.py 5 2 4645 anormal 0 10 0 4 10
#lev: 3 epoch: 10 train mean loss: 0.0034644867409951985 error: 41.31746292114258
#normal-normal : 19944 normal-proud : 22 proud:normal : 20 proud:proud : 14 nan cnt : 0
#accuracy: 0.9979 precision: 0.9989981967541575 recall: 0.9988981268155865 f1 score: 0.9989481592787378
#reverse precision: 0.4117647058823529 recall: 0.3888888888888889 f1 score: 0.39999999999999997
#--------------------------------------------------------------------------------
#False_3_16_True_0.1_0.0001, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 5 5
#lev: 3 epoch: 10 train mean loss: 0.0032204453675195766 error: 34.75234603881836
#normal-normal : 19934 normal-proud : 32 proud:normal : 17 proud:proud : 17 nan cnt : 0
#accuracy: 0.99755 precision: 0.9991479123853441 recall: 0.9983972753681258 f1 score: 0.9987724528396423
#reverse precision: 0.49999999852941174 recall: 0.3469387748021658 f1 score: 0.4096385532297866

#False_6_16_True_0.1_0.0001, 반복해도 대략 그래로, python repeat2.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 5 5
#lev: 3 epoch: 100 train mean loss: 0.0034113973058246747 error: 37.20463180541992 threshold: 0.99
#normal-normal : 19912 normal-proud : 54 proud:normal : 15 proud:proud : 19 nan cnt : 0
#accuracy: 0.99655 precision: 0.999247252471521 recall: 0.9972954021837123 f1 score: 0.9982703732484395
#reverse precision: 0.5588235277681661 recall: 0.2602739722462001 f1 score: 0.35514018625207444

#False_6_16_True_0.4_0.0001, 반복해도 대략 그래로, python repeat2.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 5 5
#lev: 3 epoch: 100 train mean loss: 0.003467783084115348 error: 82.95068359375 threshold: 0.99
#normal-normal : 19921 normal-proud : 45 proud:normal : 16 proud:proud : 18 nan cnt : 0
#accuracy: 0.99695 precision: 0.9991974720369163 recall: 0.9977461684864269 f1 score: 0.9984712928852468
#reverse precision: 0.5294117631487889 recall: 0.285714285260771 f1 score: 0.3711340198533319

#False_3_32_True_0.1_0.0001, python repeat2.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 5 5
#lev: 3 epoch: 10 train mean loss: 0.0026972743828828707 error: 31.35028839111328
#normal-normal : 19941 normal-proud : 25 proud:normal : 21 proud:proud : 13 nan cnt : 0
#accuracy: 0.9977 precision: 0.9989480012022843 recall: 0.9987478713813482 f1 score: 0.998847926267281
#reverse precision: 0.3823529400519031 recall: 0.3421052622576177 f1 score: 0.3611111101080247

#False_3_16_True_0.1_0.0001, python repeat2.py 계속 내려감 hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 20 20
#lev: 3 epoch: 10 train mean loss: 0.007939401788648982 error: 56.99658966064453
#normal-normal : 19873 normal-proud : 93 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9949 precision: 0.999547329242531 recall: 0.9953420815386157 f1 score: 0.9974402730375428
#reverse precision: 0.735294115484429 recall: 0.2118644066001149 f1 score: 0.3289473679882271

#False_3_16_True_0.1_0.0001, python repeat2.py 계속 좋아짐 hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 15 15
#lev: 3 epoch: 10 train mean loss: 0.007644414582337515 error: 54.528648376464844
#normal-normal : 19887 normal-proud : 79 proud:normal : 11 proud:proud : 23 nan cnt : 0
#accuracy: 0.9955 precision: 0.999447180621168 recall: 0.9960432735650606 f1 score: 0.9977423239012644
#reverse precision: 0.6764705862456747 recall: 0.22549019585736257 f1 score: 0.33823529362024224
#------------------------------------------------------------------------------------
#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 1
#threshold = 0.0001
#normal-normal : 19810 normal-proud : 156 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9918 precision: 0.9995963265718034 recall: 0.9921867174196134 f1 score: 0.9958777397948925
#reverse precision: 0.7647058801038062 recall: 0.14285714277864991 f1 score: 0.2407407139917724

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2
#threshold = 0.03
#normal-normal : 19699 normal-proud : 267 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.98625 precision: 0.9995940528746131 recall: 0.9866272663527997 f1 score: 0.993068333627404
#reverse precision: 0.7647058801038062 recall: 0.08873720133490197 f1 score: 0.15902138799764545

#False_3_16_1_False_0.3_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2
#threshold = 0.001, 쓰레기
#normal-normal : 19939 normal-proud : 27 proud:normal : 25 proud:proud : 9 nan cnt : 0
#accuracy: 0.9974 precision: 0.9987477459426969 recall: 0.9986477010918562 f1 score: 0.9986977210117707
#reverse precision: 0.26470588157439445 recall: 0.24999999930555555 f1 score: 0.2571428064489893
#False_3_16_1_False_0.3_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 1
#쓰레기

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 20 20
#threshold = 0.06, 디코드 튜닝 4번
#lev: 3 epoch: 10 train mean loss: 0.012169648418561198 error: 66.65066528320312
#normal-normal : 19760 normal-proud : 206 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.98935 precision: 0.9996458744371933 recall: 0.9896824601823099 f1 score: 0.9946392167719528
#reverse precision: 0.7941176447231834 recall: 0.11587982827644643 f1 score: 0.20224716863471465

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 100 100
#threshold = 0.06, 디코드 튜닝 3번
#lev: 3 epoch: 10 train mean loss: 0.06827820976002094 error: 129.69760131835938
#normal-normal : 18316 normal-proud : 1650 proud:normal : 4 proud:proud : 30 nan cnt : 0
#accuracy: 0.9173 precision: 0.9997816593886463 recall: 0.9173595111689873 f1 score: 0.9567988298594786
#reverse precision: 0.8823529385813148 recall: 0.017857142856079932 f1 score: 0.03500583041300391

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 3 20 20
#normal-normal : 19286 normal-proud : 680 proud:normal : 5 proud:proud : 29 nan cnt : 0
#accuracy: 0.96575 precision: 0.9997408117775128 recall: 0.9659421015726736 f1 score: 0.9825508826451335
#reverse precision: 0.8529411739619377 recall: 0.040902679824978465 f1 score: 0.07806190241663427

#False_3_16_1_True_0.1_0.0001, python repeat3.py hanormal_to.py 6 2 4647 anormal 0 10 0 4 10 1 20 20
#lev: 3 epoch: 10 train mean loss: 0.0006095053742263609 error: 10.407901763916016
#normal-normal : 19805 normal-proud : 161 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.99155 precision: 0.9995962247009539 recall: 0.991936291695883 f1 score: 0.9957515271877122
#reverse precision: 0.7647058801038062 recall: 0.13903743308072866 f1 score: 0.23529409139862287
#---------------------------------------------------------------------------
#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 1
#디코더 3 튜닝, 테스트 디코더 2, threshold = 0.00008
#normal-normal : 19813 normal-proud : 153 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.99195 precision: 0.9995963876646001 recall: 0.9923369728488813 f1 score: 0.9959534521275997
#reverse precision: 0.7647058801038062 recall: 0.14525139656689867 f1 score: 0.24413142834094056

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 3
#normal-normal : 19897 normal-proud : 69 proud:normal : 22 proud:proud : 12 nan cnt : 0
#accuracy: 0.99545 precision: 0.9988955268788648 recall: 0.9965441250075301 f1 score: 0.997718440511482
#reverse precision: 0.35294117543252596 recall: 0.14814814796524922 f1 score: 0.20869561016257923

#==============================================================================
#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 1 10 10 10
#python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 1
#lev: 3 epoch: 10 train mean loss: 7.25203436815765e-05 error: 5.675487995147705
#normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
#reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2
#lev: 3 epoch: 10 train mean loss: 0.008110312536024513 error: 61.7226448059082
#normal-normal : 19821 normal-proud : 145 proud:normal : 9 proud:proud : 25 nan cnt : 0
#accuracy: 0.9923 precision: 0.999546142203734 recall: 0.9927376540068479 f1 score: 0.9961302143437558
#reverse precision: 0.735294115484429 recall: 0.14705882344290658 f1 score: 0.24509801119761943

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 3
#lev: 3 epoch: 10 train mean loss: 1.0632710716663263 error: 763.47216796875
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0
#accuracy: 0.9914 precision: 0.9990922844125109 recall: 0.9922868877041355 f1 score: 0.9956779075792592
#reverse precision: 0.47058823391003457 recall: 0.09411764700346022 f1 score: 0.15686271716647932

#False_3_16_1_False_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#lev: 3 epoch: 10 train mean loss: 0.00583739073585886 error: 56.42653274536133
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507

#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 2 10 10 10
#python repeat.py hanormal_to.py 5 0 0 anormal 0 10 0 4 20 2
#lev: 3 epoch: 10 train mean loss: 0.009236615603395667 error: 64.72352600097656
#THRESHOLD: 0.045915089547634125 DECODE MODE: 2
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507

#False_3_16_1_True_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 2 10 10 10
#lev: 3 epoch: 10 train mean loss: 0.007032112725890982 error: 46.227256774902344
#normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
#reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351

################최종 디코드 타겟 뉴로모픽 출력 시작
#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#lev: 3 epoch: 10 train mean loss: 0.008826038682272132 error: 57.12886428833008
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 1 10 10 10
#lev: 3 epoch: 10 train mean loss: 5.6226852906641004e-05 error: 6.088313102722168
#normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
#reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351

#False_3_16_1_False_0.1_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 3 10 10 10
#lev: 3 epoch: 10 train mean loss: 0.9997351335791441 error: 758.9069213867188
#normal-normal : 19812 normal-proud : 154 proud:normal : 18 proud:proud : 16 nan cnt : 0
#accuracy: 0.9914 precision: 0.9990922844125109 recall: 0.9922868877041355 f1 score: 0.9956779075792592
#reverse precision: 0.47058823391003457 recall: 0.09411764700346022 f1 score: 0.15686271716647932

#False_3_16_1_True_0.1_0.0001, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 2 10 10 10
#python repeat.py hanormal_to.py 5 0 0 anormal 0 10 0 4 20 2
#lev: 3 epoch: 10 train mean loss: 0.00577438450179612 error: 44.377323150634766
#THRESHOLD: 0.04171229526400566 DECODE MODE: 2
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507

#False_3_16_1_False_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#python repeat.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 20 2
#lev: 3 epoch: 10 train mean loss: 0.006470513125606932 error: 91.46224975585938
#THRESHOLD: 0.04965265467762947 DECODE MODE: 2
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507

#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
#best
#lev: 3 epoch: 10 train mean loss: 0.013694376753380474 error: 59.7375602722168
#THRESHOLD: 0.049552883952856064 DECODE MODE: 2
#normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
#reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507
#####################최종 디코드 타겟 뉴로모픽 출력 끝
#------------------------------------------------------------------------------
#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 2 10 2 10 10 10
#python repeat.py hanormal_to.py 5 2 4647 anormal 0 10 0 2 10 2
#outer optim THRESHOLD: 0.049144834862852094 DECODE MODE: 2
#normal-normal : 19884 normal-proud : 82 proud:normal : 7 proud:proud : 27 nan cnt : 0
#accuracy: 0.99555 precision: 0.9996480820421313 recall: 0.9958930181258344 f1 score: 0.9977669670812555
#reverse precision: 0.7941176447231834 recall: 0.24770642179109503 f1 score: 0.37762234084796664

#False_1_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 2 10 2 10 10 10
#lev: 1 epoch: 10 train mean loss: 0.1322727958456828 error: 199.23263549804688
#not good, 로스 잘 줄지 않음
#THRESHOLD: 0.5372570157051086 DECODE MODE: 2
#normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
#reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351

#False_6_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 2 10 2 10 10 10
#bad, loss 극소화 과적합

#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 2 10 2 50 50 50
#python repeat.py hanormal_to.py 5 1 4645 anormal 0 10 0 2 20 2
#bad, 로스 잘 줄지 않음
#lev: 1 epoch: 20 train mean loss: 0.01549804700204195 error: 61.01620101928711
#THRESHOLD: 0.037828512489795685 DECODE MODE: 2
#normal-normal : 19813 normal-proud : 153 proud:normal : 17 proud:proud : 17 nan cnt : 0


#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 10 2 10 10 10
"""
lev: 3 epoch: 10 train mean loss: 0.02185361116575316 error: 77.38650512695312
THRESHOLD: 0.07331930845975876 DECODE MODE: 2
normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507
optim THRESHOLD: 0.0746931029724121 DECODE MODE: 2
normal-normal : 19824 normal-proud : 142 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99255 precision: 0.9996470172911116 recall: 0.9928879094410854 f1 score: 0.9962559491914904
reverse precision: 0.7941176447231834 recall: 0.15976331351493295 f1 score: 0.26600982406756096
outer THRESHOLD: 0.04117753356695175 DECODE MODE: 2
normal-normal : 19620 normal-proud : 346 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.9824 precision: 0.9996942830887613 recall: 0.9826705399129386 f1 score: 0.9911092650118172
reverse precision: 0.8235294093425606 recall: 0.07486631014041008 f1 score: 0.13725488661572638
outer optim THRESHOLD: 0.04737397341599464 DECODE MODE: 2
normal-normal : 19689 normal-proud : 277 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.98585 precision: 0.9996953541457237 recall: 0.9861264149004 f1 score: 0.9928644768625124
reverse precision: 0.8235294093425606 recall: 0.09180327865842515 f1 score: 0.16519172226834278
"""

#False_1_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 10 10 10
#lev: 3 epoch: 10 train mean loss: 0.046793374561298735 error: 91.83001708984375
#not good, 로스 잘 줄지 않음
#THRESHOLD: 0.1674216389656067 DECODE MODE: 2
#normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
#accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
#reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351

#False_6_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 10 10 10
#bad, loss 극소화 과적합

#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 50 50 50
#python repeat.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 20 2
"""
lev: 3 epoch: 10 train mean loss: 0.008921594648526456 error: 52.854339599609375
THRESHOLD: 0.035445742309093475 DECODE MODE: 2
normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507
optim THRESHOLD: 0.044658043073320386 DECODE MODE: 2
normal-normal : 19874 normal-proud : 92 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99505 precision: 0.9996479050299298 recall: 0.9953921666783763 f1 score: 0.9975154467703883
reverse precision: 0.7941176447231834 recall: 0.22689075611185652 f1 score: 0.3529411414413293
outer THRESHOLD: 0.02131666988134384 DECODE MODE: 2
normal-normal : 19596 normal-proud : 370 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.98115 precision: 0.9996429117941149 recall: 0.981468496439039 f1 score: 0.9904722894568605
reverse precision: 0.7941176447231834 recall: 0.06801007554961963 f1 score: 0.1252900086110664
outer optim THRESHOLD: 0.044658043073320386 DECODE MODE: 2
normal-normal : 19874 normal-proud : 92 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99505 precision: 0.9996479050299298 recall: 0.9953921666783763 f1 score: 0.9975154467703883
reverse precision: 0.7941176447231834 recall: 0.22689075611185652 f1 score: 0.3529411414413293
"""

#False_3_16_1_True_0.4_0.0001, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 100 10 10
#python repeat.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 20 2, not good
#lev: 3 epoch: 10 train mean loss: 0.005935224557432752 error: 41.48263168334961
#THRESHOLD: 0.024309348315000534 DECODE MODE: 2
#normal-normal : 19819 normal-proud : 147 proud:normal : 11 proud:proud : 23 nan cnt : 0

#=======================================================================
#False_3_16_1_True_0.1_0.0001_1_False, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 20 2 1 10 10
"""
lev: 3 epoch: 10 train mean loss: 0.0062041533737777705 error: 47.675865173339844
20 REPEAT: 18 THRESHOLD: 0.046214453876018524 DECODE MODE: 2
normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507
optim THRESHOLD: 0.060261440114879605 DECODE MODE: 2
normal-normal : 19874 normal-proud : 92 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99505 precision: 0.9996479050299298 recall: 0.9953921666783763 f1 score: 0.9975154467703883
reverse precision: 0.7941176447231834 recall: 0.22689075611185652 f1 score: 0.3529411414413293
outer THRESHOLD: 0.023236677050590515 DECODE MODE: 2
normal-normal : 19551 normal-proud : 415 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.9789 precision: 0.9996420901881601 recall: 0.9792146649254772 f1 score: 0.9893228930222855
reverse precision: 0.7941176447231834 recall: 0.06108597283685838 f1 score: 0.11344536483828979
outer optim THRESHOLD: 0.060261440114879605 DECODE MODE: 2
normal-normal : 19874 normal-proud : 92 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99505 precision: 0.9996479050299298 recall: 0.9953921666783763 f1 score: 0.9975154467703883
reverse precision: 0.7941176447231834 recall: 0.22689075611185652 f1 score: 0.3529411414413293
"""

#False_3_16_1_True_0.4_0.0001_1_False, python repeat3.py hanormal_to.py 5 2 4647 anormal 0 10 0 4 10 2 1 10 10 10
"""
lev: 3 epoch: 10 train mean loss: 0.013524994854098903 error: 57.29716110229492
10 REPEAT: 2 THRESHOLD: 0.05222705751657486 DECODE MODE: 2
normal-normal : 19823 normal-proud : 143 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.9925 precision: 0.9996469994906725 recall: 0.9928378242963396 f1 score: 0.9962307269580885
reverse precision: 0.7941176447231834 recall: 0.1588235293183391 f1 score: 0.2647058543156507
optim THRESHOLD: 0.057227921323680875 DECODE MODE: 2
normal-normal : 19840 normal-proud : 126 proud:normal : 7 proud:proud : 27 nan cnt : 0
accuracy: 0.99335 precision: 0.9996473018541863 recall: 0.9936892717570185 f1 score: 0.9966593326091616
reverse precision: 0.7941176447231834 recall: 0.17647058811995386 f1 score: 0.2887700234150278
outer THRESHOLD: 0.025863083079457283 DECODE MODE: 2
normal-normal : 19552 normal-proud : 414 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.979 precision: 0.9996932201605496 recall: 0.9792647500702231 f1 score: 0.9893734951880582
reverse precision: 0.8235294093425606 recall: 0.06334841627526054 f1 score: 0.11764704550879324
outer optim THRESHOLD: 0.039080449031496045 DECODE MODE: 2
normal-normal : 19753 normal-proud : 213 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.98905 precision: 0.9996963409028812 recall: 0.9893318641641323 f1 score: 0.9944870488006327
reverse precision: 0.8235294093425606 recall: 0.11618257256589935 f1 score: 0.2036363418181841
"""
#False_3_16_1_True_0.4_0.0001_1_False, python repeat3.py hanormal_to.py 5 1 4645 anormal 0 10 0 4 20 2 1 10 10
"""
lev: 3 epoch: 10 train mean loss: 0.010614881739736749 error: 63.87694549560547
20 REPEAT: 7 THRESHOLD: 0.050876863300800323 DECODE MODE: 2
normal-normal : 19822 normal-proud : 144 proud:normal : 8 proud:proud : 26 nan cnt : 0
accuracy: 0.9924 precision: 0.9995965708472032 recall: 0.9927877391515937 f1 score: 0.9961804706509222
reverse precision: 0.7647058801038062 recall: 0.15294117638062285 f1 score: 0.2549019327566351
optim THRESHOLD: 0.05203468410840034 DECODE MODE: 2
normal-normal : 19828 normal-proud : 138 proud:normal : 8 proud:proud : 26 nan cnt : 0
accuracy: 0.9927 precision: 0.99959669287659 recall: 0.9930882500200686 f1 score: 0.9963317926165032
reverse precision: 0.7647058801038062 recall: 0.15853658526918502 f1 score: 0.2626262339149097
outer THRESHOLD: 0.02514352835714817 DECODE MODE: 2
normal-normal : 19552 normal-proud : 414 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.979 precision: 0.9996932201605496 recall: 0.9792647500702231 f1 score: 0.9893734951880582
reverse precision: 0.8235294093425606 recall: 0.06334841627526054 f1 score: 0.11764704550879324
outer optim THRESHOLD: 0.0367231970893383 DECODE MODE: 2
normal-normal : 19750 normal-proud : 216 proud:normal : 6 proud:proud : 28 nan cnt : 0
accuracy: 0.9889 precision: 0.9996962947914573 recall: 0.9891816087298949 f1 score: 0.9944111075413304
reverse precision: 0.8235294093425606 recall: 0.11475409831362537 f1 score: 0.2014388273070775
"""
#False_3_16_1_True_0.1_0.0001_1_False, python repeat3.py hanormal_to.py 5 0 0 anormal 0 10 0 4 10 2 1 50 50 50
#bad





























#뉴로모픽 추론
#far_tuning false, hanormal_to.py 2 0 0 anormal 1 200 3
#lev:  0 epoch:  200 i:  227840 train mean loss:  2.508992895450485
#accuracy:  0.375081866979599  threshold:  0.99
#normal-normal : 227451 normal-proud : 0 proud:normal : 50 proud:proud : 344 nan cnt : 0
#normal-normal : 56856 normal-proud : 8 proud:normal : 63 proud:proud : 35 nan cnt : 0

#epoch:  299 i:  227200 train loss:  2.369776964187622
#lev:  0 epoch:  300 i:  227840 train mean loss:  2.3208014276925097
#accuracy:  0.4119824469089508  threshold:  0.99
#normal-normal : 227451 normal-proud : 0 proud:normal : 64 proud:proud : 330 nan cnt : 0
#normal-normal : 56856 normal-proud : 8 proud:normal : 68 proud:proud : 30 nan cnt : 0

#far_tuning false, python hanormal_to.py 2 2 4647 anormal 1 100 3
#epoch:  99 i:  227200 train loss:  3.0323102474212646
#lev:  0 epoch:  100 i:  227840 train mean loss:  3.0347832628850187
#accuracy:  0.2949821352958679  threshold:  0.9
#normal-normal : 56836 normal-proud : 28 proud:normal : 59 proud:proud : 39 nan cnt : 0
#far_tuning false,
#epoch:  99 i:  227200 train loss:  3.088752031326294
#lev:  0 epoch:  100 i:  227840 train mean loss:  3.053796175959405
#accuracy:  0.291633665561676  threshold:  0.9
#normal-normal : 56852 normal-proud : 12 proud:normal : 60 proud:proud : 38 nan cnt : 0
#far_tuning true
#epoch:  99 i:  227200 train loss:  2.8543286323547363
#lev:  0 epoch:  100 i:  227840 train mean loss:  2.904023071755184
#accuracy:  0.31455302238464355  threshold:  0.9

#python hanormal_to.py 2 1 4645 anormal 2 500 3, python hanormal_to.py 9 1 4645 anormal 2 0, python repeat.py hanormal_to.py 2 1 4645 anormal 0 100 0 2 20
#False_6_16_True_0.4_0.0001, 2레벨 500 에포크 학습, 100 에포크 튜닝 학습
#lev: 1 epoch: 100 train mean loss: 0.7106785688262719 accuracy: 0.9184749722480774 threshold: 0.99
#normal-normal : 19958 normal-proud : 8 proud:normal : 13 proud:proud : 21 nan cnt : 7
#accuracy: 0.99895 precision: 0.9993490561313905 recall: 0.9995993188420315 f1 score: 0.9994741718206174
#reverse precision: 0.6176470588235294 recall: 0.7241379310344828 f1 score: 0.6666666666666667

#python hanormal_to.py 2 1 4645 anormal 2 500 3, python hanormal_to.py 9 1 4645 anormal 2 0, python repeat.py hanormal_to.py 2 1 4645 anormal 0 100 0 2 20
#False_6_32_True_0.4_0.0001, 2레벨 500 에포크 학습, 600 에포크 튜닝 학습
#lev: 1 epoch: 100 train mean loss: 0.3953837595688991 accuracy: 0.9414874911308289 threshold: 0.99
#normal-normal : 19962 normal-proud : 4 proud:normal : 11 proud:proud : 23 nan cnt : 2
#accuracy: 0.99925 precision: 0.9994492564962699 recall: 0.9997996594210157 f1 score: 0.9996244272515585
#reverse precision: 0.6764705882352942 recall: 0.8518518518518519 f1 score: 0.7540983606557378


#normal-normal : 56841 normal-proud : 23 proud:normal : 75 proud:proud : 23 nan cnt : 0
#lev: 1 epoch: 2 train mean loss: 11.136497557029296 accuracy: 0.06408633291721344
#normal-normal : 56830 normal-proud : 34 proud:normal : 71 proud:proud : 27 nan cnt : 0
#lev: 1 epoch: 3 train mean loss: 9.959327614977118 accuracy: 0.08945774286985397
#normal-normal : 56861 normal-proud : 3 proud:normal : 59 proud:proud : 39 nan cnt : 1


#20000개 데이터 4번 실행
#normal-normal : 19974 normal-proud : 0 proud:normal : 8 proud:proud : 18 nan cnt : 0

"""
python hanormal_to.py 2 0 0 anormal 2 5 3
DEVICE:  cuda:0
INPUT SIZE:  31
STRIDE:  2
BKERRENL:  5
KERNEL:  5
HIDDEN:  16
LAYER:  6
BATCH SIZE:  128
MEAN SQUARE:  0
LEARNING RATE:  0.0001
DROPOUT RATE:  0.1
LAYER NORM:  False
NUM EPAIR:  2
RESIDUAL:  True
SCHEDULER:  False
DATA TYPE:  torch.float32
TUNING:  False
FAR TUNING:  False
THRESHOLD:  0.99
LEVEL: 0, YDISC: 20596
lev: 0 epoch: 5 train mean loss: 9.859227989030922 accuracy: 0.0036888890899717808
LEVEL: 1, YDISC: 62
lev: 1 epoch: 5 train mean loss: 0.20695319253465402 accuracy: 0.9943749904632568
normal-normal : 2996 normal-proud : 2 proud:normal : 0 proud:proud : 2 nan cnt : 0
"""