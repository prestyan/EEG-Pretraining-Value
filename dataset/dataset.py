import torch
import os
import sys
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import copy
import gc

import scipy
from dataset.sea import StableEA, compute_R_ref, EA

from torch.utils.data import Dataset,DataLoader

from scipy import stats

# from Modules.spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
# from Data_process.process_function import Load_BCIC_2a_raw_data
from collections import Counter
current_path = os.path.abspath('./')
root_path = current_path # os.path.split(current_path)[0]

sys.path.append(root_path)

# data_path = os.path.join(root_path,'Data','BCIC_2a_0_38HZ')
# if not os.path.exists(data_path):
#     print('BCIC_2a_0_38HZ数据不存在，开始初始化！')
#     Load_BCIC_2a_raw_data(0,4,[0,38])
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def select_devices(num_device,gpus=None):
    if gpus is None:
        gpus = torch.cuda.device_count()  
        gpus = [i for i in range(gpus)]
        
    res = []
    last_id = 0
            
    min_memory = 25447170048 // 2  
    for i in range(num_device):
        device_id = gpus[last_id%len(gpus)]
        last_id+=1
        while torch.cuda.get_device_properties(device_id).total_memory < min_memory:
            device_id = gpus[last_id%len(gpus)]
            last_id+=1
        res.append(torch.device(f'cuda:{device_id}') )
    return res

def select_free_gpu():  
    """选择空闲显存最大的GPU"""  
    gpus = torch.cuda.device_count()  
    if gpus == 0:  
        return None  
    else:  
        device_id = 0  
        min_memory = 25447170048 // 2  
        while True:
            i = random.randint(0, gpus-1)
        # for i in range(gpus):  
            mem_info = torch.cuda.get_device_properties(i)  
            # print(mem_info.total_memory)
            if mem_info.total_memory > min_memory:  
                device_id = i  
                break

        return torch.device(f'cuda:{device_id}') 

def rand_mask(feature):

    for _ in range(np.random.randint(0,4)):
        c = np.random.randint(0,22)

        a = np.random.normal(1,0.4,1)[0]

        feature[:,c] *=a
    return feature

def rand_cov(x):
    # print('xt shape:',xt.shape)
    E = torch.matmul(x, x.transpose(1,2))
    # print(E.shape)
    R = E.mean(0)
    
    U, S, V = torch.svd(R)
    R_mat = U@torch.diag(torch.rand(S.shape[0])*2)@V
    new_x = torch.einsum('n c s,r c -> n r s',x,R_mat)
    return new_x


def shuffle_data(dataset):
    x = rearrange(dataset.x,'(n i) c s->n i c s',n=16)
    y = rearrange(dataset.y,'(n i)->n i',n=16)
    new_x = []
    new_y = []

    for i in np.random.permutation(x.shape[0]):
        index = np.random.permutation(x.shape[1])
        new_x.append(x[i][index])
        new_y.append(y[i][index])

    new_x = torch.stack(new_x)
    new_y = torch.stack(new_y)
    new_x = rearrange(new_x,'a b c d->(a b) c d')
    new_y = rearrange(new_y,'a b->(a b)')

    return eeg_dataset(new_x,new_y)


def print_log(s,path="log.txt"):
    with open(path,"a+") as f:
        f.write((str(s) if type(s) is not str else s) +"\n")
def callback(res):
        print('<进程%s> subject %s accu %s' %(os.getpid(),res['sub'], str(res["accu"])))
        
        
def geban(batch_size=10, n_class=4):
    res = [random.randint(0, batch_size) for i in range(n_class-1) ]
    res.sort()
    # print(res)
    ret=[]
    last=0
    for r in res:
        ret.append(r-last)
        last=r
    ret.append(batch_size-last)
    return ret

def geban_entropy(batch_size=10, n_class=4, entropy_scope=[0,1]):
    while True:
        num_class = geban(batch_size, n_class)
        total = sum(num_class)
        ent = stats.entropy([x/total for x in num_class], base=n_class)
        if entropy_scope[0]<=ent and ent<=entropy_scope[1]: break
    return num_class

def sample(batch_size=10, n_class=4):
    res = [random.randint(0, n_class-1) for i in range(batch_size) ]
    res = Counter(res)
    ret = []
    for i in range(n_class):
        ret.append(res[i])
    return ret

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

# 构建用于读取验证集和测试集数据的Dataset类
class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        

    
def get_subj_data(sub, data_path, few_shot_number = 1, is_few_EA = False, target_sample=-1, sess=None, use_average=False):
    
    # target_y_data = []
    
    i=sub
    R=None
    source_train_x = []
    source_train_y = []
    source_valid_x = []
    source_valid_y = []
    
    if sess is not None:
        
        train_path = os.path.join(data_path,r'sub{}_sess{}_train/Data.mat'.format(i, sess))
        test_path = os.path.join(data_path,r'sub{}_sess{}_test/Data.mat'.format(i, sess))
    else:
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        
    train_data = sio.loadmat(train_path)
    test_data = sio.loadmat(test_path)
    if use_average:
        train_data['x_data'] = train_data['x_data'] - train_data['x_data'].mean(-2, keepdims=True)
    if is_few_EA is True:
        session_1_x = EA(train_data['x_data'],R)
    else:
        session_1_x = train_data['x_data']

    session_1_y = train_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    if use_average:
        test_data['x_data'] = test_data['x_data'] - test_data['x_data'].mean(-2, keepdims=True)
        
    if is_few_EA is True:
        session_2_x = EA(test_data['x_data'],R)
    else:
        session_2_x = test_data['x_data']

    session_2_y = test_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    valid_datset  = eeg_dataset(source_valid_x,source_valid_y)
    
    return train_dataset, valid_datset

import glob
def get_seed4_data(target_sub, data_path, valid_ratio=0., is_few_EA=True, use_channels=None):
    """
    LOSO-style split:
    - target_sub: test subject id (int)
    - data_path: directory containing 1_i.mat
    """
    original_channels = [
        # ===== Frontal pole / Anterior frontal =====
        'FP1', 'FPZ', 'FP2',
        'AF3',        'AF4',

        # ===== Frontal =====
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',

        # ===== Fronto-temporal / Fronto-central =====
        'FT7',
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
        'FT8',

        # ===== Temporal / Central =====
        'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'T8',

        # ===== Temporo-parietal / Centro-parietal =====
        'TP7',
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8',

        # ===== Parietal =====
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',

        # ===== Parieto-occipital =====
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',

        # ===== Occipital / Cerebellar =====
        'CB1',
        'O1', 'OZ', 'O2',
        'CB2',
    ]

    train_data_list = []
    train_label_list = []

    test_data = None
    test_label = None

    R = None

    # ---- 1. 扫描所有被试文件 ----
    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))

    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in {data_path}")

    for mat_path in mat_files:
        name = os.path.basename(mat_path)
        # 假设文件名是 1_i.mat
        sub_id = int(os.path.splitext(name)[0].split("_")[1])

        mat = sio.loadmat(mat_path)
        data = mat["data"]    # (N, 62, 1024)
        if is_few_EA:
            data = EA(data, R)
        label = mat["label"].squeeze()  # (N,)

        if sub_id == target_sub:
            test_data = data
            if is_few_EA:
                test_data = EA(test_data, R)
            test_label = label
        else:
            train_data_list.append(data)
            train_label_list.append(label)

    if test_data is None:
        raise ValueError(f"Target subject {target_sub} not found")

    # ---- 2. 合并训练被试 ----
    train_data_all = np.concatenate(train_data_list, axis=0)
    train_label_all = np.concatenate(train_label_list, axis=0)

    # 在这里补充，如果use_channels不为None，则从original_channels选择在use_channels中的通道
    # 并修改train_data_all, test_data
    if use_channels is not None:
        channel_indices = [original_channels.index(ch) for ch in use_channels if ch in original_channels]
        train_data_all = train_data_all[:, channel_indices, :]
        test_data = test_data[:, channel_indices, :]

    # ---- 3. 从训练集中划分验证集 ----
    if valid_ratio > 0.0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_data_all,
            train_label_all,
            test_size=valid_ratio,
            stratify=train_label_all
        )
    else:
        X_train = train_data_all
        X_valid = np.array([])
        y_train = train_label_all
        y_valid = np.array([])

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train).view(-1)

    test_data  = torch.FloatTensor(test_data)
    test_label  = torch.LongTensor(test_label).view(-1)

    if X_valid is not None:
        X_valid = torch.FloatTensor(X_valid)
        y_valid = torch.LongTensor(y_valid).view(-1)

    # ---- 4. 构建 Dataset ----
    train_dataset = eeg_dataset(X_train, y_train)
    print(X_train.shape)
    if len(X_valid) > 0:
        valid_dataset = eeg_dataset(X_valid, y_valid)
    else :
        valid_dataset = None
    test_dataset  = eeg_dataset(test_data, test_label)
    print(test_data.shape)

    return train_dataset, valid_dataset, test_dataset

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    # 1) 先算全局参考（只用 source train）
    # R_ref = None
    # if is_few_EA:
    #     R_ref = compute_R_ref(data_path, target_sub=sub)
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        print(source_train_x.shape)
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        print(source_train_x.shape)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_default_10_20_32ch_names():
    ch32 = [
        "Fp1", "Fp2",
        "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "Oz", "O2",
        "A1", "A2",   # 常见的参考（或耳垂/乳突）
        "AFz"         # 有些帽子会有 AFz/POz 等（这里占位为 32）
    ]
    return ch32

def get_default_30ch_names():
    ch32 = get_default_10_20_32ch_names()
    DROP_CHS = {"A1", "A2"}
    ch30 = [c for c in ch32 if c not in DROP_CHS]

    if len(ch30) != 30:
        ch30 = ch32[:30]
    return ch30

def _normalize_name(name: str) -> str:
    name = name.strip()
    alias = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "Pz": "PZ",
        "Fz": "FZ",
        "Cz": "CZ",
        "Oz": "OZ",
    }
    std = name[0].upper() + name[1:]
    std = std.replace("z", "Z")
    return alias.get(std, std)

def print_label_stats(y, name):
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print(f"\n{name} label distribution:")
    for l, c in zip(labels, counts):
        print(f"  label {l}: {c} ({c/total:.2%})")
    print(f"  total: {total}")

def get_wmsub_data(data_path, sub, use_channels=None, seed=2024, split=(0.6, 0.2, 0.2)):
    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]
    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30
    fp = os.path.join(data_path, f"Sub_S1_{sub}.mat")
    mat = sio.loadmat(fp)
    data = mat["data1"]
    label = mat["label1"]
    label = np.asarray(label).reshape(-1)
    label_map = {11: 0, 14: 1}
    label = np.vectorize(label_map.get)(label)
    if pick_idx is not None:
        data = data[pick_idx, :, :]  # [C_sel, T, N]
    data = np.transpose(data, (2, 0, 1))  # [N, C, T]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(label))
    rng.shuffle(idx)
    n_total = len(idx)
    n_train = int(round(split[0] * n_total))
    n_valid = int(round(split[1] * n_total))

    n_test = n_total - n_train - n_valid

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    X_train, y_train = data[idx_train], label[idx_train]
    X_valid, y_valid = data[idx_valid], label[idx_valid]
    X_test, y_test = data[idx_test], label[idx_test]

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wmsub_data_ind(data_path, source, target, use_channels=None, seed=2024, split=(0.9, 0.1)):
    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _load_one_subject(sub_id):
        fp = os.path.join(data_path, f"Sub_S1_{sub_id}.mat")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"文件不存在: {fp}")

        mat = sio.loadmat(fp)
        data = mat["data1"]   # [C, T, N]
        label = mat["label1"] # [N, 1] or similar

        label = np.asarray(label).reshape(-1)
        label_map = {11: 0, 14: 1}
        label = np.vectorize(label_map.get)(label)

        if np.any(label == None):
            raise ValueError(f"被试 {sub_id} 的标签中存在未映射的类别，请检查 label_map。")

        label = label.astype(np.int64)

        if pick_idx is not None:
            data = data[pick_idx, :, :]  # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1))  # [N, C, T]
        return data.astype(np.float32), label

    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )

        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    if not isinstance(split, (list, tuple)) or len(split) != 2:
        raise ValueError("split 必须是长度为 2 的 tuple/list，例如 (0.9, 0.1)")
    train_ratio, valid_ratio = split
    if train_ratio <= 0 or valid_ratio <= 0:
        raise ValueError("split 中的比例必须都大于 0")
    if not np.isclose(train_ratio + valid_ratio, 1.0):
        raise ValueError(f"split 之和必须为 1，当前为 {train_ratio + valid_ratio}")

    source_list = _as_list(source)
    X_source_list, y_source_list = [], []

    for s in source_list:
        x_s, y_s = _load_one_subject(s)
        X_source_list.append(x_s)
        y_source_list.append(y_s)

    X_source = np.concatenate(X_source_list, axis=0)  # [N_all, C, T]
    y_source = np.concatenate(y_source_list, axis=0)  # [N_all]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_source,
        y_source,
        test_size=valid_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y_source
    )

    target_list = _as_list(target)
    X_target_list, y_target_list = [], []

    for t in target_list:
        x_t, y_t = _load_one_subject(t)
        X_target_list.append(x_t)
        y_target_list.append(y_t)

    X_test = np.concatenate(X_target_list, axis=0)
    y_test = np.concatenate(y_target_list, axis=0)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wm_data(data_path, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    assert abs(sum(split) - 1.0) < 1e-6

    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"在 {data_path} 下没有找到任何 .mat 文件")

    # 默认 30 通道名（你可按真实设备改）
    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    # 如果要筛通道：构建索引
    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    X_list, y_list = [], []

    for fp in mat_files:
        mat = sio.loadmat(fp)
        if "data1" not in mat or "label1" not in mat:
            raise KeyError(f"{os.path.basename(fp)} 缺少 data1 或 label1")

        data = mat["data1"]
        label = mat["label1"]

        # label: [N,1] 或 [N,]
        label = np.asarray(label).reshape(-1)
        label_map = {11: 0, 14: 1}
        label = np.vectorize(label_map.get)(label)

        if pick_idx is not None:
            data = data[pick_idx, :, :]  # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1))  # [N, C, T]

        X_list.append(data.astype(np.float32))
        y_list.append(label.astype(np.int64))

    X = np.concatenate(X_list, axis=0)  # [N_total, C, T]
    y = np.concatenate(y_list, axis=0)  # [N_total]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n_total = len(idx)
    n_train = int(round(split[0] * n_total))
    n_valid = int(round(split[1] * n_total))

    n_test = n_total - n_train - n_valid

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    X_test, y_test = X[idx_test], y[idx_test]
    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    info = {
        "total_samples": int(n_total),
        "train_samples": int(n_train),
        "valid_samples": int(n_valid),
        "test_samples": int(n_test),
        "channels": picked_names,
        "data_shape": {"X": list(X.shape), "y": list(y.shape)},
        "split": split,
        "seed": seed,
        "files": [os.path.basename(p) for p in mat_files],
    }
    print(info)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wm_data_per_subject_split(data_path, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    assert abs(sum(split) - 1.0) < 1e-6

    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"在 {data_path} 下没有找到任何 .mat 文件")

    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}
        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n默认通道列表为：{ch_names_30}")
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    label_map = {11: 0, 14: 1}

    for fp in mat_files:
        mat = sio.loadmat(fp)
        if "data1" not in mat or "label1" not in mat:
            raise KeyError(f"{os.path.basename(fp)} 缺少 data1 或 label1")

        data = mat["data1"]          # [30, 500, 240]
        label = mat["label1"]        # [240, 1] 或 [240]

        if data.ndim != 3:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 维度应为 3，实际为 {data.shape}")

        C, T, N = data.shape
        if C != 30:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 第一维应为 30 通道，实际为 {C}")

        label = np.asarray(label).reshape(-1).astype(int)

        label = np.vectorize(label_map.get)(label).astype(np.int64)

        if label.shape[0] != N:
            raise ValueError(f"{os.path.basename(fp)} label1 样本数 {label.shape[0]} 与 data1 样本数 {N} 不一致")

        if pick_idx is not None:
            data = data[pick_idx, :, :]   # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1)).astype(np.float32)

        sub_seed = (abs(hash(os.path.basename(fp))) + seed) % (2**32)
        rng = np.random.default_rng(sub_seed)

        idx = np.arange(N)
        rng.shuffle(idx)

        n_train = int(round(split[0] * N))
        n_valid = int(round(split[1] * N))
        n_test = N - n_train - n_valid

        idx_tr = idx[:n_train]
        idx_va = idx[n_train:n_train + n_valid]
        idx_te = idx[n_train + n_valid:]

        X_tr_list.append(data[idx_tr]); y_tr_list.append(label[idx_tr])
        X_va_list.append(data[idx_va]); y_va_list.append(label[idx_va])
        X_te_list.append(data[idx_te]); y_te_list.append(label[idx_te])

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    X_valid = np.concatenate(X_va_list, axis=0)
    y_valid = np.concatenate(y_va_list, axis=0)
    X_test  = np.concatenate(X_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    info = {
        "channels": picked_names,
        "split": split,
        "seed": seed,
        "files": [os.path.basename(p) for p in mat_files],
        "train_samples": int(len(y_train)),
        "valid_samples": int(len(y_valid)),
        "test_samples": int(len(y_test)),
    }
    print(info)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset  = eeg_dataset(X_test,  y_test)

    return train_dataset, valid_dataset, test_dataset

def get_data_openbmi(data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1):
    test_rate = 0.5
    subs_list = np.int32(np.linspace(1,54, 54))
    np.random.shuffle(subs_list)
    test_size = int(test_rate* len(subs_list))
    test_subs, train_subs = subs_list[:test_size],subs_list[test_size:]
    print(test_subs)
    source_test_x = []
    source_test_y = []
    for sub in test_subs:
        target_session_1_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(sub))
        target_session_2_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(sub))

        session_1_data = sio.loadmat(target_session_1_path)
        session_2_data = sio.loadmat(target_session_2_path)
        R = None
        if is_few_EA is True:
            session_1_x = EA(session_1_data['x_data'],R)
        else:
            session_1_x = session_1_data['x_data']
            
        if is_few_EA is True:
            session_2_x = EA(session_2_data['x_data'],R)
        else:
            session_2_x = session_2_data['x_data']
        
        # -- debug for BCIC 2b
        test_x_1 = torch.FloatTensor(session_1_x)      
        test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

        test_x_2 = torch.FloatTensor(session_2_x)      
        test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
        
        if target_sample>0:
            test_x_1 = temporal_interpolation(test_x_1, target_sample)
            test_x_2 = temporal_interpolation(test_x_2, target_sample)
        source_test_x.extend([test_x_1, test_x_2])
        source_test_y.extend([test_y_1, test_y_2])
            
    test_dataset = eeg_dataset(torch.cat(source_test_x,dim=0),torch.cat(source_test_y,dim=0))

    source_train_x = []
    source_train_y = []
    for i in train_subs:
        train_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)
        
        source_train_x.append(session_1_x)
        source_train_y.append(session_1_y)


        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)
        
        source_train_x.append(session_2_x)
        source_train_y.append(session_2_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))

    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    
    return train_dataset,test_dataset

def get_data_Nakanishi2015(sub,data_path="Data/Nakanishi2015_8_64HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,11)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_data_Wang2016(sub,data_path="Data/Wang2016_4_20HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,36)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset
if __name__=="__main__":
    train_dataset,valid_dataset,test_dataset = get_data_Wang2016(1,"Data/Wang2016_4_20HZ/", 1, is_few_EA = False, target_sample=-1)
    # # train_dataset,valid_dataset,test_dataset = get_data(1,data_path,1,True)
    # avg_ent = 0 

    # for i in range(1000):
    #     # print(geban()) 
    #     # print(sample())
    #     num_class = geban_entropy(entropy_scope=[1.2,1e6])#geban()#sample()
    #     total = sum(num_class)
    #     num_class = [x/total for x in num_class]
    #     # print(num_class)
    #     # print(sum([-x*(math.log(x)) for x in num_class if x>0]))
    #     ent = stats.entropy(num_class) 
    #     avg_ent+=ent
    #     print(avg_ent/1000) # sample 1.2110981470145854 geban 0.9734407215366253
    
    
import mne
import torch
import tqdm
import pandas as pd 
import csv
import numpy as np
import os
import scipy.io as scio

import random
mne.set_log_level("ERROR")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
    
    
use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

# -- read Kaggle ERN
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 200
    raw = pd.read_csv(filename)
    
    data = torch.tensor(raw.iloc[:,1:-2].values) # exclude time EOG Feedback
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    # print(stim_pos)
    datas = []
    
    # -- get channel id by use chan names
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        # print(start_i, end___i)
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        # print(trial.shape)
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",  # channel type
            sfreq=200,  # frequency
            #
        )
        raw = mne.io.RawArray(trial, info)  # create raw
        # raw = raw.filter(5,40)
        # raw = raw.resample(256)
        
        trial = torch.tensor(raw.get_data()).float()

        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas
    
def read_kaggle_ern_test(
                    path     = "datas/",
                    subjects = [1,3,4,5,8,9,10,15,19,25],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
    
    # -- read datas
    label_id = 0
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i,j))

            # -- read data
            for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names): 
                label = labels[label_id]
                label_id += 1
                datas.append((data, int(label)))
    return datas

def read_kaggle_ern_train(
                    path     = "datas/",
                    subjects = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26, ],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = []
    with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if(i>0): labels.append(row)
    labels = dict(labels) # [['S02_Sess01_FB001', '1'],
    
    # -- read datas
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            if i>9:
                if(i == 22 and j == 5):
                    print("Skipped error file " + "KagglERN/train/Data_S"+ str(i)+"_Sess0"+str(j)+".csv" )
                else:
                    filename = os.path.join(path, "KaggleERN","train","Data_S"+ str(i)+"_Sess0"+str(j)+".csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", "Data_S0"+ str(i)+"_Sess0"+str(j)+".csv")
            
            # -- read data
            for fb,trial in enumerate(read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names),1): 
                label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i,j,fb)]
                datas.append((trial, int(label)))
    return datas

def add_awgn(x, snr_db, seed=2025):
    if snr_db is None: return x
    generator = torch.Generator(device=x.device).manual_seed(seed)
    signal_power = torch.mean(x ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * torch.sqrt(noise_power + 1e-12)
    return x + noise


def get_data_noise(sub,data_path,few_shot_number=1,is_few_EA=True,target_sample=-1,use_avg=True,use_channels=None,snr_db=None,noise_seed=2025):
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))
    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None

    if is_few_EA is True: session_1_x = EA(session_1_data['x_data'],R)
    else: session_1_x = session_1_data['x_data']
    if is_few_EA is True: session_2_x = EA(session_2_data['x_data'],R)
    else: session_2_x = session_2_data['x_data']

    test_x_1 = torch.FloatTensor(session_1_x)
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    test_x_2 = torch.FloatTensor(session_2_x)
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)

    if target_sample > 0:
        test_x_1 = temporal_interpolation(test_x_1,target_sample,use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2,target_sample,use_avg=use_avg)

    test_x = torch.cat([test_x_1,test_x_2],dim=0)
    test_y = torch.cat([test_y_1,test_y_2],dim=0)
    if use_channels is not None: test_x = test_x[:,use_channels,:]

    # ==========================================================
    # Domain shift: ONLY add noise to the held-out test subject
    # ==========================================================
    test_x = add_awgn(test_x,snr_db=snr_db,seed=noise_seed)
    test_dataset = eeg_dataset(test_x,test_y)

    source_train_x,source_train_y,source_train_s = [],[],[]
    source_valid_x,source_valid_y,source_valid_s = [],[],[]
    subject_id = 0

    for i in range(1,10):
        if i == sub: continue

        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)

        if is_few_EA is True: session_1_x = EA(train_data['x_data'],R)
        else: session_1_x = train_data['x_data']
        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size=0.1,stratify=session_1_y)
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)
        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True: session_2_x = EA(test_data['x_data'],R)
        else: session_2_x = test_data['x_data']
        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size=0.1,stratify=session_2_y)
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)
        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id += 1

    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s,dim=0)
    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s,dim=0)

    if target_sample > 0:
        source_train_x = temporal_interpolation(source_train_x,target_sample,use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x,target_sample,use_avg=use_avg)

    if use_channels is not None: train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else: train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)

    if use_channels is not None: valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else: valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)

    return train_dataset,valid_datset,test_dataset

EEG_CHANNELS = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
    'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
    'O2', 'P4', 'P8', 'CP6', 'CP2', 'FCz', 'C4', 'T8',
    'FT8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
    'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7',
    'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4',
    'FT10', 'F6', 'AF8', 'AF4', 'F2'
]


def _resolve_channels(use_channels):
    """
    Convert channel names or indices into zero-based indices.

    Examples:
        None                         -> use all channels
        [0, 1, 2]                    -> use the first 3 channels
        ['Fp1', 'Fz', 'F3']          -> use the named channels
    """
    if use_channels is None:
        return None

    if isinstance(use_channels, (str, int, np.integer)):
        use_channels = [use_channels]
    else:
        use_channels = list(use_channels)

    if len(use_channels) == 0:
        raise ValueError("use_channels cannot be an empty list.")

    if all(isinstance(ch, str) for ch in use_channels):
        channel_map = {
            name.upper(): idx
            for idx, name in enumerate(EEG_CHANNELS)
        }

        indices = []
        for ch in use_channels:
            key = ch.strip().upper()
            if key not in channel_map:
                raise ValueError(
                    f"Unknown EEG channel: {ch}. "
                    f"Available channels: {EEG_CHANNELS}"
                )
            indices.append(channel_map[key])

    elif all(
        isinstance(ch, (int, np.integer))
        and not isinstance(ch, (bool, np.bool_))
        for ch in use_channels
    ):
        indices = [int(ch) for ch in use_channels]

        if any(ch < 0 or ch >= len(EEG_CHANNELS) for ch in indices):
            raise ValueError(
                f"Channel indices must be in [0, {len(EEG_CHANNELS) - 1}]."
            )

    else:
        raise TypeError(
            "use_channels must contain either all channel names "
            "or all zero-based integer indices."
        )

    return indices


def _load_subject_data(mat_path, channel_indices=None):
    """
    Load one subject's MATLAB file.

    Original:
        S1Data:  (61, 500, N)
        S1Label: (N, 1)

    Returned:
        x: (N, C, T), float32
        y: (N,), int64
    """
    mat = sio.loadmat(mat_path)

    if 'S1Data' not in mat or 'S1Label' not in mat:
        raise KeyError(
            f"{mat_path} must contain 'S1Data' and 'S1Label'."
        )

    x = np.asarray(mat['S1Data'], dtype=np.float32)
    y = np.asarray(mat['S1Label']).reshape(-1)

    if x.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data in {mat_path}, got shape {x.shape}."
        )

    # Convert (C, T, N) to (N, C, T).
    if x.shape[-1] == len(y):
        x = np.transpose(x, (2, 0, 1))
    elif x.shape[0] != len(y):
        raise ValueError(
            f"Data/label size mismatch in {mat_path}: "
            f"x.shape={x.shape}, len(y)={len(y)}."
        )

    if x.shape[0] != len(y):
        raise ValueError(
            f"Expected one label per trial in {mat_path}."
        )

    if x.shape[1] != len(EEG_CHANNELS):
        raise ValueError(
            f"Expected {len(EEG_CHANNELS)} channels, "
            f"got {x.shape[1]} in {mat_path}."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite labels found in {mat_path}.")

    if not np.all(y == np.floor(y)):
        raise ValueError(f"Labels must be integers in {mat_path}.")

    y = y.astype(np.int64)

    # Select channels, if requested.
    if channel_indices is not None:
        x = x[:, channel_indices, :]

    return np.ascontiguousarray(x), y


def _resize_time(x, target_length=None):
    """
    Resize temporal dimension of a tensor:
        (N, C, T) -> (N, C, target_length)
    """
    if target_length is None:
        return x

    target_length = int(target_length)

    if target_length <= 0:
        raise ValueError("target_length must be a positive integer.")

    if x.shape[-1] == target_length:
        return x

    return F.interpolate(
        x,
        size=target_length,
        mode='linear',
        align_corners=False
    )

import re
def get_data_matb(sub, data_path, use_channels=None, target_length=None):
    """
    Leave-one-subject-out data loading.

    Args:
        sub:
            Target subject ID, e.g. 10 corresponds to Sub_S1_10.mat.

        data_path:
            Directory containing Sub_S1_*.mat.

        use_channels:
            None, a list of zero-based channel indices, or a list
            of channel names.

        target_length:
            Desired temporal length. None preserves the original
            500 time points.

    Returns:
        train_dataset:
            Source-domain training dataset.

        valid_dataset:
            Source-domain validation dataset.

        test_dataset:
            All trials from the held-out target subject.
    """
    sub = int(sub)

    # --------------------------------------------------------
    # 1. Discover subject files automatically.
    # --------------------------------------------------------
    subject_files = {}

    for path in glob.glob(os.path.join(data_path, 'Sub_S1_*.mat')):
        filename = os.path.basename(path)
        match = re.fullmatch(r'Sub_S1_(\d+)\.mat', filename)

        if match:
            subject_id = int(match.group(1))
            subject_files[subject_id] = path

    if not subject_files:
        raise FileNotFoundError(
            f"No Sub_S1_*.mat files found in {data_path}."
        )

    if sub not in subject_files:
        raise FileNotFoundError(
            f"Target subject {sub} not found in {data_path}. "
            f"Available subjects: {sorted(subject_files.keys())}"
        )

    channel_indices = _resolve_channels(use_channels)

    print(f"Target subject: {sub}")
    print(f"Available subjects: {sorted(subject_files.keys())}")
    print(f"Selected channels: "
          f"{len(EEG_CHANNELS) if channel_indices is None else len(channel_indices)}")
    print(f"Target temporal length: {target_length or 'original (500)'}")

    # --------------------------------------------------------
    # 2. Load the entire target subject as the test set.
    # --------------------------------------------------------
    test_path = subject_files[sub]

    test_x_np, test_y_np = _load_subject_data(
        test_path, channel_indices
    )

    test_x = torch.from_numpy(test_x_np).float()
    test_y = torch.from_numpy(test_y_np).long()

    test_x = _resize_time(test_x, target_length)

    test_dataset = eeg_dataset(test_x, test_y)

    # --------------------------------------------------------
    # 3. Split every source subject into train / validation.
    #    Each source subject contributes 80% / 20%.
    # --------------------------------------------------------
    train_x_parts = []
    train_y_parts = []
    train_s_parts = []

    valid_x_parts = []
    valid_y_parts = []
    valid_s_parts = []

    # Source-domain IDs are contiguous: 0, 1, ..., n_source-1.
    source_domain_id = 0

    for sid in sorted(subject_files.keys()):

        # The target subject must not enter source train/valid.
        if sid == sub:
            continue

        subject_path = subject_files[sid]

        x, y = _load_subject_data(
            subject_path, channel_indices
        )

        try:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x,
                y,
                test_size=0.1,
                random_state=42 + sid,
                stratify=y
            )
        except ValueError as exc:
            raise ValueError(
                f"Failed to split subject {sid} with stratification. "
                f"Check whether every class has enough samples."
            ) from exc

        # Convert EEG arrays into PyTorch tensors.
        x_train = torch.from_numpy(
            np.ascontiguousarray(x_train)
        ).float()

        x_valid = torch.from_numpy(
            np.ascontiguousarray(x_valid)
        ).float()

        y_train = torch.from_numpy(y_train).long()
        y_valid = torch.from_numpy(y_valid).long()

        # Optional temporal interpolation.
        x_train = _resize_time(x_train, target_length)
        x_valid = _resize_time(x_valid, target_length)

        train_x_parts.append(x_train)
        train_y_parts.append(y_train)

        valid_x_parts.append(x_valid)
        valid_y_parts.append(y_valid)

        # Match the source subject-ID convention in your example.
        train_s_parts.append(
            torch.full(
                (len(y_train),),
                float(source_domain_id),
                dtype=torch.float32
            )
        )

        valid_s_parts.append(
            torch.full(
                (len(y_valid),),
                float(source_domain_id),
                dtype=torch.float32
            )
        )

        print(
            f"Subject {sid}: "
            f"train={len(y_train)}, valid={len(y_valid)}"
        )

        source_domain_id += 1

    if source_domain_id == 0:
        raise ValueError(
            "No source subjects available. At least two subjects "
            "are required for leave-one-subject-out evaluation."
        )

    # --------------------------------------------------------
    # 4. Concatenate all source subjects.
    # --------------------------------------------------------
    source_train_x = torch.cat(train_x_parts, dim=0)
    source_train_y = torch.cat(train_y_parts, dim=0)
    source_train_s = torch.cat(train_s_parts, dim=0)

    source_valid_x = torch.cat(valid_x_parts, dim=0)
    source_valid_y = torch.cat(valid_y_parts, dim=0)
    source_valid_s = torch.cat(valid_s_parts, dim=0)

    # --------------------------------------------------------
    # 5. Build datasets using your existing eeg_dataset class.
    # --------------------------------------------------------
    train_dataset = eeg_dataset(
        source_train_x,
        source_train_y,
        source_train_s
    )

    valid_dataset = eeg_dataset(
        source_valid_x,
        source_valid_y,
        source_valid_s
    )

    # --------------------------------------------------------
    # 6. Print summary and return.
    # --------------------------------------------------------
    print("\n========== Dataset Summary ==========")
    print(
        f"Train: X={tuple(source_train_x.shape)}, "
        f"y={tuple(source_train_y.shape)}, "
        f"subject={tuple(source_train_s.shape)}"
    )
    print(
        f"Valid: X={tuple(source_valid_x.shape)}, "
        f"y={tuple(source_valid_y.shape)}, "
        f"subject={tuple(source_valid_s.shape)}"
    )
    print(
        f"Test:  X={tuple(test_x.shape)}, "
        f"y={tuple(test_y.shape)}"
    )
    print("=====================================")

    return train_dataset, valid_dataset, test_dataset


def get_data_ratio(sub,data_path,few_shot_number=1,is_few_EA=False,target_sample=-1,use_avg=True,use_channels=None,train_ratio=1.0,split_seed=2025,sample_seed=2025,):

    train_ratio = float(train_ratio)
    if not np.isfinite(train_ratio) or not 0 < train_ratio <= 1:
        raise ValueError("train_ratio 必须满足 0 < train_ratio <= 1。")

    for name, seed in (
        ("split_seed", split_seed),
        ("sample_seed", sample_seed),
    ):
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"{name} 必须是非负整数。")

    R = None

    def load_session(subject, session):
        file_path = os.path.join(
            data_path,
            f"sub{subject}_{session}",
            "Data.mat",
        )
        data = sio.loadmat(file_path)

        x = data["x_data"]
        y = data["y_data"].reshape(-1)

        if is_few_EA:
            x = EA(x, R)

        return x, y

    def make_seed(base_seed, source_subject, session_index):

        seed_sequence = np.random.SeedSequence(
            [int(base_seed), int(sub), source_subject, session_index]
        )
        return int(seed_sequence.generate_state(1)[0])

    def select_train_indices(y, seed):

        if train_ratio == 1.0:
            return np.arange(len(y))

        rng = np.random.default_rng(seed)
        selected = []

        for label in np.unique(y):
            class_indices = np.flatnonzero(y == label)

            class_indices = rng.permutation(class_indices)

            number = max(
                1,
                int(np.floor(len(class_indices) * train_ratio)),
            )
            selected.append(class_indices[:number])


        return np.sort(np.concatenate(selected))

    def process_x(x):
        x = torch.as_tensor(
            np.asarray(x),
            dtype=torch.float32,
        )

        if target_sample > 0:
            x = temporal_interpolation(
                x,
                target_sample,
                use_avg=use_avg,
            )

        if use_channels is not None:
            x = x[:, use_channels, :]

        return x

    target_x_1, target_y_1 = load_session(sub, "train")
    target_x_2, target_y_2 = load_session(sub, "test")

    test_x = torch.cat(
        [
            process_x(target_x_1),
            process_x(target_x_2),
        ],
        dim=0,
    )
    test_y = torch.as_tensor(
        np.concatenate([target_y_1, target_y_2]),
        dtype=torch.long,
    )

    test_dataset = eeg_dataset(test_x, test_y)

    source_train_x = []
    source_train_y = []
    source_train_s = []

    source_valid_x = []
    source_valid_y = []
    source_valid_s = []

    original_train_number = 0
    subject_id = 0

    for source_subject in range(1, 10):
        if source_subject == sub:
            continue

        for session_index, session in enumerate(("train", "test")):
            session_x, session_y = load_session(
                source_subject,
                session,
            )

            train_x, valid_x, train_y, valid_y = train_test_split(
                session_x,
                session_y,
                test_size=0.1,
                stratify=session_y,
                random_state=make_seed(
                    split_seed,
                    source_subject,
                    session_index,
                ),
            )

            original_train_number += len(train_y)

            selected_indices = select_train_indices(
                train_y,
                seed=make_seed(
                    sample_seed,
                    source_subject,
                    session_index,
                ),
            )

            train_x = train_x[selected_indices]
            train_y = train_y[selected_indices]

            source_train_x.append(train_x)
            source_train_y.append(train_y)
            source_train_s.append(
                torch.full(
                    (len(train_y),),
                    float(subject_id),
                    dtype=torch.float32,
                )
            )

            source_valid_x.append(valid_x)
            source_valid_y.append(valid_y)
            source_valid_s.append(
                torch.full(
                    (len(valid_y),),
                    float(subject_id),
                    dtype=torch.float32,
                )
            )

        subject_id += 1

    source_train_x = process_x(
        np.concatenate(source_train_x, axis=0)
    )
    source_train_y = torch.as_tensor(
        np.concatenate(source_train_y),
        dtype=torch.long,
    )
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = process_x(
        np.concatenate(source_valid_x, axis=0)
    )
    source_valid_y = torch.as_tensor(
        np.concatenate(source_valid_y),
        dtype=torch.long,
    )
    source_valid_s = torch.cat(source_valid_s, dim=0)

    train_dataset = eeg_dataset(
        source_train_x,
        source_train_y,
        source_train_s,
    )
    valid_dataset = eeg_dataset(
        source_valid_x,
        source_valid_y,
        source_valid_s,
    )

    actual_ratio = len(source_train_y) / original_train_number
    print(
        f"[Subject {sub}] "
        f"Requested ratio={train_ratio:.1%}, "
        f"Actual ratio={actual_ratio:.2%}, "
        f"Train={len(source_train_y)}/{original_train_number}, "
        f"Valid={len(source_valid_y)}, "
        f"Test={len(test_y)}"
    )

    return train_dataset, valid_dataset, test_dataset