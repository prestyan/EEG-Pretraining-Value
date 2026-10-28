import os
import scipy
import scipy.io as sio
import numpy as np
# 欧氏空间的对齐方式 其中x：NxCxS
def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x

def StableEA(x, new_R=None, eps=1e-6, normalize='time'):
    """
    x: (N, C, S)
    new_R: (C, C) global reference covariance (R_ref). If None -> whiten to I.
    """
    # (N, S, C)
    xt = np.transpose(x, axes=(0, 2, 1))

    # (N, C, C) = X X^T
    E = np.matmul(x, xt)

    if normalize == 'time':
        # divide by time length S
        E = E / x.shape[-1]
    elif normalize == 'trace':
        tr = np.trace(E, axis1=1, axis2=2)  # (N,)
        E = E / (tr[:, None, None] + eps)

    R = np.mean(E, axis=0)  # (C, C)
    R = (R + R.T) / 2.0
    R = R + eps * np.eye(R.shape[0], dtype=R.dtype)

    R_inv_sqrt = scipy.linalg.fractional_matrix_power(R, -0.5)
    # whiten: (C,C) @ (N,C,S) on channel dim
    xw = np.einsum('rc,ncs->nrs', R_inv_sqrt, x)

    if new_R is None:
        return xw

    new_R = (new_R + new_R.T) / 2.0
    new_R = new_R + eps * np.eye(new_R.shape[0], dtype=new_R.dtype)
    R_ref_sqrt = scipy.linalg.fractional_matrix_power(new_R, 0.5)

    # color to R_ref
    xa = np.einsum('rc,ncs->nrs', R_ref_sqrt, xw)
    return xa


def compute_R_ref(data_path, target_sub, eps=1e-6):
    Rs = []
    for i in range(1, 10):
        if i == target_sub:
            continue

        train_path = os.path.join(data_path, f"sub{i}_train/Data.mat")
        train_data = sio.loadmat(train_path)
        x = train_data['x_data']  # (N,C,S)

        xt = np.transpose(x, (0,2,1))
        E = np.matmul(x, xt) / x.shape[-1]  # /S
        R = np.mean(E, axis=0)
        R = (R + R.T) / 2.0
        R = R + eps * np.eye(R.shape[0], dtype=R.dtype)
        Rs.append(R)

    R_ref = np.mean(np.stack(Rs, axis=0), axis=0)
    R_ref = (R_ref + R_ref.T) / 2.0
    R_ref = R_ref + eps * np.eye(R_ref.shape[0], dtype=R_ref.dtype)
    return R_ref