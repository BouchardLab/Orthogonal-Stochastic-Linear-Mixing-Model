import numpy as np
# from scipy.io import loadmat
# from scipy.io import savemat
from hdf5storage import loadmat
from hdf5storage import savemat
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle


def process(domain, save=False, dat_perm_seeds=1234):
    data = {}
    data['dname'] = domain

    if domain == 'jura':
        filename = 'data/jura/jura.mat'
        raw = loadmat(filename, squeeze_me=True, struct_as_record=False, mat_dtype=True)['jura']
        X_all = raw[:, 0:2]  # X, Y
        Y_all = raw[:, [2, -2, -1]]  # Cd, Ni, Zn
        N_all = X_all.shape[0]
        N_train = 249
        N_test = 100
        DList = [3, 1, 1]
        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    elif domain == 'pm25':
        filename = 'data/pm25/pm25.mat'
        raw = loadmat(filename, struct_as_record=False, mat_dtype=True)
        X_all = raw['Xall'].T
        Y_all = raw['yall'].T
        Y_all = Y_all.reshape([Y_all.shape[0], Y_all.shape[1] * Y_all.shape[2]])
        N_all = X_all.shape[0]
        N_train = 256
        N_test = 32
        DList = [10, 10, 1]
        # padding Y in case of the original D cannot decompse into small factors
        gap = DList[0] * DList[1] * DList[2] - Y_all.shape[1]
        if gap is not 0:
            Y_all = np.concatenate((Y_all, np.zeros([Y_all.shape[0], gap])), axis=1)
        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    elif domain == "concrete":
        filename = 'data/concrete_dataset/slump_test.data'
        raw = pd.read_csv(filename)
        # breakpoint()
        X_all = raw.iloc[:, 1:8].values
        Y_all = raw.iloc[:, 8:].values
        N_all = X_all.shape[0]
        N_train = 80
        N_test = 23
        DList = [3,1,1]
        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    elif domain == "equity":
        filename_X = 'data/equity/data.csv'
        filename_Y = 'data/equity/truth.txt'
        raw_X = pd.read_csv(filename_X, header=None, sep='  ', engine='python')
        raw_Y = pd.read_csv(filename_Y, header=None, sep='  ', engine='python')
        # breakpoint()
        X_all = raw_X.values
        Y_all = raw_Y.values
        N_all = X_all.shape[0]
        N_train = 200
        N_test = 200
        DList = [5, 5, 1]
        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    elif domain == 'neuron':
        filename = 'data/ECoG/ecog.plk'
        with open(filename, 'rb') as f:
            times, resps = pickle.load(f)
        X_all = times.reshape([-1,1])  # X, Y
        Y_all = resps  # Cd, Ni, Zn
        N_all = X_all.shape[0]
        N_train = 100
        N_test = 100
        DList = [128, 1, 1]
        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    else:
        print('No valid dataset found... program terminated...')
        return

    scaler = StandardScaler()
    scaler.fit(X_all)
    N_X_all = scaler.transform(X_all)
    X_mean = scaler.mean_
    X_std = scaler.scale_
    scaler.fit(Y_all)
    N_Y_all = scaler.transform(Y_all)
    Y_mean = scaler.mean_
    Y_std = scaler.scale_

    np.random.seed(dat_perm_seeds)
    perm = np.random.permutation(N_all)
    N_X_all = N_X_all[perm]
    N_Y_all = N_Y_all[perm]
    X_train = N_X_all[0:N_train, :]
    X_test = N_X_all[-N_test:, :]
    Y_train = N_Y_all[0:N_train, :]
    Y_test = N_Y_all[-N_test:, :]

    data['N_all'] = N_all
    data['N_train'] = N_train
    data['N_test'] = N_test

    data['X_all'] = X_all
    data['Y_all'] = Y_all
    data['DList'] = DList

    data['X_train'] = X_train
    data['X_test'] = X_test
    data['X_mean'] = X_mean
    data['X_std'] = X_std

    data['Y_train'] = Y_train
    data['Y_test'] = Y_test
    data['Y_mean'] = Y_mean
    data['Y_std'] = Y_std
    data['Y_test_ground'] = Y_test * Y_std + Y_mean

    if save == True:
        if not os.path.exists('processed'):
            os.makedirs('processed')
        savemat('processed/' + data['dname'] + '.mat', data)
        print('saved to', 'processed/' + data['dname'] + '.mat')

    return data

if __name__ == "__main__":
    data = process("jura", save=True)
    data = process("pm25", save=True)
    data = process("concrete", save=True)
    data = process("equity", save=True)
    data = process("neuron", save=True)
