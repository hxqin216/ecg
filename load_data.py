import torch
import numpy as np

def load_data():
    X_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_EU_ST-T_V5_2.csv',
              'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_MIT_ST_ECG_2.csv',
              'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_SU_ECG_cleaned_2.csv']
    Y_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_EU_ST-T_V5_2.csv',
              'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_MIT_ST_ECG_2.csv',
              'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_SU_ECG_cleaned_2.csv']

    # 从文件加载数据和标签
    X = torch.tensor(np.loadtxt('data_load/X_MITARR_MLII_2.csv', delimiter=',', skiprows=1).astype('float64'))
    Y = np.loadtxt('data_load/Y_MITARR_MLII_2.csv', dtype="str", delimiter=',', skiprows=1)

    # 合并数据集
    print("begin concatenating...")
    for x_file, y_file in zip(X_list, Y_list):
        X = torch.cat((X, torch.tensor(np.loadtxt(x_file, delimiter=',', skiprows=1).astype(np.float64))))
        Y = np.concatenate((Y, np.loadtxt(y_file, dtype="str", delimiter=',', skiprows=1)))

    return X, Y