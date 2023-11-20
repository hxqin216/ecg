import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def preprocess_data():
    torch.manual_seed(7)
    X_list = [r'C:\Users\qinxi\PycharmProjects\ecg_zh\data_load\3X_eu_MLII.csv']
    Y_list = [r'C:\Users\qinxi\PycharmProjects\ecg_zh\data_load\3Y_eu_MLII.csv']
    X = torch.tensor(np.loadtxt())
    return train_loader, test_loader
