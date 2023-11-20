import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import wfdb
import time
import pandas as pd

# start_time = time.time()

# type = []
# rootdir = 'mit-bih-st-change-database-1.0.0'
# rootdir = 'sudden-cardiac-death-holter-database-1.0.0'
# rootdir = 'european-st-t-database-1.0.0'
rootdir = 'mit-bih-arrhythmia-database-1.0.0'

files = os.listdir(rootdir)
name_list = []
MLII = []
type = {}

for file in files:
    if file[0:3] in name_list:
        continue
    else:
        name_list.append(file[0:3])

for name in name_list:
    if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        continue
    record = wfdb.rdrecord(rootdir + '/' + name)

    if 'MLII' in record.sig_name:
        MLII.append(name)
        annotation = wfdb.rdann(rootdir+'/'+name, 'atr')
    for symbol in record.sig_name:
        if symbol in list(type.keys()):
            type[symbol] += 1
        else:
            type[symbol] = 1
    print('symbol_name', type)
