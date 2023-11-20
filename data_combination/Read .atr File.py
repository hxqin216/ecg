from IPython.display import display
import wfdb

annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/101', 'atr')
display(annotation.__dict__)

sample = annotation.sample
print(f"Sample: {sample}")
# "atr"：用于读取 R-峰注释文件，它包含了心电信号中每个R波的位置。
# "qrs"：用于读取 QRS 注释文件，包含了QRS波的位置。
# "ecg"：用于读取一般的心电图注释文件。

symbol = annotation.symbol
print(f"symbol: {symbol}")