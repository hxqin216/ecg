import matplotlib.pyplot as plt
from IPython.display import display
import wfdb
import seaborn as sns
import plotly.express as px
import numpy as np
f = 360
segment_len = 10
k = 0
# record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/105')

# sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len

record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/105',sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len)
display(record.__dict__)
# fr = record.fs
# print(f"fr:{fr}")
#
# 绘图
signal_data = record.p_signal.transpose()
p_signal_data = signal_data[0]
p_signal_data_p = p_signal_data.transpose()
sns.set_style("darkgrid")

plt.plot(p_signal_data_p)
plt.title("signal record")
plt.xlabel("time")
plt.ylabel("amp")
plt.show()

# whole_signal = wfdb.rdrecord('ecg_data/101').p_signal.transpose()
# print("S")

# 测试sig_len用法
# signal_length = record.sig_len
# print(f"Signal Length: {signal_length} data points")
# fs：采样频率；
# n_sig：信号通道数；
# sig_len：信号长度；
# p_signal：模拟信号值，储存形式为ndarray或者是list；
# d_signal：数字信号值，储存形式为ndarray或者是list。
# 这些属性都能直接进行访问（如：使用record.fs可以直接读取到采样频率)。


# 假设 signal_data 包含从 wfdb 读取的生理信号数据
# time_points = np.linspace(0, len(signal_data) - 1, len(signal_data))
#
# fig = px.line(x=time_points, y=signal_data)
# fig.update_layout(title="Physiological Signal Record", xaxis_title="Time", yaxis_title="Amplitude")
# fig.show()




