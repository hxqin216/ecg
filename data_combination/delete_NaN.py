import pandas as pd

# 读取X_SU_ECG.csv文件
input_x_csv = 'X_SU_ECG_2.csv'  # 输入X数据的CSV文件名
input_y_csv = 'Y_SU_ECG_2.csv'  # 输入Y数据的CSV文件名
output_x_csv = 'X_SU_ECG_cleaned_2.csv'  # 输出X数据的CSV文件名
output_y_csv = 'Y_SU_ECG_cleaned_2.csv'  # 输出Y数据的CSV文件名

# 读取X和Y数据
x_df = pd.read_csv(input_x_csv)
y_df = pd.read_csv(input_y_csv)

# 找到X数据中的空行的索引
empty_rows_index = x_df.index[x_df.isnull().all(axis=1)].tolist()

# 删除X和Y数据中的对应空行
cleaned_x_df = x_df.drop(empty_rows_index)
cleaned_y_df = y_df.drop(empty_rows_index)

# 保存处理后的X和Y数据
cleaned_x_df.to_csv(output_x_csv, index=False)
cleaned_y_df.to_csv(output_y_csv, index=False)

print("X_SU_ECG_2.csv和Y_SU_ECG_2.csv中的空行已删除，并保存到", output_x_csv, "和", output_y_csv)
