my_list = [1, 2, 3, 4, 5]
my_iter = iter(my_list)

print(next(my_iter))  # 输出 1
print(next(my_iter))  # 输出 2
print(next(my_iter))  # 输出 3
print(next(my_iter))  # 输出 4
print(next(my_iter))  # 输出 5

# 已经到达末尾，再次调用会引发 StopIteration 异常
# print(next(my_iter))  # 引发 StopIteration


index = 3
value = my_list[3]
print(f'{value}')