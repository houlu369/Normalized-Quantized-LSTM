import os
file = 'warpeace_input.txt'
assert os.path.exists(file), 'Please download warpeace_input.txt by yourself and place in data/linux directory.'

data = open(file, 'rb').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(file + ' has %d characters, %d unique.' % (data_size, vocab_size))
splits = [0.8, 0.1, 0.1]
train_num, val_num = data_size * splits[0], data_size * splits[1]
train_num, val_num = int(train_num), int(val_num)
train_data = data[:train_num]
val_data = data[train_num: train_num + val_num]
test_data = data[train_num + val_num:]
open('train.txt', 'wb').write(train_data)
open('valid.txt', 'wb').write(val_data)
open('test.txt', 'wb').write(test_data)
