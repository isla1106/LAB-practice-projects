import torch # torch==1.7.1
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

MAX_WORD = 10000  # 只保留最高频的10000词
MAX_LEN = 300     # 句子统一长度为200
word_count={}     # 词-词出现的词数 词典

#清理文本，去标点符号，转小写
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

# 分词方法
def tokenizer(sentence):
    return sentence.split()

#  数据预处理过程
def data_process(text_path, text_dir): # 根据文本路径生成文本的标签

    print("data preprocess")
    file_pro = open(text_path,'w',encoding='utf-8')
    for root, s_dirs, _ in os.walk(text_dir): # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取train和test文件夹下所有的路径
            text_list = os.listdir(i_dir)
            tag = os.path.split(i_dir)[-1] # 获取标签
            if tag == 'pos':
                label = '1'
            if tag == 'neg':
                label = '0'
            if tag =='unsup':
                continue

            for i in range(len(text_list)):
                if not text_list[i].endswith('txt'): # 判断若不是txt,则跳过
                    continue
                f = open(os.path.join(i_dir, text_list[i]),'r',encoding='utf-8') # 打开文本
                raw_line = f.readline()
                pro_line = clean_str(raw_line)
                tokens = tokenizer(pro_line) # 分词统计词数
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] = word_count[token] + 1
                    else:
                        word_count[token] = 0
                file_pro.write(label + ' ' + pro_line +'\n')
                f.close()
                file_pro.flush()
    file_pro.close()

    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item : item[1], reverse=True) # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab

# 定义Dataset
class MyDataset(Dataset):
    def __init__(self, text_path):
        file = open(text_path, 'r', encoding='utf-8')
        self.text_with_tag = file.readlines()  # 文本标签与内容
        file.close()

    def __getitem__(self, index): # 重写getitem
        line = self.text_with_tag[index] # 获取一个样本的标签和文本信息
        label = int(line[0]) # 标签信息
        text = line[2:-1]  # 文本信息
        return text, label

    def __len__(self):
        return len(self.text_with_tag)


# 根据vocab将句子转为定长MAX_LEN的tensor
def text_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:
        sentence_idx = [vocab[token] if token in vocab.keys() else vocab['<UNK>'] for token in tokenizer(sentence)] # 句子分词转为id

        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN-len(sentence_idx)): # 对长度不够的句子进行PAD填充
                sentence_idx.append(vocab['<PAD>'])

        sentence_idx = sentence_idx[:MAX_LEN] # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
    return torch.LongTensor(sentence_index_list) # 将转为idx的词转为tensor


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) # embedding层

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1] # 取LSTM最后一层结果
        outs = self.softmax(self.decoder(encoding)) # 输出层为二维概率[a,b]
        return outs


# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) # embedding层

        self.encoder = nn.GRU(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1] # 取LSTM最后一层结果
        outs = self.softmax(self.decoder(encoding)) # 输出层为二维概率[a,b]
        return outs



# 模型训练
def train(model, train_data,test_data, vocab, epoch=15):
    print('train model')
    model = model.to(device)
    loss_sigma = 0.0
    correct = 0.0
    # 定义损失函数和优化器
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    for epoch_num in range(epoch):
        model.train()
        print("epoch:",epoch_num+1,"training")
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):
            train_x = text_transform(text, vocab).to(device)
            train_y = label.to(device)
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred.log(), train_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, train_y)
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)
        train_loss_all.append(avg_loss)
        train_acc_all.append(avg_acc)
        print("train_avg_loss:", avg_loss, " train_avg_acc:,", avg_acc)
        # 保存训练完成后的模型参数

        print("epoch:",epoch_num+1,"eval")
        model.eval()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(test_data)):
            test_x = text_transform(text, vocab).to(device)
            test_y = label.to(device)
            optimizer.zero_grad()
            pred = model(test_x)
            loss = criterion(pred.log(), test_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, test_y)
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(test_data)
        avg_acc = avg_acc / len(test_data)
        test_loss_all.append(avg_loss)
        test_acc_all.append(avg_acc)
        print("test_avg_loss:", avg_loss, " test_avg_acc:,", avg_acc)

    torch.save(model.state_dict(), 'LSTM_IMDB_parameter.pkl')

    train_process = pd.DataFrame(
        data={"epoch": range(epoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "test_loss_all": test_loss_all,
              "test_acc_all": test_acc_all})
    return model, train_process

# 模型测试
def test(model, test_data, vocab):
    print('test model')
    model = model.to(device)
    model.eval()
    avg_acc = 0
    # avg_acc_0=0
    # class_correct_0 = 0
    # class_total_0 = 0
    # avg_acc_1=0
    # class_correct_1 = 0
    # class_total_1 = 0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text, vocab).to(device)
        train_y = label.to(device)
        pred = model(train_x)
        label_pred = pred.max(dim=1)[1]
        c = (label_pred == train_y).squeeze()
        avg_acc += accuracy(pred, train_y)
        # print("train_x:",train_x.size(),"train_y",train_y.size(),"pred:",pred.size(),"c:",c.size())
        # print("c:",c)  
        for i in range(len(label_pred)):
            y = train_y[i]
            class_correct[y] += c[i]
            class_total[y] += 1
    avg_acc = avg_acc / len(test_data)


    print('Accuracy : %d %%' % (100 * avg_acc))
    print('Accuracy of class0 : %2d %%' % (100 * class_correct[0] / class_total[0]))
    print('Accuracy of class1 : %2d %%' % (100 * class_correct[1] / class_total[1]))

    return avg_acc

# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)



train_dir = '/content/data/aclImdb/train'  # 原训练集文件地址
train_path = './train.txt'  # 预处理后的训练集文件地址

test_dir = '/content/data/aclImdb/test'  # 原训练集文件地址
test_path = './test.txt'  # 预处理后的训练集文件地址

# vocab = data_process(train_path, train_dir) # 数据预处理
# data_process(test_path, test_dir)
# np.save('vocab.npy', vocab) # 词典保存为本地
vocab = np.load('./vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

# 构建MyDataset实例
train_data = MyDataset(text_path=train_path)
test_data = MyDataset(text_path=test_path)

# print("train_data.__getitem__(1)",test_data.__getitem__(1))

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False)

# 生成LSTM模型
# model = LSTM(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)  # 定义模型
model = GRU(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)  # 定义模型

model, train_process = train(model=model, train_data=train_loader,test_data=test_loader, vocab=vocab, epoch=10)
train_process.to_csv("model_process.csv", index=False)
## 可视化模型训练过程中
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all,
          "r.-", label="Train loss")
plt.plot(train_process.epoch, train_process.test_loss_all,
          "bs-", label="Test loss")
plt.legend()
plt.xlabel("Epoch number", size=13)
plt.ylabel("Loss value", size=13)
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all,
          "r.-", label="Train acc")
plt.plot(train_process.epoch, train_process.test_acc_all,
          "bs-", label="Test acc")
plt.xlabel("Epoch number", size=13)
plt.ylabel("Acc", size=13)
plt.legend()
plt.show()
# 加载训练好的模型
model.load_state_dict(torch.load('LSTM_IMDB_parameter.pkl', map_location=torch.device('cpu')))


# 测试结果
print("testing")
acc = test(model=model, test_data=test_loader, vocab=vocab)
print(acc)


