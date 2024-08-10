import torch
import time
from model import DNN_model
from dataset import MyData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
# 使用的参数
epoch_num=3000
batchsize_num=64
learnRate=0.01
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

path_train="./bert_Train.pk"
path_validation="./bert_Test.pk"

dataset_train=MyData(path_train)
dataset_validation=MyData(path_validation)




dataloader_train=DataLoader(dataset=dataset_train,batch_size=batchsize_num,shuffle=True)
dataloader_validation=DataLoader(dataset=dataset_validation,batch_size=batchsize_num,shuffle=True)


def make_batch(inputs):
    x,y=inputs
    # print(x.shape)
    # print(y)
    # list_y=[]
    # for i in range(len(y)):
    #     if(int(y[i][0])==0):
    #         list_y.append([0])
    #     else:
    #         list_y.append([1])
    return (x.to(device),y.view(len(y)))

model=DNN_model()
model.to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=learnRate, weight_decay=1e-6)

print("开始训练！")
aa = time.time()  # 开始时间

for epoch in range(epoch_num):
    runing_loss = 0.0
    model.train()
    corret_num = 0
    aa = time.time()#开始时间
    for i, data in enumerate(dataloader_train):
        input,label=make_batch(data)#数据加载
        optimizer.zero_grad()  # 2.初始化梯度
        #print(input.shape)
        input=input.view(input.shape[0],1,3,256)
        #print(input.shape)
        output = model(input)  # 3.计算前馈
        loss = F.cross_entropy(output.cpu(), label)  # 4.计算损失
        loss.backward()  # 5.计算梯度
        optimizer.step()  # 6.更新权值
        runing_loss += loss.item()

        for j in range(output.shape[0]):
            output_label = output[j]
            if ((int)(label[j].item()) == int(torch.max(F.softmax(output_label).unsqueeze(0), 1).indices.item())):
                # print(int(torch.max(F.softmax(output_label).unsqueeze(0), 1).indices.item()))
                corret_num += 1
    # if(epoch%50==0):
    #     PATH = "./model_wehave/state_dict_model_" + str(epoch) + ".pth"
    #     torch.save(model.state_dict(), PATH)
    sum_num = len(dataset_train)
    bb = time.time()  # 结束时间.

    cc = bb - aa
    print("第"+str(epoch)+"个epoch"+ ":用了" + str(cc // 1) + "秒！")
    print("训练集合句子数量：" + str(sum_num))
    print("正确数量为：" + str(corret_num))
    sum_num = len(dataset_validation)
    print("测试集合句子数量：" + str(sum_num))
    corret_num = 0
    for i, data in enumerate(dataloader_validation):
        model.eval()
        input,label=make_batch(data)#数据加载
        input=input.view(input.shape[0],1,3,256)
        output = model(input)  # 3.计算前馈
        for j in range(output.shape[0]):
            output_label=output[j]

            if((int)(label[j].item())==int(torch.max(F.softmax(output_label).unsqueeze(0),1).indices.item())):
                corret_num+=1
    print("正确数量为："+str(corret_num))

