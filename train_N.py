import torch
from torch import nn

def train(Net,train_data,text_data,num,optimizer,criterion):
    
    train_len=0
    num_correct=0
    j=1
    while True:
        i=1
        for im,label in train_data:
            #print(im)
            #print(im.size())
            #print(label)
            #print(Net)
            output = Net(im)
            loss = criterion(output,label)
            print('%d-%d loss is'%(j,i),loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,pred = torch.max(output,1)
            num_correct += (pred == label).sum().item()
            train_len += len(im)
            i+=1
        j+=1
        #print('-'*20)
        print('答对率是：',num_correct/train_len)
        print('-'*20)
        if(loss.item()<0.001):
            break
    ##接下来开始检测

    Net = Net.eval()
    text_len = 0
    text_correct = 0
    
    for im,label in text_data:

        output = Net(im)
        _,pred = torch.max(output,1)
        text_correct = (pred == label).sum().item()
        text_len += len(im)

    print('训练后答对率为：',text_correct/text_len)
