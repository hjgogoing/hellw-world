'学到这里，我有点迷茫'

import torch
from torch import nn
from torch.utils.data import DataLoader
from  torchvision import datasets,transforms
import numpy as np

from train_N import train

def data_t(x):
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])]
        )
    x = data_tf(x)
    #x = x.resize((96,96),1)
    return x

'''
def data_tf(x):
    #x = x.resize((96, 96), 2) # 将图片放大到 96 x 96
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    #x = x.transpose((2, 0, 1)) # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x
'''
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 3),   #16 26 26
            nn.BatchNorm2d(16),
            nn.ReLU(True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size = 3),   #32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2,2))                  #32 12 12

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3),   #64 10 10
            nn.BatchNorm2d(64),
            nn.ReLU(True))
 

        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 3),  #128 8 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2,2))                  #128 4 4


        self.fc = nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(True),
            nn.Linear(1024,128),
            nn.ReLU(True),
            nn.Linear(128,10))

    def forward(self,x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


####...使用自写的训练模块...#####


train_dataset = datasets.MNIST(
    root = './data',train = True,transform = data_t,download = True)

text_dataset = datasets.MNIST(
    root = './data',train = False,transform = data_t,download = True)
    
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True)
text_loader  = DataLoader(text_dataset,batch_size = 64,shuffle = False)

Net = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net.parameters(),lr = 1e-2)
        
train(Net,train_loader,text_loader,30,optimizer,criterion)



            
