# coding='utf-8'

import torch

import torch.nn as nn
import torch.optim as optim


from data.dataset import   MyDataset
from models.net_points import MainNet
import matplotlib.pylab as plt
import shutil
import os

import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default="400", type=int)
parser.add_argument("--lr", default="0.0001", type=float)
args = parser.parse_args()
batch_size = 64
def train(point,name):


    net = MainNet(point)

    loss_mse_fn = nn.MSELoss()

    # Optimizer and learning rate
    opt = optim.Adam(net.parameters(), lr=args.lr,weight_decay=1e-5)

    net = net.cuda()
    if os.path.exists('../weight/weights_{}.pt'.format(name)):
        net.load_state_dict(torch.load(r'../weight/weights_{}.pt'.format(name)))

    lr = args.lr
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式getOverlap_Line
    plt.ion()
    train_=[]
    val_ = []
    for epoch in range(args.epochs):
        print("[i]: {}".format(epoch))

        for dir in   ['train.txt','val.txt']:
            if dir =='train.txt':
                if epoch%40==3 and epoch>40:
                    lr = lr*0.95
                    opt = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-5 )

                dataloader = torch.utils.data.DataLoader(MyDataset(r'../train_data/{}/train.txt'.format(name)),
                                                         batch_size=batch_size,
                                                         shuffle=True, num_workers=0, pin_memory=True)
                loss_train = 0
                for step, samples in enumerate(dataloader):
                    images, targets= samples[0].cuda(),samples[1].cuda()

                    # Forward and backward
                    opt.zero_grad()
                    outputs = net(images)
                    loss = loss_mse_fn(targets,outputs)
                    loss_train +=loss.item()
                    loss.backward()
                    opt.step()

                    if step % 100 == 99:
                        torch.save(net.state_dict(), '../weight/weights_{}.pt'.format(name))
                        # torch.save(nets.state_dict(), '../weight/weights_.pt')
                        shutil.copy('../weight/weights_{}.pt'.format(name), '../weight/weights_copy_{}.pt'.format(name))
                        print('save......')

                torch.save(net.state_dict(), '../weight/weights_{}.pt'.format(name))
                # torch.save(nets.state_dict(), '../weight/weights_.pt')
                shutil.copy('../weight/weights_{}.pt'.format(name), '../weight/weights_copy_{}.pt'.format(name))
                print('save......')
                loss_train_mean = loss_train/(step+1)
                train_.append(loss_train_mean)

            else:
                dataloader = torch.utils.data.DataLoader(MyDataset(r'../train_data/{}/val.txt'.format(name)),
                                                         batch_size=batch_size,
                                                         shuffle=True, num_workers=0, pin_memory=True)
                loss_val=0
                net.eval()
                for step, samples in enumerate(dataloader):
                    images, targets = samples[0].cuda(), samples[1].cuda()
                    opt.zero_grad()
                    with torch.no_grad():
                        outputs = net(images)
                    loss = loss_mse_fn(targets, outputs)
                    loss_val +=loss.item()
                loss_val_mean = loss_val/(step+1)
                val_.append(loss_val_mean)



        plt.plot(train_,'r-')
        plt.plot(val_,'g-')
        plt.pause(0.1)
        plt.ioff()
    plt.savefig('./loss——{}.png'.format(name))


def A(point,name):
    loss_mse_fn = nn.MSELoss()
    net = MainNet(point).cuda()
    net.eval()
    net.load_state_dict(torch.load(r'../weight/weights_{}.pt'.format(name)))
    dataloader = torch.utils.data.DataLoader(MyDataset(r'../train_data/{}/test.txt'.format(name)),
                                             batch_size=32,
                                             shuffle=True, num_workers=0, pin_memory=True)
    loss_test = 0
    for step, samples in enumerate(dataloader):
        images, target = samples[0].cuda(), samples[1].cuda()

        with torch.no_grad():
            outputs = net(images)
        print(target,outputs)
        loss = loss_mse_fn(target, outputs)


        loss_test  += loss.item()
    loss_test_mean = loss_test  / (1+step)
    print('test_loss:', loss_test_mean, 1111111)


if __name__ == "__main__":
    names = [[3,'ROI']] #[3,'ROI'],[3,'C2'],[8,'C3'],[8,'C4']
    for name in names:
        train(name[0],name[1])
        # A(name[0],name[1])

