import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from copy import deepcopy as dc
class prioritization_net(nn.Module):
    def __init__(self,num_param,num_pos,num_evader,max_steps):
        super(prioritization_net, self).__init__()
        # input:actor_param  critc_param  ego_pos evaders_pos action reward
        self.num_param=num_param
        # self.num_critc_param=num_critc_param
        self.num_pos=num_pos
        self.num_evader=num_evader
        self.max_steps=max_steps

        self.fc_n1 = nn.Linear(self.num_param+self.max_steps*2, 2048)
        self.fc_n2 = nn.Linear(2048, 512)#
        self.fc_n3 = nn.Linear(512, 128)



        #batch_size*max_steps*num_pos
        self.conv_ego_pos_1=nn.Conv1d(in_channels=self.max_steps,out_channels=512,kernel_size=2)
        self.conv_ego_pos_2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=2)
        self.conv_ego_pos_3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2)
        self.fc_ego_pos_1 = nn.Linear(64*(self.num_pos-3), 128)

        # batch_size*num_evader*max_steps*num_pos
        self.conv_eva_pos_1 = nn.Conv2d(in_channels=self.num_evader, out_channels=10, kernel_size=2,stride=2,padding=1)
        self.conv_eva_pos_2 = nn.Conv2d(in_channels=10, out_channels=2, kernel_size=2,stride=2,padding=1)
        self.fc_eva_pos_1 = nn.Linear(1206, 512)
        self.fc_eva_pos_2 = nn.Linear(512, 128)




        self.fc_out_1 = nn.Linear(128*3, 64)
        self.fc_out_2 = nn.Linear(64, 1)


    def forward(self,param,ego_pos,eva_pos,reward,action):

        input_all=torch.cat((param,reward,action),1)
        feature = F.elu(self.fc_n1(input_all))
        feature = F.elu(self.fc_n2(feature))
        feature = F.elu(self.fc_n3(feature))
        #通过三个全连接层得到一个特征向量


        ego_pos_feature = F.relu(self.conv_ego_pos_1(ego_pos))
        ego_pos_feature = F.relu(self.conv_ego_pos_2(ego_pos_feature))
        ego_pos_feature = F.relu(self.conv_ego_pos_3(ego_pos_feature))
        print(ego_pos_feature.size())
        #对自身位置输入和评估位置输入进行卷积操作，得到两个特征向量
        eva_pos_feature = F.relu(self.conv_eva_pos_1(eva_pos))
        eva_pos_feature_1 = F.relu(self.conv_eva_pos_2(eva_pos_feature))
        print(eva_pos_feature_1.size())

        ego_pos_feature=F.elu(self.fc_ego_pos_1(ego_pos_feature.view(-1,64*(self.num_pos-3))))
        # a=dc(eva_pos_feature)
        eva_pos_feature=F.elu(self.fc_eva_pos_1(eva_pos_feature_1.view(-1,1206)))
        eva_pos_feature=F.elu(self.fc_eva_pos_2(eva_pos_feature))



        all_features = torch.cat((feature, ego_pos_feature, eva_pos_feature), 1)
        all_features=F.elu(self.fc_out_1(all_features))
        value = self.fc_out_2(all_features)
#        value = self.fc_out_3(all_features)

        return value





