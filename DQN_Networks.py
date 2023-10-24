import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from copy import deepcopy as dc
import numpy as np
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_net(nn.Module):
    def __init__(self,num_edge,num_pos,num_evader,num_action=3):
        super(DQN_net, self).__init__()

        self.num_edge=num_edge
        self.num_pos=num_pos
        self.num_evader=num_evader
        self.num_action=num_action

        self.fc_traf_st = nn.Linear(self.num_edge, 48)#全连接层转换为48维度输出


        self.conv1_link=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=7,stride=3,padding=1)
        self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=3, padding=1)
        self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.fc_link = nn.Linear(81, 48)#将81维度转换成48维度的输出

        self.fc_feature_1 = nn.Linear(48*2+1, 32)
        self.fc_feature_2 = nn.Linear(32, 1)


        self.multihead_attn = nn.MultiheadAttention(num_pos+1,1)

        self.fc_hid1 = nn.Linear(self.num_pos*2+self.num_evader+1, 32)####全连接层定义，4层
        self.fc_hid2 = nn.Linear(32, 48)
        self.fc_hid3 = nn.Linear(48, 32)
        self.fc_hid4 = nn.Linear(32, 16)
        self.fc_action = nn.Linear(16, num_action)


    def forward(self, steps,ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos):
        # ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos
        #64*4*8
        #64*1



        topo=F.relu( self.conv1_link(topo_link_array.view(-1,1,self.num_edge,self.num_edge)))#
        topo = F.relu(self.conv2_link(topo))
        topo = F.relu(self.conv3_link(topo))
        topo=F.elu(self.fc_link(topo.view(-1,81)))#正向传递

        traffic_state_input=torch.tensor(traffic_state).view(-1,self.num_edge)
        traffic_state_input = F.elu(self.fc_traf_st(traffic_state_input))

        all_input = torch.cat((steps, traffic_state_input, topo), 1)#将steps，traffic_state_input和topo拼接在一起，得到all_input。
        feature = F.elu(self.fc_feature_1(all_input))
        feature = self.fc_feature_2(feature)

        ego_pos_input=torch.cat((ego_pos,feature),1)
        feature_cat=(feature.view(-1,1,1)).repeat(1,self.num_evader,1)
        all_evaders_pos_input=torch.cat((all_evaders_pos,feature_cat),2)
        atten_output, atten_weights = self.multihead_attn(torch.transpose(ego_pos_input.view(-1,1,self.num_pos+1),0,1), torch.transpose(all_evaders_pos_input,0,1), torch.transpose(all_evaders_pos_input,0,1))
        atten_weights=torch.transpose(atten_weights,0,1).view(-1,self.num_evader)#################用多头注意力机制处理，得到atten_output和atten_weights。

        # print(np.array(dc(atten_weights).cpu()).reshape(3).tolist())
        # path = 'log/' + '/atten_weight.csv'
        # with open(path, 'a+', newline="") as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow(np.array(dc(atten_weights).cpu()).reshape(3).tolist())

        all_features=torch.cat((ego_pos,target_pos,atten_weights,feature),1)
        all_features=F.relu(self.fc_hid1(all_features))
        all_features = F.relu(self.fc_hid2(all_features))
        all_features = F.relu(self.fc_hid3(all_features))
        all_features = F.relu(self.fc_hid4(all_features))
        Q_values = F.softmax(self.fc_action(all_features), dim=1)
        return Q_values

    def select_target(self,steps,ego_pos,traffic_state,topo_link_array,all_evaders_pos):
        topo = F.relu(self.conv1_link(topo_link_array.view(-1,1,self.num_edge,self.num_edge)))
        topo = F.relu(self.conv2_link(topo))
        topo = F.relu(self.conv3_link(topo))
        topo = F.elu(self.fc_link(topo.view(-1, 81)))  # 正向传递

        traffic_state_input = torch.tensor(traffic_state).view(-1, self.num_edge)
        traffic_state_input = F.elu(self.fc_traf_st(traffic_state_input))

        all_input = torch.cat((steps, traffic_state_input, topo),
                              1)  # 将steps，traffic_state_input和topo拼接在一起，得到all_input。
        feature = F.elu(self.fc_feature_1(all_input))
        feature = self.fc_feature_2(feature)

        ego_pos_input = torch.cat((ego_pos, feature), 1)
        feature_cat = (feature.view(-1, 1, 1)).repeat(1, self.num_evader, 1)
        all_evaders_pos_input = torch.cat((all_evaders_pos, feature_cat), 2)
        atten_output, atten_weights = self.multihead_attn(
            torch.transpose(ego_pos_input.view(-1, 1, self.num_pos + 1), 0, 1),
            torch.transpose(all_evaders_pos_input, 0, 1), torch.transpose(all_evaders_pos_input, 0, 1))
        # atten_output,atten_weights=self.multihead_attn(torch.transpose(ego_pos.view(-1,1,self.num_pos),0,1),torch.transpose(all_evader_pos,0,1),torch.transpose(all_evader_pos,0,1))

        batch_size=np.array(ego_pos.cpu()).shape[0]
        if batch_size==1:
            target_id=[torch.argmax(atten_weights.squeeze()).item()]#还剩一个车
        else:
            target_id=torch.argmax(atten_weights.squeeze(), dim=1).numpy().tolist()#从剩余车里面找权重最大的
        return target_id#返回网络选择的目标
