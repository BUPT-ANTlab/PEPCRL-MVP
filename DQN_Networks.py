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

        self.fc_n1 = nn.Linear(num_pos+num_pos+num_edge+1, 32)


        self.conv1_link=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=7,stride=3,padding=1)
        self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=3, padding=1)
        self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.fc_link = nn.Linear(81, 24)

        self.multihead_attn = nn.MultiheadAttention(num_pos,1)

        self.fc_hid1 = nn.Linear(32+24+num_evader, 48)
        self.fc_action = nn.Linear(48, num_action)

    def forward(self, steps,ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos):
        # ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos
        all_input=torch.cat((steps,ego_pos,target_pos,traffic_state),1)
        feature = F.elu(self.fc_n1(all_input))


        topo=F.relu( self.conv1_link(topo_link_array))
        topo = F.relu(self.conv2_link(topo))
        topo = F.relu(self.conv3_link(topo))
        # print("#############")
        # print(topo.size())
        topo=F.elu(self.fc_link(topo.view(-1,81)))


        atten_output, atten_weights = self.multihead_attn(torch.transpose(ego_pos.view(-1,1,self.num_pos),0,1), torch.transpose(all_evaders_pos,0,1), torch.transpose(all_evaders_pos,0,1))
        atten_weights=torch.transpose(atten_weights,0,1).view(-1,self.num_evader)
        print(np.array(dc(atten_weights).cpu()).reshape(3).tolist())
        path = 'log/' + '/atten_weight.csv'
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(np.array(dc(atten_weights).cpu()).reshape(3).tolist())

        all_features=torch.cat((feature,topo,atten_weights),1)
        all_features=F.elu(self.fc_hid1(all_features))
        Q_values = F.softmax(self.fc_action(all_features), dim=1)
        return Q_values

    def select_target(self,ego_pos,all_evader_pos):
        atten_output,atten_weights=self.multihead_attn(torch.transpose(ego_pos.view(-1,1,self.num_pos),0,1),torch.transpose(all_evader_pos,0,1),torch.transpose(all_evader_pos,0,1))
        batch_size=np.array(ego_pos.cpu()).shape[0]
        if batch_size==1:
            target_id=[torch.argmax(atten_weights.squeeze()).item()]
        else:
            target_id=torch.argmax(atten_weights.squeeze(), dim=1).numpy().tolist()
        return target_id
