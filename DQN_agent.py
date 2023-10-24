import numpy as np
import os
from DQN_Networks import DQN_net
import random
import torch.optim as optim
import torch
from copy import deepcopy as dc
from torch.distributions import Normal, Categorical
from env.utils import calculate_dis
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import collections
class DQN():
    def __init__(self,params,pursuer_id):


        self.train_times=0
        self.update_times = 0
        # self.critic_update_times = 0
        self.pursuer_id=pursuer_id#这可是单个id

        self.params=params

        self.epsilon = self.params["epsilon"]
        self.epsilon_decay = self.params["epsilon_decay"] # "epsilon_decay": 0.0001,      # 贪婪策略epsilon衰减
        self.epsilon_min = self.params["epsilon_min"]
        self.num_actions=self.params["num_action"] #    params["num_action"] =3 向左，向右，向前
        self.tau=self.params["tau"]  #        "tau":0.005, #软更新系数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
        self.DQN_Net = DQN_net(self.params["num_edge"], self.params["lane_code_length"] + 1, self.params["num_evader"])
        #lane_code_length 是车道编码的长度，用于表示车辆所在的车道位置。一般为7，分别对应道路ID、方向、车道序号等。
        self.target_DQN_Net = DQN_net(self.params["num_edge"], self.params["lane_code_length"] + 1, self.params["num_evader"])
        self.DQN_Net.to(self.device)
        self.target_DQN_Net.to(self.device)

        self.load_param()
        self.target_DQN_Net.load_state_dict(self.DQN_Net.state_dict())#同步Q网络参数
        self.target_DQN_Net.eval()#将目标网络设置为评估模式，以确保在使用目标网络进行值函数估计或策略执行时具有稳定的行为。目标网络是新的，作为目标的网络

        self.batch_size=self.params["dqn_batch_size"]
        # if torch.cuda.is_available():
        #     self.actor_net=self.actor_net.cuda()
        #     self.critic_net=self.critic_net.cuda()


        self.DQN_optimizer = optim.Adam(self.DQN_Net.parameters(), self.params["DQN_learning_rate"])#Adam 优化器
        # self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), self.params["critic_learning_rate"])

        # self.best_critic_loss=float("inf")
        self.loss_list = collections.deque(maxlen=10)#
        self.test_reward=collections.deque(maxlen=10)

    def select_action(self, ego_state):##
        if np.random.rand() < self.epsilon:
            action_prob = F.softmax(
                torch.tensor(np.random.rand(1, self.num_actions), device=self.device, dtype=torch.float32), dim=1)#随机选择一个action
        else:
            ego_pos_tensor = torch.tensor(ego_state["ego_pos"], dtype=torch.float,device=self.device)
            target_pos_tensor=torch.tensor(ego_state["target_pos"], dtype=torch.float,device=self.device)
            traffic_state_tensor= torch.tensor(ego_state["traffic_state"], dtype=torch.float,device=self.device)
            topo_link_array_tensor=torch.tensor([ego_state["topo_link_array"]], dtype=torch.float,device=self.device)
            all_evaders_pos_tensor=torch.tensor(ego_state["all_evaders_pos"], dtype=torch.float,device=self.device)
            steps_tensor=torch.tensor(ego_state["steps"], dtype=torch.float,device=self.device)

            with torch.no_grad():#不训练，只用
                self.DQN_Net.eval()
                action_prob = self.DQN_Net(steps_tensor,ego_pos_tensor,target_pos_tensor,traffic_state_tensor,topo_link_array_tensor,all_evaders_pos_tensor)

        # target_pos,pursuers_pos,topo_link_array,background_veh,all_evader_pos
        action = torch.argmax(action_prob,dim=1).item()
        return action, action_prob[:,action].item()


    def update(self,training_set,w=1):
        self.DQN_Net.eval()
        all_loss=[]
        state=training_set["state"]
        next_state = training_set["next_state"]
        ego_pos_tensor=torch.tensor(state["ego_pos"], dtype=torch.float, device=self.device)
        target_pos_tensor=torch.tensor(state["target_pos"], dtype=torch.float, device=self.device)
        traffic_state_tensor=torch.tensor(state["traffic_state"], dtype=torch.float, device=self.device)
        topo_link_array_tensor=torch.tensor(state["topo_link_array"], dtype=torch.float, device=self.device)
        all_evaders_pos_tensor=torch.tensor(state["all_evaders_pos"], dtype=torch.float, device=self.device)
        steps_tensor=torch.tensor(state["steps"], dtype=torch.float, device=self.device).view(-1,1)
        #所有的量都转换成张量

        done = training_set["done"]
        done_tensor=torch.tensor(done).view(-1,1).to(self.device)
        action=training_set["action"]
        action_tensor=torch.tensor(action).view(-1,1).to(self.device)
        action_prob_tensor=torch.tensor(training_set["action_prob"], dtype=torch.float, device=self.device)
        reward_tensor=torch.tensor(training_set["reward"], dtype=torch.float32).view(-1,1).to(self.device)
        #还在转换张量
        next_ego_pos_tensor = torch.tensor(next_state["ego_pos"], dtype=torch.float, device=self.device)
        next_target_pos_tensor = torch.tensor(next_state["target_pos"], dtype=torch.float, device=self.device)
        next_traffic_state_tensor = torch.tensor(next_state["traffic_state"], dtype=torch.float, device=self.device)
        next_topo_link_array_tensor = torch.tensor(next_state["topo_link_array"], dtype=torch.float, device=self.device)
        next_all_evaders_pos_tensor = torch.tensor(next_state["all_evaders_pos"], dtype=torch.float, device=self.device)
        next_steps_tensor=torch.tensor(next_state["steps"], dtype=torch.float, device=self.device).view(-1,1)
        #仍然在转换张量

        for i in range(self.params["dqn_update_times"]):
            for index in BatchSampler(SubsetRandomSampler(range(action_tensor.shape[0])), self.batch_size, False):#开始一个循环，循环每个批次的索引，批次大小为self.batch_size。
                Q_values = self.DQN_Net(steps_tensor[index],ego_pos_tensor[index], target_pos_tensor[index], traffic_state_tensor[index],
                                                 topo_link_array_tensor[index], all_evaders_pos_tensor[index]).gather(1,
                                                                                                        action_tensor[index])  # new policy
                #用网络计算当前状态每一个Q值
                next_Q_values=self.target_DQN_Net(next_steps_tensor[index],next_ego_pos_tensor[index], next_target_pos_tensor[index], next_traffic_state_tensor[index],
                                                 next_topo_link_array_tensor[index], next_all_evaders_pos_tensor[index])
                #下一个状态每一个Q值
                next_Q_values = next_Q_values.max(1)[0].view(-1, 1)

                target_Q_values=reward_tensor[index]+self.params["gamma"]*(next_Q_values*done_tensor[index])

                self.DQN_optimizer.zero_grad() #梯度清零
                loss=F.mse_loss(Q_values, target_Q_values)*w  #mse loss函数
                loss.backward()#####对损失进行反向传播，计算梯度。
                self.DQN_optimizer.step() #用优化器更新神经网络模型的参数。
                all_loss.append(loss.item()#Q的loss
)

        self.soft_update_target_network()
        self.train_times=self.train_times+1#训练次数+1
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        return np.array(all_loss).mean()

    def save_param(self):#保存神经网络模型的参数到文件中。
#        self.update_times=self.update_times+1
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.update_times=self.update_times+1
        torch.save(self.DQN_Net.state_dict(), 'agent_param/' + self.params["env_name"] +'/'+self.params["exp_name"]+'/'+self.pursuer_id+'_dqn.pth')

    def load_param(self):#加载参数
        file_path='agent_param/' + self.params["env_name"] +'/'+self.params["exp_name"]+'/'+self.pursuer_id+'_dqn.pth'
        if os.path.exists(file_path):
            print("loading", self.pursuer_id,"from param file....")
            self.DQN_Net.load_state_dict(torch.load(file_path))
            self.DQN_Net.to(self.device)
        else:
            print("creating new param for", self.pursuer_id,"....")

    def update_target_network(self):
        if self.train_times % self.params["target_update_period"] == 0:#每隔几步更新一下
            self.target_DQN_Net.load_state_dict(self.DQN_Net.state_dict())
            self.target_DQN_Net.eval()

    def soft_update_target_network(self):#软更新目标神经网络模型的参数，使其接近神经网络模型的参数
        if self.train_times % self.params["target_update_period"]== 0:
            for param, target_param in zip(self.DQN_Net.parameters(), self.target_DQN_Net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_param(self):
        param_num=0#参数计数器
        self.DQN_Net.eval()
        for parm_index,parm_key in enumerate(self.DQN_Net.state_dict().keys()):#遍历DQN网络的状态字典，即包含所有参数的字典。
            if ("bias" in str(parm_key)) or ("weight" in str(parm_key)):
                if param_num==0:
                    all_param=dc(torch.flatten(self.DQN_Net.state_dict()[parm_key]))
                else:
                    all_param=torch.cat((all_param,dc(torch.flatten(self.DQN_Net.state_dict()[parm_key]))))#这个state_dict（）是继承的
                param_num=param_num+1
        return all_param.view(1,-1) #DQN_Net.state_dict()字典里的所有参数的数量








