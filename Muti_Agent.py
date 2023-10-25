import numpy as np
from prioritization_network import prioritization_net
from DQN_agent import DQN
import torch
import os
import torch.optim as optim
from copy import deepcopy as dc
from env.utils import calculate_dis
from buffer import buffer
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from prioritization_net_buffer import Prioritization_Net_Buffer
import collections
import math
class Muti_agent():
    def __init__(self, params):#初始化写法

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.test_steps_num=collections.deque(maxlen=10) #deque容器,一局内走的步数

        self.params=params
        self.train_model = self.params["train_model"]
        self.replay_buffer=buffer(params)#实例化
        self.agent_list={}
        # self.min_steps =self.params[]
        for pursuer_id in self.params["pursuer_ids"]:#params["pursuer_ids"] = ["p0", "p1", "p2", "p3", "p4", "p5"]
            self.agent_list[pursuer_id]=DQN(self.params,pursuer_id)#实例化，agent_list里的["p1":DQN1,"p2":DQN2......."p5":DQN5]
        if self.train_model=="evaluate":
            num_param=self.agent_list[self.params["pursuer_ids"][0]].get_param().shape[-1]#每个追车dqn网络的参数数量
            # num_critc_param=self.agent_list[self.params["pursuer_ids"][0]].get_critic_param().shape[-1]
            self.pr_exp_net = prioritization_net(num_param,self.params["lane_code_length"]+1,
                                                 self.params["num_evader"],self.params["max_steps"])#
            self.pr_exp_net.to(self.device)
            self.pr_net_optimizer = optim.Adam(self.pr_exp_net.parameters(), self.params["evaluate_net_learning_rate"])
            self.prioritization_net_buffer=Prioritization_Net_Buffer(params)
            self.beta = self.params["beta0"]
            self.lamda=self.params["lamda"]#两个参数


    def process_state(self,state):
        #return all_states dic for agent networks:{pursuer_id: {ego_pos: [ego_pos], target_pos:[target_pos],traffic_state: [traffic_state],topo_link_array: [topo_link_array],all_evaders_pos:[all_evaders_pos] } }
        #pro_state add"target"
        pro_state = dc(state)#深拷贝，防止指针
        all_states={}

        if not ("target" in state):
            pursuer_target = {}
        else:
            pursuer_target=state["target"]

        all_evaders_pos = []
        for evader_id in self.params["evader_ids"]:
            all_evaders_pos.append(dc(state["evader_pos"][evader_id]))#所有逃车的位置

        all_evaders_pos=[all_evaders_pos]#
        traffic_state=[dc(state["background_veh"])]
        topo_link_array=[dc(state["topology_array"])]

        for pursuer_id in self.params["pursuer_ids"]:
            ego_pos = [dc(state["pursuer_pos"][pursuer_id])]
            if "target" in state:
                target_pos=[dc(state["evader_pos"][state["target"][pursuer_id]])]#有目标就选
            else:
                ego_pos_tensor = torch.tensor(ego_pos, dtype=torch.float,device=self.device)
                all_evaders_pos_tensor = torch.tensor(all_evaders_pos, dtype=torch.float,device=self.device)
                steps_tensor=torch.tensor([[dc(state["steps"])]], dtype=torch.float,device=self.device)
                traffic_state_tensor=torch.tensor(traffic_state, dtype=torch.float,device=self.device)
                topo_link_array_tensor=torch.tensor(topo_link_array, dtype=torch.float,device=self.device)
                target_code = self.agent_list[pursuer_id].DQN_Net.select_target(steps_tensor,ego_pos_tensor,traffic_state_tensor,topo_link_array_tensor,all_evaders_pos_tensor)[0]#选择目标
                if state["evader_pos"][self.params["evader_ids"][target_code]][0] == -1:
                    target_id = self.min_dis_evader(state, pursuer_id)
                else:
                    target_id=self.params["evader_ids"][target_code]
                pursuer_target[pursuer_id]=target_id
                target_pos = [dc(state["evader_pos"][target_id])]
            ego_state={
                "ego_pos":ego_pos,
                "target_pos":target_pos,
                "traffic_state":traffic_state,
                "topo_link_array":topo_link_array,
                "all_evaders_pos":all_evaders_pos,
                "steps":[[dc(state["steps"])]]
            }#自身状态列表
            all_states[pursuer_id]= ego_state
        pro_state["target"]=dc(pursuer_target)
        return pro_state,all_states


    def select_action(self,pro_state, pro_all_states):
        # pro_state, pro_all_states=self.process_state(state)
        actions={}
        actions_prob={}
        for pursuer_id in self.params["pursuer_ids"]:
            ego_state=pro_all_states[pursuer_id]
            action,action_prob=self.agent_list[pursuer_id].select_action(ego_state)#好几个select_action
            actions[pursuer_id]=action
            actions_prob[pursuer_id]=action_prob
        return actions,actions_prob,pro_state["target"]


    def min_dis_evader(self,state,pursuer_id):
        min_dis = float('inf')
        min_evader_id = None
        for eva_index, evader_id in enumerate(self.params["evader_ids"]):
            if state["evader_pos"][evader_id][0] !=-1:
                eva_x, eva_y = state["evader_xy"][evader_id]["x"], state["evader_xy"][evader_id]["y"]
                dis = calculate_dis(eva_x, eva_y,
                                          state["pursuer_xy"][pursuer_id]["x"], state["pursuer_xy"][pursuer_id]["y"])
                if dis <= min_dis:
                    min_dis = dis
                    min_evader_id = evader_id
        return min_evader_id  #遍历寻找最短距离

    def train_agents(self):
        print("prepare for training......")
        agents_loss={}
        for pursuer_id in self.params["pursuer_ids"]:
            # loss_dic={}
            if self.train_model=="evaluate" :
                print("selecting training set for",pursuer_id,".....")
                train_set,chosen_exp,exp_prob,probs = self.select_train_set(self.agent_list[pursuer_id].get_param())
                self.prioritization_net_buffer.store_input_exp(pursuer_id,chosen_exp)#另一个网络的缓存
                w=math.pow(len(probs)*exp_prob,self.lamda)/max(np.power(len(probs)*probs,self.lamda))

            else:
                # train_sets=self.replay_buffer.random_sample()
                train_set = self.replay_buffer.random_sample()[-1]
                w=1
            print("training for", pursuer_id, ".....")
            loss=self.agent_list[pursuer_id].update(train_set,w)




            print(pursuer_id,"loss:",loss)
            agents_loss[pursuer_id]=loss
        return agents_loss

    def select_train_set(self, net_param):
        buffer_length = self.replay_buffer.get_length()
        ego_pos_input = np.zeros((buffer_length, self.params["max_steps"], self.params["lane_code_length"] + 1))
        eva_pos_input = np.zeros((buffer_length, self.params["max_steps"], self.params["num_evader"],
                                  self.params["lane_code_length"] + 1))
        action_input = np.zeros((buffer_length, self.params["max_steps"]))#np.zero 创建全0数组
        reward_input = np.zeros((buffer_length, self.params["max_steps"]))
        param_input = dc(net_param)
        # critic_param_input=dc(critic_param)
        for i in range(buffer_length):
            if i != 0:
                param_input = torch.cat((param_input, dc(net_param)), 0)
                # critic_param_input = torch.cat((critic_param_input, dc(critic_param)), 0)
            exp = dc(self.replay_buffer.memory_pool[i])
            if "ego_pos_eva_input" not in list(exp.keys()):

                exp_steps = len(exp["action"])
                ego_pos_input[i][0:exp_steps] = exp["state"]["ego_pos"]
                action_input[i][0:exp_steps] = exp["action"]
                reward_input[i][0:exp_steps] = exp["reward"]

                eva_pos_input[i][0:exp_steps] = exp["state"]["all_evaders_pos"]


                self.replay_buffer.memory_pool[i]["ego_pos_eva_input"] = dc(ego_pos_input[i])
                self.replay_buffer.memory_pool[i]["eva_pos_eva_input"] = dc(eva_pos_input[i])
                self.replay_buffer.memory_pool[i]["action_eva_input"] = dc(action_input[i])
                self.replay_buffer.memory_pool[i]["reward_eva_input"] = dc(reward_input[i])
            else:
                # print("=====")
                ego_pos_input[i] = exp["ego_pos_eva_input"]
                eva_pos_input[i] = exp["eva_pos_eva_input"]
                action_input[i] = exp["action_eva_input"]
                reward_input[i] = exp["reward_eva_input"]
        self.pr_exp_net.eval()
        value = self.pr_exp_net(param_input.type(torch.float32).to(self.device),
                                      torch.from_numpy(ego_pos_input).type(torch.float32).to(self.device),
                                      torch.from_numpy(eva_pos_input.swapaxes(1, 2)).type(torch.float32).to(
                                          self.device),
                                      torch.from_numpy(reward_input).type(torch.float32).to(self.device),
                                      torch.from_numpy(action_input).type(torch.float32).to(self.device)).T#这个网络的forward函数确实会返回一个value
        # print("=================")
        # print(value)

        probs = self.get_sample_prob(np.array(value))
        # print(value)
        # print("=================")
        c = Categorical(probs)#使用Categorical分布进行采样，得到一个随机的经验索引exp_index。
        exp_index = c.sample().item()
        chosen_exp = {#！！！这块不太懂
            "param_input": net_param.type(torch.float32).view(1, -1),
            "ego_pos_input": torch.from_numpy(np.expand_dims(ego_pos_input[exp_index], axis=0)).type(torch.float32),
            "eva_pos_input": torch.from_numpy(np.expand_dims(eva_pos_input[exp_index].swapaxes(0, 1), axis=0)).type(
                torch.float32),
            "reward_input": torch.from_numpy(np.expand_dims(reward_input[exp_index], axis=0)).type(torch.float32),
            "action_input": torch.from_numpy(np.expand_dims(action_input[exp_index], axis=0)).type(torch.float32)
        }#根据exp_index，从缓冲区中获取对应的经验，并将其输入数据整理成一个字典chosen_exp，用于训练网络。
        return self.replay_buffer.memory_pool[exp_index], chosen_exp,probs[exp_index],probs#复制返回出去

    def get_sample_prob(self,value):
        q=(value-min(value))/(max(value)-min(value))+0.00001
        p=np.power(q,self.beta)/sum(np.power(q,self.beta))
        self.beta=min(1,self.beta+0.05)
        return p#！！！根据value计算每个经验的采样概率probs

    def train_evaluate_net(self):

        all_loss = 0

        if self.prioritization_net_buffer.get_length()>self.params["evaluate_net_batch_size"]*0:
            print("training evaluate network......")
            for i in range(self.params["evaluate_net_update_times"]):
                self.pr_exp_net.eval()
                param_input, \
                ego_pos_input, eva_pos_input, \
                reward_input, action_input, target_output=\
                    self.prioritization_net_buffer.get_train_batch(self.params["evaluate_net_batch_size"])
                #从缓冲区赋值
                output =self.pr_exp_net(param_input.to(self.device),ego_pos_input.to(self.device),eva_pos_input.to(self.device),reward_input.to(self.device),action_input.to(self.device))
                loss = torch.nn.functional.mse_loss(output.view(-1,1), target_output.view(-1,1).to(self.device))
                self.pr_net_optimizer.zero_grad()
                loss.backward()
                self.pr_net_optimizer.step()
                all_loss += loss.item()
            self.save_evaluate_net(all_loss/self.params["evaluate_net_update_times"])
        return all_loss/self.params["evaluate_net_update_times"]

    def save_evaluate_net(self,loss):#存
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.pr_exp_net.state_dict(),
                   'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]+'/' +'evaluate_net.pth')

    def load_evaluate_net(self):   ###加载训练好的评估网络.pth是pytorch模型文件
        file_path = 'agent_param/' + self.params["env_name"] + '/' + self.params[
            "exp_name"] + '/' +'evaluate_net.pth'
        if os.path.exists(file_path):
            print("loading evaluate_net from param file....")
            self.pr_exp_net.load_state_dict(torch.load(file_path))
            self.pr_exp_net.to(self.device)
        else:
            print("creating new param for evaluate_net....")

    def load_params(self):
        for pursuer_id in self.params["pursuer_ids"]:
            self.agent_list[pursuer_id].load_param()

    # def load_critics_param(self):
    #     for pursuer_id in self.params["pursuer_ids"]:
    #         self.agent_list[pursuer_id].load_critic_param()
    #
    # def load_all_networks(self):
    #     self.load_actors_param()
    #     self.load_critics_param()









