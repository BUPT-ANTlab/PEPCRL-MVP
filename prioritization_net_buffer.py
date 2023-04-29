import numpy as np
import random
import torch
from copy import deepcopy as dc
class Prioritization_Net_Buffer():
    def __init__(self,params):
        self.params=params
        self.evaluate_net_memory_pool = []
        #[{"actor_param_input","critc_param_input","ego_pos_input","eva_pos_input","reward_input","action_input","target_output"}]
        self.max_capacity = self.params["evaluate_net_buffer_capacity"]
        self.temp_memory_pool = {}
        # {"id":{"actor_param_input","critc_param_input","ego_pos_input","eva_pos_input","reward_input","action_input","target_output"}}


    def store_input_exp(self,pursuer_id,chosen_exp):
        # chosen_exp["target_output"]=None
        self.temp_memory_pool[pursuer_id]=dc(chosen_exp)


    def store_target(self,target_output):
        for id in self.params["pursuer_ids"]:
            self.temp_memory_pool[id]["target_output"]=torch.tensor([target_output[id]]).type(torch.float32)
            # print(self.temp_memory_pool[id]["target_output"])
            self.evaluate_net_memory_pool.append(self.temp_memory_pool[id])
        self.temp_memory_pool={}
        self.check_length()


    def check_length(self):
        while(len(self.evaluate_net_memory_pool)>self.max_capacity):
            del self.evaluate_net_memory_pool[0]


    def get_length(self):
        return len(self.evaluate_net_memory_pool)


    def get_train_batch(self,batch_size):
        if batch_size<self.get_length():
            train_set_list=random.sample(dc(self.evaluate_net_memory_pool), batch_size)

        else:
            train_set_list=dc(self.evaluate_net_memory_pool)

        param_input = dc(train_set_list[0]["param_input"])
        # critic_param_input = dc(train_set_list[0]["critic_param_input"])
        ego_pos_input = dc(train_set_list[0]["ego_pos_input"])
        eva_pos_input = dc(train_set_list[0]["eva_pos_input"])
        reward_input = dc(train_set_list[0]["reward_input"])
        action_input = dc(train_set_list[0]["action_input"])
        target_output = dc(train_set_list[0]["target_output"])

        for train_set_index, train_set in enumerate(train_set_list):
            if train_set_index != 0:
                param_input = torch.cat((param_input, dc(train_set["param_input"])), dim=0)
                # critic_param_input = torch.cat((critic_param_input, dc(train_set["critic_param_input"])), dim=0)
                ego_pos_input = torch.cat((ego_pos_input, dc(train_set["ego_pos_input"])), dim=0)
                eva_pos_input = torch.cat((eva_pos_input, dc(train_set["eva_pos_input"])), dim=0)
                reward_input = torch.cat((reward_input, dc(train_set["reward_input"])), dim=0)
                action_input = torch.cat((action_input, dc(train_set["action_input"])), dim=0)
                target_output = torch.cat((target_output, dc(train_set["target_output"])), dim=0)

        return param_input.type(torch.float32),\
               ego_pos_input.type(torch.float32),eva_pos_input.type(torch.float32),\
               reward_input.type(torch.float32),action_input.type(torch.float32),\
               target_output.type(torch.float32)














