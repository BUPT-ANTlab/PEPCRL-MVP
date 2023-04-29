#-*-coding:gb2312-*-
from Muti_Agent import Muti_agent
from env.environment import Environment
from settings import params
from copy import deepcopy as dc
import numpy as np
import csv
import os
from pathlib import Path

def run_episode(episode,agents):
    env=Environment(params)
    state=env.reset()
    pro_state, pro_all_states = agents.process_state(state)
    episode_mean_reward_dic = {}
    episode_reward_dic={}
    for pursuer_id in params["pursuer_ids"]:
        episode_reward_dic[pursuer_id]=[]
    for step in range(params["max_steps"]):
        # print("=============step",step)
        all_action,all_action_prob,pursuit_target=agents.select_action(pro_state,pro_all_states)
        next_state,done,all_reward=env.step(all_action,pursuit_target)
        if not done:
            next_pro_state, next_pro_all_states = agents.process_state(next_state)
        else:
            next_pro_state, next_pro_all_states = dc(pro_state), dc(pro_all_states)
        for pursuer_id in params["pursuer_ids"]:
            episode_reward_dic[pursuer_id].append(all_reward[pursuer_id])
        agents.replay_buffer.storage(pro_all_states,all_action,all_action_prob,all_reward,next_pro_all_states,done)
        state=dc(next_state)
        pro_state = dc(next_pro_state)
        pro_all_states = dc(next_pro_all_states)
        if done:
            break
    for pursuer_id in params["pursuer_ids"]:
        episode_mean_reward_dic[pursuer_id]=np.array(episode_reward_dic[pursuer_id]).mean()
    print("=====episode:",episode," steps:",env.steps," rewards:",episode_mean_reward_dic,"=====")
    return episode_mean_reward_dic,env.steps


def run_test(episode,agents,test_times=params["test_episodes"]):
    test_mean_reward_dic = {}
    test_reward_dic={}
    for pursuer_id in params["pursuer_ids"]:
        test_reward_dic[pursuer_id] = []
    test_steps=[]
    for test_time in range (test_times):
        env = Environment(params)
        state = env.reset()
        pro_state, pro_all_states = agents.process_state(state)
        episode_mean_reward_dic={}
        episode_reward_dic = {}
        for pursuer_id in params["pursuer_ids"]:
            episode_reward_dic[pursuer_id] = []
        for step in range(params["max_steps"]):
#            print(step)
            all_action, all_action_prob, pursuit_target = agents.select_action(pro_state,pro_all_states)
            next_state, done, all_reward = env.step(all_action, pursuit_target)
            if not done:
                next_pro_state, next_pro_all_states = agents.process_state(next_state)
            else:
                next_pro_state, next_pro_all_states=dc(pro_state),dc(pro_all_states)
            for pursuer_id in params["pursuer_ids"]:
                episode_reward_dic[pursuer_id].append(all_reward[pursuer_id])
            agents.replay_buffer.storage(pro_all_states, all_action, all_action_prob, all_reward, next_pro_all_states, done)
            state = dc(next_state)
            pro_state=dc(next_pro_state)
            pro_all_states=dc(next_pro_all_states)
            if done:
                break
        for pursuer_id in params["pursuer_ids"]:
            test_reward_dic[pursuer_id].append(np.array(episode_reward_dic[pursuer_id]).mean())
        test_steps.append(env.steps)
        for pursuer_id in params["pursuer_ids"]:
            episode_mean_reward_dic[pursuer_id] = np.array(episode_reward_dic[pursuer_id]).mean()
        print("=====episode:", episode," test_num:", test_time," steps:", env.steps, " rewards:", episode_mean_reward_dic, "=====")

    for pursuer_id in params["pursuer_ids"]:
        test_mean_reward_dic[pursuer_id]=np.array(test_reward_dic[pursuer_id]).mean()
    print("======test_episode:", episode, " steps:", np.array(test_steps).mean(), " rewards:", test_mean_reward_dic,"======")
    return test_mean_reward_dic,np.array(test_steps).mean()

def train_episode(agents):
    test_reward={}
    test_steps=0
    train_times=0
    if params["train_model"]=="evaluate":
        agents.load_evaluate_net()
    for episode in range(params["Episode"]):
        agents.load_params()
        if episode==0:
            test_reward,test_steps=run_test(episode,agents,test_times=params["warmup_episodes"])
        else:
            # agents.load_all_networks()
            agents_loss=agents.train_agents()
            test_reward,test_steps=run_test(episode,agents)
            train_times=agents.agent_list[params["pursuer_ids"][0]].train_times
            delta_rewards=calculate_delta_rewards(agents, test_reward)
            if params["train_model"] == "evaluate":
                agents.evaluate_net_buffer.store_target(delta_rewards)
                evaluate_net_loss=agents.train_evaluate_net()
                print("episode:", episode, "evaluate_net_loss:",evaluate_net_loss)
                save_loss_log(train_times, agents_loss, evaluate_net_loss)
            else:
                save_loss_log(train_times, agents_loss)
            save_test_log(train_times, test_reward, test_steps)



        agents.test_steps_num.append(test_steps)
        sorted_reward=sorted(test_reward.items(), key=lambda x: x[1])

        for id in params["pursuer_ids"]:
            agents.agent_list[id].test_reward.append(test_reward[id])
            if episode==0:
                if np.mean(agents.agent_list[id].test_reward) <= test_reward[id]:
                    agents.agent_list[id].save_param()
                elif (test_steps <= np.mean(agents.test_steps_num)) and (
                        (id in sorted_reward[0]) or (id in sorted_reward[1]) or (id in sorted_reward[2])):
                    agents.agent_list[id].save_param()

            else:
                agents.agent_list[id].loss_list.append(agents_loss[id])
                if agents_loss[id] <= np.mean(agents.agent_list[id].loss_list):
                    agents.agent_list[id].save_param()
                elif np.mean(agents.agent_list[id].test_reward) <= test_reward[id]:
                    agents.agent_list[id].save_param()
                elif (test_steps <= np.mean(agents.test_steps_num)) and (
                        (id in sorted_reward[0]) or (id in sorted_reward[1]) or (id in sorted_reward[2])):
                    agents.agent_list[id].save_param()




        if episode>=0:
            save_stable_log(train_times,agents)

    return 0


def save_test_log(episode,test_reward,test_steps):
    path='log/'+params["env_name"] +'/'+params["exp_name"]+'/test_log.csv'
    dir_path='log/'+params["env_name"] +'/'+params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row=[]
    data_row.append(episode)
    data_row.append(test_steps)
    for id in params["pursuer_ids"]:
        data_row.append(test_reward[id])
    if os.path.exists(path):
        with open(path, 'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        with open(path, 'a+',newline="") as f:
            csv_write = csv.writer(f)
            title=dc(params["pursuer_ids"])
            title.insert(0,"steps")
            title.insert(0,"train_times")
            csv_write.writerow(title)
            csv_write.writerow(data_row)
    return 0

def save_loss_log(train_times,agents_loss,evaluate_net_loss=None):
    path = 'log/' + params["env_name"] + '/' + params["exp_name"] + '/loss_log.csv'
    dir_path = 'log/' + params["env_name"] + '/' + params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row = []
    data_row.append(train_times)
    # data_row.append(test_steps)
    data_row.append(evaluate_net_loss)
    for id in params["pursuer_ids"]:
        data_row.append(agents_loss[id])
        # data_row.append(agents_loss[id]["critic_loss"])
    if os.path.exists(path):
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        title=[]
        title.append("train_times")
        title.append("evaluate_net_loss")
        for id in params["pursuer_ids"]:
            title.append(id+"_loss")
            # title.append(id + "_critic_loss")
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(title)
            csv_write.writerow(data_row)



def save_stable_log(train_times,agents):
    path = 'log/' + params["env_name"] + '/' + params["exp_name"] + '/stable_log.csv'
    dir_path = 'log/' + params["env_name"] + '/' + params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row = []
    data_row.append(train_times)
    data_row.append(np.mean(agents.test_steps_num))
    # data_row.append(evaluate_net_loss)
    for id in params["pursuer_ids"]:
        data_row.append(np.mean(agents.agent_list[id].test_reward))
        data_row.append(np.mean(agents.agent_list[id].loss_list))
        data_row.append(agents.agent_list[id].update_times)

    if os.path.exists(path):
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        title = []
        title.append("train_times")
        title.append("steps")
        for id in params["pursuer_ids"]:
            title.append(id + "_reward")
            title.append(id + "_loss")
            title.append(id + "_update")

        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(title)
            csv_write.writerow(data_row)


def calculate_delta_rewards(agents,rewards):
    delta_rewards={}
    for id in params["pursuer_ids"]:
        delta_rewards[id]=(dc(rewards[id])-dc(np.mean(agents.agent_list[id].test_reward)))*10
    return delta_rewards


agents=Muti_agent(params)
train_episode(agents)


