import argparse

params = {
          "exp_name":"PEPCRL-MVP",
          "Episode": 100000,             # 总的训练轮数
          "max_steps": 800,                # 单次仿真最大步数****
          "test_episodes": 8,          # 测试时执行的轮数****
          "memory_capacity": 5,       # 经验池保存回合数/agent_num
          "train_set_epoch": 1,         # 训练集选取的回合数
          "warmup_episodes": 2,         # 热启动轮数***
          "gamma": 0.99,                # 奖励的折扣因子:DQN
          "alpha": 0.001,               #
          "epsilon": 0.1,            # 贪婪策略epsilon初始值
          "epsilon_decay": 0.0001,      # 贪婪策略epsilon衰减
          "epsilon_min": 0.01,          # 贪婪策略epsilon最小值
          # ==========================DQN=============================
          "target_update_period": 1,  # 目标网络更新周期
          "DQN_learning_rate":1e-5,
          "dqn_update_times": 5,
          "dqn_batch_size":64,
          "tau":0.005, #软更新系数
          # ==========================MADDPG==========================
          "minimax": True,
          #===========================PPO=========================
          "actor_learning_rate":1e-2,
          "critic_learning_rate":5e-2,
          "ppo_update_times":1,
          "clip_param":0.2,
          "max_grad_norm":0.5,
          #===========================env=======================
          "port":1,
          "gui": False,
          "env_name": "RealMap_6_3",
          #=====================prioritization network====================
          "train_model": "evaluate", #random or evaluate
          "evaluate_net_buffer_capacity": 100000,
          "evaluate_net_learning_rate":1e-4,
          "evaluate_net_batch_size": 64,
          "evaluate_net_update_times": 5,
          "beta0":0.01,
          "lamda":0.5

          }



if params["env_name"] == "RealMap_6_3":
    params["rou_path"] = "./env/RealMap/RealMap63.rou.xml"
    params["cfg_path"] = "./env/RealMap/RealMap63.sumocfg"
    params["net_path"] = "./env/RealMap/RealMap.net.xml"
    params["num_pursuit"] = 6
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3", "p4", "p5"]
    params["num_evader"] = 3
    params["evader_ids"] = ["e0", "e1", "e2"]
    params["num_action"] = 3
    params["lane_code_length"] = 7

    params["num_background_veh"] = 300
    params["congested_lane"] = ["-E14"]
    params["congested_prob"] = 0.2
    params["strat_warm_step"] = 50
    params["num_edge"] = 106
    params["max_steps"] = 800


elif params["env_name"] == "RealMap_7_4":  # 240辆背景车辆
    params["rou_path"] = "./env/RealMap/RealMap74.rou.xml"
    params["cfg_path"] = "./env/RealMap/RealMap74.sumocfg"
    params["net_path"] = "./env/RealMap/RealMap.net.xml"
    params["num_pursuit"] = 7
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]
    params["num_evader"] = 4
    params["evader_ids"] = ["e0", "e1", "e2", "e3"]
    params["num_action"] = 3
    params["lane_code_length"] = 7

    params["num_background_veh"] = 300
    params["congested_lane"] = ["-E14"]
    params["congested_prob"] = 0.2
    params["strat_warm_step"] = 50
    params["num_edge"] = 106
    params["max_steps"] = 800


elif params["env_name"] == "RealMap_8_5":  # 240辆背景车辆
    params["rou_path"] = "./env/RealMap/RealMap85.rou.xml"
    params["cfg_path"] = "./env/RealMap/RealMap85.sumocfg"
    params["net_path"] = "./env/RealMap/RealMap.net.xml"
    params["num_pursuit"] = 8
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
    params["num_evader"] = 5
    params["evader_ids"] = ["e0", "e1", "e2", "e3", "e4"]
    params["num_action"] = 3
    params["lane_code_length"] = 7
    params["num_background_veh"] = 300
    params["congested_lane"] = ["-E14"]
    params["congested_prob"] = 0.2
    params["strat_warm_step"] = 50
    params["num_edge"] = 106
    params["max_steps"] = 800



GLOBAL_SEED = 520
import torch
import numpy
import random
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

