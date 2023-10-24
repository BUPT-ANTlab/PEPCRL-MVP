import copy
import time
import numpy as np
import traci
from env.utils import generate_topology, get_junction_links, get_adj, get_bin
import env.utils as utils
import random
import traci
import subprocess
import sys
import logging
import heapq
import platform
from copy import deepcopy as dc

if platform.system().lower() == 'windows':
    sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui"
    sumoBinary_nogui = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo"
elif platform.system().lower() == 'linux':
    sumoBinary = "/usr/share/sumo/bin/sumo-gui"
    sumoBinary_nogui = "/usr/share/sumo/bin/sumo"


# 车辆随机选择下一节点
def random_select_next_lane(_next_lanes):
    num_list = list(range(len(_next_lanes)))
    next_lane = random.choice(num_list)
    return list(_next_lanes)[next_lane]

def random_select_next_lane_for_background(current_edge,lane_list,congested_lane_list,congested_prob):
    selected_lane_list=[]
    for lane_name_ in lane_list:
        lane_name=lane_name_.split("_")[0]
        if lane_name not in congested_lane_list and lane_name!=current_edge:
            selected_lane_list.append(lane_name)
    if random.random()<= congested_prob :
        selected_lane=random.choice(congested_lane_list)
        if selected_lane == current_edge:
            selected_lane = random.choice(selected_lane_list)
    else:
        selected_lane=random.choice(selected_lane_list)

    # print(current_edge)
    # print(lane_list)
    # print(selected_lane_list)
    return selected_lane


def generate_dict_lane_num(lane_keys):
    lane_to_num = {}
    num = 0
    for key in lane_keys:
        lane_to_num[key] = num
        num += 1
    return lane_to_num


def get_turn_lane(lane_links):
    turn_term = {"l": None,
                 "s": None,
                 "r": None}
    for i in range(len(lane_links)):
        lane_link = lane_links[i]
        edge = lane_link[0].split("_")[0]
        turn_term[lane_link[6]] = edge
    return turn_term


def get_action(current_lane, action):
    action_trans = {0: "l",
                    1: "s",
                    2: "r"}
    current_lane_links = traci.lane.getLinks(current_lane)
    turn_term = get_turn_lane(current_lane_links)
    turn_str = action_trans[action]
    turn_action = turn_term[turn_str]
    next_edge = None
    action_true = False
    if turn_action is not None:
        next_edge = turn_action
        action_true = True
    else:
        for turn_other in ["l", "s", "r"]:
            if turn_term[turn_other] is not None:
                next_edge = turn_term[turn_other]
                break
            else:
                continue
    return next_edge, action_true


class Environment:
    # 初始化环境
    def __init__(self, params):
        self.steps=0
        self.PORT = params["port"]
        self.rou_path = params["rou_path"]
        self.cfg_path = params["cfg_path"]
        self.net_path = params["net_path"]
        # print(params["net_path"])
        self.params = params
        self.topology, self.node_pos,self.topology_dict = generate_topology(net_xml_path=self.net_path)
        adj = self.topology.adj
        self.lane2num = generate_dict_lane_num(adj)
        self.get_link_array()
        # print(self.lane2num)
        # 记录所有车辆信息
        self.vehicle_list = {}
        # 记录每条路上车辆的数目
        self.lane_vehs = {}
        # 记录追逐车辆的信息
        self.pursuit_vehs = {}
        # 记录逃避车辆的信息
        self.evader_vehs = {}

        # 记录每个追逐者位置
        self.pursuer_state = {}
        # 记录追逐车辆位置
        self.evader_state = []
        #记录状态：追逐者位置，追逐车辆位置，背景车辆，邻接矩阵
        self.state={}

        self.success_evader=[]

        # 启动仿真环境
        self.sumoProcess = self.simStart()
        self.laneIDList = traci.lane.getIDList()
        self.junctionLinks, self.laneList = get_junction_links(self.laneIDList)
        self.adj = np.array(get_adj(self.topology))
        self.adj[self.adj > 0] = 1
        self.params["adj_matrix"] = self.adj
        self.vehicles = traci.vehicle.getIDList()

        # self.total_veh={}







    def simStart(self):
        
        if self.params["gui"]:
            sumoProcess = subprocess.Popen(
                [sumoBinary, "-c", self.cfg_path, "--remote-port", str(self.PORT), "--start"],
                stdout=sys.stdout, stderr=sys.stderr)
        else:
            sumoProcess = subprocess.Popen(
                [sumoBinary_nogui, "-c", self.cfg_path, "--remote-port", str(self.PORT), "--start"],
                stdout=sys.stdout, stderr=sys.stderr)
        traci.init(self.PORT)

        logging.info("start TraCI.")

        return sumoProcess

    def reset(self):
        traci.close()
        self.sumoProcess.kill()
        # ==============================重置状态信息=================================
        # 记录所有车辆信息
        self.vehicle_list = {}
        # 记录每条路上车辆的数目
        self.lane_vehs = {}
        # 记录追逐车辆的信息
        self.pursuit_vehs = {}
        # 记录逃避车辆的信息
        self.evader_vehs = {}
        self.success_evader = []

        self.state = {}
        self.sumoProcess = self.simStart()
        self.vehicles = traci.vehicle.getIDList()

        self.steps = 0
        
        for warm_step in range(0,self.params["strat_warm_step"]):
            # print("***",warm_step)
            self.step(None,None)

        # for lane_id in self.laneList:
        #     self.total_veh[lane_id]=0
        return dc(self.state)



    def step(self,command,pur_task):
        if command is not None:
            # self.AssignTask(pur_task)
            self.pursuitVehControl(choice_random=False, commands=command)
            self.evadeVehControl(choice_random=True)
            traci.simulationStep()
            self.steps = self.steps + 1
            # print("**************")

        else:
            self.pursuitVehControl(choice_random=True)
            self.evadeVehControl(choice_random=True)
            # print(len(traci.vehicle.getIDList()))
            traci.simulationStep()
            self.vehicles = traci.vehicle.getIDList()
            # print("**************")
            # print(len(self.vehicles))
        self.vehicles = traci.vehicle.getIDList()
        # print(self.vehicles)

        for vehicle in self.vehicles:
            self.vehicle_list[vehicle] = {"routeLast": traci.vehicle.getRoute(vehicle)[-1]}
            if "p" in vehicle:
                p_x, p_y = traci.vehicle.getPosition(vehicle)
                p_lane = traci.vehicle.getLaneID(vehicle)
                p_lane = self.checkLane(p_lane)
                next_lane_links = traci.lane.getLinks(p_lane)
                p_turn_term = get_turn_lane(next_lane_links)
                p_lane_position = traci.vehicle.getLanePosition(vehicle)
                p_target = traci.vehicle.getRoute(vehicle)[-1]
                if vehicle not in self.pursuit_vehs.keys():
                    self.pursuit_vehs[vehicle] = {"x": p_x,
                                                  "y": p_y,
                                                  "p_lane": p_lane,
                                                  "p_edge": p_lane.split("_")[0],
                                                  "p_lane_left": p_turn_term["l"],
                                                  "p_lane_straight": p_turn_term["s"],
                                                  "p_lane_right": p_turn_term["r"],
                                                  "p_lane_position": p_lane_position,
                                                  "p_target": p_target,
                                                  "change_target_evader":False,
                                                  "target_evader": None,
                                                  "target_evader_dis": 100,
                                                  "target_evader_dis_last": 100,
                                                  "num_capture": 0}
                else:
                    if self.steps==0:
                        target_evader = None
                        if_change_traget =False
                        target_evader_dis = self.pursuit_vehs[vehicle]["target_evader_dis"]
                        target_evader_dis_last = self.pursuit_vehs[vehicle]["target_evader_dis_last"]
                    else:
                        if_change_traget= not (pur_task[vehicle]==self.pursuit_vehs[vehicle]["target_evader"])
                        target_evader=pur_task[vehicle]
                        target_evader_dis_last =self.pursuit_vehs[vehicle]["target_evader_dis"]
                        target_evader_dis=utils.calculate_dis(p_x, p_y,
                                              self.evader_vehs[target_evader]["x"], self.evader_vehs[target_evader]["y"])
                    num_capture = self.pursuit_vehs[vehicle]["num_capture"]
                    self.pursuit_vehs[vehicle] = {"x": p_x,
                                                  "y": p_y,
                                                  "p_lane": p_lane,
                                                  "p_edge": p_lane.split("_")[0],
                                                  "p_lane_left": p_turn_term["l"],
                                                  "p_lane_straight": p_turn_term["s"],
                                                  "p_lane_right": p_turn_term["r"],
                                                  "p_lane_position": p_lane_position,
                                                  "p_target": p_target,
                                                  "target_evader": target_evader,
                                                  "change_target_evader": if_change_traget,
                                                  "target_evader_dis": target_evader_dis,
                                                  "target_evader_dis_last": target_evader_dis_last,
                                                  "num_capture": num_capture}

            if "e" in vehicle:
                e_x, e_y = traci.vehicle.getPosition(vehicle)
                e_lane = traci.vehicle.getLaneID(vehicle)
                e_lane = self.checkLane(e_lane)
                next_lane_links = traci.lane.getLinks(e_lane)
                e_turn_term = get_turn_lane(next_lane_links)
                e_lane_position = traci.vehicle.getLanePosition(vehicle)
                e_target = traci.vehicle.getRoute(vehicle)[-1]
                self.evader_vehs[vehicle] = {"x": e_x,
                                             "y": e_y,
                                             "e_lane": e_lane,
                                             "e_edge": e_lane.split("_")[0],
                                             "e_lane_left": e_turn_term["l"],
                                             "e_lane_straight": e_turn_term["s"],
                                             "e_lane_right": e_turn_term["r"],
                                             "e_lane_position": e_lane_position,
                                             "e_target": e_target}







        # ==============================统计车流信息，更新背景车辆路径=====================================
        if len(self.vehicles) > 0:
            # =============================统计每条车道上的车辆数目=======================================
            for lane_i in range(len(self.laneList)):
                self.lane_vehs[self.laneList[lane_i]] = 0

            for id_num in range(len(self.vehicles)):
                # ===============================为背景车辆重新规划路径===================================
                if "Background" in self.vehicles[id_num]:
                    current_edge = traci.vehicle.getLaneID(self.vehicles[id_num]).split("_")[0]
                    route_last_edge = traci.vehicle.getRoute(self.vehicles[id_num])[-1]
                    if current_edge == route_last_edge:
                        next_edge_target = random_select_next_lane_for_background(current_edge, self.laneList,
                                                                                  self.params["congested_lane"],
                                                                                  self.params["congested_prob"])
                        traci.vehicle.changeTarget(self.vehicles[id_num], next_edge_target)
                        # next_edges = self.topology.out_edges(current_edge)
                        # next_edge_target = random_select_next_lane(next_edges)
                        # route_list = list(next_edge_target)
                        # traci.vehicle.setRoute(self.vehicles[id_num], route_list)
                # print("**************")
                # =================================计算车流量===========================================
                    current_lane = traci.vehicle.getLaneID(self.vehicles[id_num])
                    if current_lane in self.laneList:
                        self.lane_vehs[current_lane] += 1
                    else:
                        self.lane_vehs[self.junctionLinks[current_lane]] += 1
            self.update_congested_prob()




            if command is None:
                self.generateState()
                return dc(self.state),False, 0

            else:
                if_stop = (self.checkPursuit() or self.steps>=self.params["max_steps"])
                rewards = self.calculateReward()
                self.generateState()
                # self.get_total_veh()
                if if_stop:
                    traci.close()
                    self.sumoProcess.kill()
                return dc(self.state),if_stop, rewards


    # def get_total_veh(self):
    #     for lane_id in self.laneList:
    #         self.total_veh[lane_id] =  self.total_veh[lane_id]+self.lane_vehs[lane_id]

    # ==========================================检查逃避车辆是否被追到======================================
    def checkPursuit(self):

        remove_list = []
        for evader_id in self.evader_vehs.keys():
            e_x, e_y = self.evader_vehs[evader_id]["x"], self.evader_vehs[evader_id]["y"]
            for pursuit_id in self.pursuit_vehs.keys():
                p_x, p_y = self.pursuit_vehs[pursuit_id]["x"], self.pursuit_vehs[pursuit_id]["y"]
                dis_p_e = utils.calculate_dis(e_x, e_y, p_x, p_y)
                if dis_p_e < 5:
                    if evader_id not in remove_list:
                        traci.vehicle.remove(evader_id)
                        remove_list.append(evader_id)
                        self.success_evader.append(evader_id)
                    else:
                        print("%s had been removed!" % evader_id)

                    self.pursuit_vehs[pursuit_id]["num_capture"] += 1
        if len(remove_list) > 0:
            for rm_id in remove_list:
                print("remove: %s" % rm_id)
                try:
                    if rm_id in self.vehicle_list:
                        del self.vehicle_list[rm_id]
                    else:
                        print("%s had been removed!" % rm_id)
                    if rm_id in self.evader_vehs:
                        del self.evader_vehs[rm_id]
                    else:
                        print("%s had been removed!" % rm_id)
                except:
                    pass
                finally:
                    pass
            self.vehicles = traci.vehicle.getIDList()
            # ======================================判断终止条件========================================
            print(len(self.evader_vehs))
            if len(self.evader_vehs) == 0:
                return True
        return False





    def update_congested_prob(self):
        congested_lane_vehicle = []
        for congested_lane in self.params["congested_lane"]:
            congested_lane_vehicle.append(self.lane_vehs[congested_lane + '_0'])

        if np.mean(congested_lane_vehicle) <= 5:
            self.params["congested_prob"] = 0.05*len(congested_lane_vehicle)
        elif np.mean(congested_lane_vehicle) > 5 and np.mean(congested_lane_vehicle) <= 10:
            self.params["congested_prob"] = 0.02*len(congested_lane_vehicle)
        elif np.mean(congested_lane_vehicle) > 10 and np.mean(congested_lane_vehicle) <= 15:
            self.params["congested_prob"] = 0.012
        elif np.min(congested_lane_vehicle) > 15 and np.mean(congested_lane_vehicle) <= 25:
            self.params["congested_prob"] = 0.009
        if np.max(congested_lane_vehicle) > 25:
            self.params["congested_prob"] = 0



    def generateState(self):
        # 记录每个追逐着信息
        self.pursuer_state = {}
        self.evader_state = {}
        self.pursuer_x_y={}
        self.evader_x_y={}
        self.state = {}


        #记录车流量信息
        self.background_veh=[]
        veh_num = []
        for lane_id in self.lane_vehs.keys():
            veh_num.append(self.lane_vehs[lane_id])

        self.background_veh = veh_num
        #记录追逐车辆与逃跑车辆位置信息:车道编码与位置坐标
        #position=[0,0,1,1,0.3]
        for pursuit_id in list(self.pursuit_vehs.keys()):
            x_y={}
            x_y["x"]=self.pursuit_vehs[pursuit_id]["x"]
            x_y["y"] = self.pursuit_vehs[pursuit_id]["y"]
            lane = self.pursuit_vehs[pursuit_id]["p_edge"]
            lane_id=self.lane2num[lane]
            lane_bin_code=get_bin(lane_id,self.params["lane_code_length"])
            position=lane_bin_code+[self.pursuit_vehs[pursuit_id]["p_lane_position"]/self.topology_dict[lane]["length"]]
            self.pursuer_state[pursuit_id]=position
            self.pursuer_x_y[pursuit_id]=x_y

            # print(lane)
        for evader_id in list(self.evader_vehs.keys()):
            x_y = {}
            x_y["x"] = self.evader_vehs[evader_id]["x"]
            x_y["y"] = self.evader_vehs[evader_id]["y"]
            lane = self.evader_vehs[evader_id]["e_edge"]
            lane_id=self.lane2num[lane]
            lane_bin_code=get_bin(lane_id,self.params["lane_code_length"])
            position=lane_bin_code+[self.evader_vehs[evader_id]["e_lane_position"]/self.topology_dict[lane]["length"]]
            self.evader_state[evader_id]=position
            self.evader_x_y[evader_id] = x_y
        for evader_id in self.success_evader:
            position=[-1]*self.params["lane_code_length"]+[0]
            self.evader_state[evader_id] = position
            x_y = {}
            x_y["x"] = -1
            x_y["y"] = -1
            self.evader_x_y[evader_id] = x_y


        self.state["pursuer_pos"]=dc(self.pursuer_state)
        self.state["evader_pos"]=dc(self.evader_state)
        self.state["background_veh"]=dc(self.background_veh)
        self.state["topology_array"]=dc(self. link_array)
        self.state["pursuer_xy"] = dc(self.pursuer_x_y)
        self.state["evader_xy"] = dc(self.evader_x_y)
        self.state["steps"]=dc(self.steps)


        # print("***********************************************")


    def get_link_array(self):
        self.link_array = np.zeros((len(self.lane2num.keys()),len(self.lane2num.keys())))
        for from_lane_name in list(self.lane2num.keys()):
            from_jun=self.topology_dict[from_lane_name]["to"]
            for to_lane_name in list(self.lane2num.keys()):
                if from_jun==self.topology_dict[to_lane_name]["from"]:
                    self.link_array[self.lane2num[from_lane_name]][self.lane2num[to_lane_name]]=1


    def pursuitVehControl(self, choice_random=False, commands=None):
        if choice_random:
            for pursuit_id in self.pursuit_vehs.keys():
                # ===============================追逐车辆随机选择目的地===================================
                current_edge = traci.vehicle.getLaneID(pursuit_id).split("_")[0]
                route_last_edge = traci.vehicle.getRoute(pursuit_id)[-1]
                if current_edge == route_last_edge:
                    next_edges = self.topology.out_edges(current_edge)
                    next_edge_target = random_select_next_lane(next_edges)
                    route_list = list(next_edge_target)
                    traci.vehicle.setRoute(pursuit_id, route_list)
        else:
            # assert commands.shape == (self.params["num_pursuit"], )
            for _i, pur_veh in enumerate(self.params["pursuer_ids"]):
                if pur_veh in self.pursuit_vehs.keys():
                    current_lane = self.checkLane(traci.vehicle.getLaneID(pur_veh))
                    action_next_lane, action_true = get_action(current_lane, commands[pur_veh])
                    route_list = [current_lane.split("_")[0], action_next_lane]
                    traci.vehicle.setRoute(pur_veh, route_list)
                else:
                    continue

    def evadeVehControl(self, choice_random=False):
        if choice_random:
            for evader_id in self.evader_vehs.keys():
                # ===============================追逐车辆随机选择目的地===================================
                current_edge = traci.vehicle.getLaneID(evader_id).split("_")[0]
                route_last_edge = traci.vehicle.getRoute(evader_id)[-1]
                if current_edge == route_last_edge:
                    next_edges = self.topology.out_edges(current_edge)
                    next_edge_target = random_select_next_lane(next_edges)
                    route_list = list(next_edge_target)
                    traci.vehicle.setRoute(evader_id, route_list)

    def checkLane(self, lane):
        if "J" in lane:
            next_lane = self.junctionLinks[lane]

            return next_lane
        else:
            return lane

    def calculateReward(self):
        inter_dis = 10
        rewards = {}
        # 时间步损失
        for pursuit_id in self.pursuit_vehs.keys():
            reward=0

            reward = reward-0.02*self.steps
            reward += self.pursuit_vehs[pursuit_id]["num_capture"]*400
            self.pursuit_vehs[pursuit_id]["num_capture"] = 0

            if self.pursuit_vehs[pursuit_id]["change_target_evader"]:
                reward += 5*(self.pursuit_vehs[pursuit_id]["target_evader_dis_last"] - self.pursuit_vehs[pursuit_id]["target_evader_dis"])
            else:
                reward += 5 * (
                            self.pursuit_vehs[pursuit_id]["target_evader_dis_last"] - self.pursuit_vehs[pursuit_id][
                        "target_evader_dis"])
            # reward += (inter_dis - self.pursuit_vehs[pursuit_id]["target_evader_dis"])/(inter_dis/2)
            rewards[pursuit_id]=reward
        return rewards








