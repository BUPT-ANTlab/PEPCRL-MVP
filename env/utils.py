import networkx as nx
import xml.dom.minidom
import traci
import numpy as np
import matplotlib.pyplot as plt


def calculate_dis(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


def generate_topology(net_xml_path):
    topology_dict = {}
    DG = nx.DiGraph()
    pos = {}
    dom = xml.dom.minidom.parse(net_xml_path)

    root = dom.documentElement

    content = root.getElementsByTagName('edge')

    for i in range(len(content)):
        content_detail = content[i]
        # print("attribute id")
        # print(content_detail.getAttribute('id'))
        # 处理Edge，略过Junction
        if 'J' not in content_detail.getAttribute('id'):
            id_content = content_detail.getAttribute('id')

            from_content = content_detail.getAttribute('from')
            # print(from_content)
            to_content = content_detail.getAttribute('to')
            # print(to_content)
            in_content = content_detail.getElementsByTagName('lane')
            length_edge = in_content[0].getAttribute('length')
            topology_dict[id_content] = {
                "from": from_content,
                "to": to_content,
                "length": float(length_edge)
            }
            shape_content = in_content[0].getAttribute('shape')
            first_node, second_node = shape_content.split(" ")[0], shape_content.split(" ")[1]
            first_x, first_y = first_node.split(",")[0], first_node.split(",")[1]
            second_x, second_y = second_node.split(",")[0], second_node.split(",")[1]
            pos[id_content] = ((float(first_x)+float(second_x))/2,
                               (float(first_y)+float(second_y))/2)

    keys = topology_dict.keys()
    for from_id in keys:
        for to_id in keys:
            if from_id != to_id and topology_dict[from_id]["to"] == topology_dict[to_id]["from"]:
                # 防止自环，即无调头的操作
                if topology_dict[from_id]["from"] == topology_dict[to_id]["to"]:
                    pass
                else:
                    DG.add_weighted_edges_from([(from_id, to_id, topology_dict[from_id]["length"])])
    return DG, pos,topology_dict


def get_junction_links(_laneIDList):
    junction_links = {}
    lane_list = []
    for i in range(len(_laneIDList)):
        if "J" in _laneIDList[i]:
            junction_links[_laneIDList[i]] = traci.lane.getLinks(_laneIDList[i])[0][0]
        else:
            lane_list.append(_laneIDList[i])
    return junction_links, lane_list


def get_adj(graph):
    return nx.adjacency_matrix(graph).todense()


def get_bin(num, length):
    result = [0 for _ in range(length)]
    str_bin = bin(num).replace('0b', '')
    str_len = len(str_bin)
    for i, num_ in enumerate(str_bin):
        if num_ == "1":
            result[length - (str_len - i)] = 1
    return result


if __name__ == '__main__':
    a = get_bin(30, 6)
