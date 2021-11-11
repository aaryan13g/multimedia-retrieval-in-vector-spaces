import copy
import math
import sys
import numpy as np


class Node:
    def __init__(self, id):
        self.id = id
        self.children = []

    def print_node(self):
        print("Node: ", self.id)
        print("Children: ", [(child[0].id, child[1]) for child in self.children])


def create_sim_graph(sub_sub_similarity_matrix, n):
    graph = []
    for i in range(len(sub_sub_similarity_matrix)):
        temp = sub_sub_similarity_matrix[i]
        rank = np.argpartition(temp, -(n + 1))[-(n + 1):]
        rank = rank[np.argsort(temp[rank])]
        for j in range(len(rank) - 1):
            graph.append([i + 1, rank[j] + 1, {'weight': sub_sub_similarity_matrix[i][rank[j]]}])

    return np.array(graph)


def convert_graph_to_nodes(graph, len_total_nodes):
    child_dict = {}
    for i in range(len_total_nodes):
        child_dict[i + 1] = []
    for node_pair in graph:
        parent = node_pair[0]
        child = node_pair[1]
        weight = node_pair[2]
        child_dict[parent].append([child, weight])
    nodes = []
    for i in range(len_total_nodes):
        id = i + 1
        node = Node(id)
        nodes.append(node)
    for parent in child_dict:
        for child in child_dict[parent]:
            nodes[parent-1].children.append([nodes[child[0]-1], {'weight': child[1]['weight']}])
    return nodes


def _is_converge(sim, sim_old, len_total_nodes, eps=1e-4):
    for i in range(len_total_nodes):
        for j in range(len_total_nodes):
            if abs(sim[i, j] - sim_old[i, j]) >= eps:
                return False
    return True


def ascos_plus_plus(nodes, c=0.9, max_iter=100):
    sim = np.eye(len(nodes))
    sim_old = np.zeros(shape=(len(nodes), len(nodes)))
    for iter_ctr in range(max_iter):
        if _is_converge(sim, sim_old, len(nodes)):
            break
        sim_old = copy.deepcopy(sim)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                s_ij = 0.0
                w_i = 0
                for child in nodes[i].children:
                    if j == int(child[0].id) - 1:
                        w_ik = child[1]['weight']
                        w_i += w_ik
                    else:
                        w_ik = 0
                    s_ij += float(w_ik) * (1 - math.exp(- w_ik)) * sim_old[int(child[0].id) - 1, j]
                sim[i, j] = c * s_ij / w_i if w_i > 0 else 0
    return sim


def get_sys_args():
    return str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    subject_subject_similarity_matrix, m, n = get_sys_args()
    subject_subject_similarity_matrix = np.loadtxt("../Outputs/" + subject_subject_similarity_matrix + ".csv", delimiter=',')
    graph = create_sim_graph(subject_subject_similarity_matrix, n)
    nodes = convert_graph_to_nodes(graph, len(subject_subject_similarity_matrix))
    # for node in nodes:
    #     node.print_node()
    sim = ascos_plus_plus(nodes, 0.9, 100)
    score_dict = {}
    i = 1
    for row in sim:
        score_dict[str(i)] = np.sum(row) - 1
        i = i + 1
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
    print("\nMost significant ", m, " subjects: \n")
    for subject in score_dict:
        if m == 0:
            break
        print("Subject ", subject, " : \t ASCOS++ Measure: ", score_dict[subject])
        m = m - 1
