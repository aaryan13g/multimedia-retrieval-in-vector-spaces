import sys
import os
import numpy as np
import pymongo


class Node:
    def __init__(self, id, subject_len):
        self.id = id
        self.children = []
        self.parents = []
        self.pagerank = 1.0 / subject_len
        self.differ = 1.0

    def update_pagerank(self, beta, query_subjects):
        in_neighbors = self.parents
        if len(in_neighbors) != 0:
            pagerank_sum = sum((node.pagerank / len(node.children)) for node in in_neighbors)
        else:
            pagerank_sum = 0
        if self.id in query_subjects:
            teleport = beta / len(query_subjects)
        else:
            teleport = 0
        temp = self.pagerank
        self.pagerank = teleport + (1 - beta) * pagerank_sum
        self.differ = abs(self.pagerank - temp)
        return self.differ

    def print_node(self):
        print("Node: ", self.id)
        print("Children: ", [child.id for child in self.children])
        print("Parents: ", [parent.id for parent in self.parents])
        print("Page Rank: ", self.pagerank)


def pagerank_one_iter(node_list, beta, query_subjects):
    converge = True
    for node in node_list:
        differ = node.update_pagerank(beta, query_subjects)
        if differ > 0.00001:
            converge = False
    if converge is False:
        return False
    else:
        return True


def create_sim_graph(sub_sub_similarity_matrix, n):
    graph = []
    for i in range(len(sub_sub_similarity_matrix)):
        temp = sub_sub_similarity_matrix[i]
        rank = np.argpartition(temp, -(n + 1))[-(n + 1):]
        rank = rank[np.argsort(temp[rank])]
        for j in range(len(rank) - 1):
            graph.append([i + 1, rank[j] + 1])

    return np.array(graph)


def convert_graph_to_nodes(graph, len_total_nodes):
    parent_dict = {}
    child_dict = {}
    for i in range(len_total_nodes):
        parent_dict[i + 1] = []
        child_dict[i + 1] = []
    for node_pair in graph:
        parent = node_pair[0]
        child = node_pair[1]
        child_dict[parent].append(child)
        parent_dict[child].append(parent)
    nodes = []
    for i in range(len_total_nodes):
        id = i + 1
        node = Node(id, len_total_nodes)
        nodes.append(node)
    for parent in child_dict:
        for child in child_dict[parent]:
            nodes[parent-1].children.append(nodes[child-1])
    for child in parent_dict:
        for parent in parent_dict[child]:
            nodes[child-1].parents.append(nodes[parent-1])
    return nodes


def print_matching_subjects(m, nodes):
    match_dict = {}
    for node in nodes:
        match_dict[node.id] = node.pagerank
    match_dict = dict(sorted(match_dict.items(), key=lambda item: item[1], reverse=True))
    print("Most similar ", m, " subjects relative to given inputs are: ")
    for match in match_dict:
        if m == 0:
            break
        print(match, " PageRank: ", match_dict[match])
        m = m - 1


def get_sys_args():
    return str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    subject_subject_similarity_matrix, m, n, subject1, subject2, subject3 = get_sys_args()
    subject_subject_similarity_matrix = np.loadtxt("../Outputs/" + subject_subject_similarity_matrix + ".csv", delimiter=',')
    graph = create_sim_graph(subject_subject_similarity_matrix, n)
    nodes = convert_graph_to_nodes(graph, len(subject_subject_similarity_matrix))
    query_subjects = [subject1, subject2, subject3]
    i = 0
    while i < 100:
        converge = pagerank_one_iter(nodes, 0.15, query_subjects)
        if converge:
            break
        i = i + 1
    print("PageRank converged in ", i, " iterations.")
    for node in nodes:
        node.print_node()
    # sum = 0
    # for node in nodes:
    #     sum = sum + node.pagerank
    # print(sum)
    print_matching_subjects(m, nodes)
