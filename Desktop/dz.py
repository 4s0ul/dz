import numpy as np
import re


# returns the list of nodes and their relation
def task(csv_string):
    # making adjacency list out of edge list
    graph_nodes = re.findall(r'\d+', csv_string)
    for i in range(len(graph_nodes)):
        graph_nodes[i] = int(graph_nodes[i])
    graph = {i: [] for i in range(1, max(graph_nodes) + 1)}

    csv_string = csv_string.split('\n')
    if csv_string[-1] == '':
        del csv_string[-1]

    for i in csv_string:
        nodes = i.split(',')
        graph[int(nodes[0])].append(nodes[1])

    # main task
    r1, r2, r3, r4, r5 = [], [], [], [], []

    for key in graph:
        if len(graph[key]) > 0:
            r1.append(key)
        for exit_node in graph[key]:
            if int(exit_node) not in r2:
                r2.append(int(exit_node))

    for i in r1:
        if len(graph[i]) > 1:
            for exit_node in graph[i]:
                if int(exit_node) not in r5:
                    r5.append(int(exit_node))
        for exit_node_1 in graph[i]:
            for exit_node_2 in graph[int(exit_node_1)]:
                if i not in r3:
                    r3.append(i)
                if int(exit_node_2) not in r4:
                    r4.append(int(exit_node_2))

    return [r1, r2, r3, r4, r5]


def main():
    with open('data.csv') as file:
        csv_string = file.read()
        result = task(csv_string)
        print(result)


if __name__ == '__main__':
    main()