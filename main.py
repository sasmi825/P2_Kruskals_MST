import random
import time
import numpy as np
import networkx as nx
import math
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd


# Disjoint Set (Union-Find) data structure
class DisjointSet:
    def __init__(self, size):
        # Initialize the parent array where each element is initially its own parent
        self.parent = [i for i in range(size)]

    def find(self, node):
        # Find the root (representative) of the set that 'node' belongs to
        if self.parent[node] != node:
            # Checks if 'node' is not its own parent, recursively find the root
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, u, v):
        # Union operation merge two sets ('u' and 'v')
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # If 'u' and 'v' belong to different sets, make one's root the parent of the other
            self.parent[root_u] = root_v


def apply_kruskals_algorithm(graph):
    """ Kruskal's algorithm for Minimum Spanning Tree """
    # gather all unique edges and weights
    edges = []  # will append tuple with following format: (i, j, w)
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    # sort edges by weight
    edges.sort(key=lambda x: x[2])

    num_nodes = len(graph)
    # Initialize the minimum spanning tree
    mst = []
    # initializing the disjoint set
    ds = DisjointSet(num_nodes)

    for edge in edges:
        u, v, weight = edge
        # checking if this edge in mst does not form a cycle
        if ds.find(u) != ds.find(v):
            # checks if adding this edge does not create a cycle in the MST, add it to the MST
            mst.append(edge)
            ds.union(u, v)

    return mst


def generate_random_graph(n):
    """Function to generate a random graph with 'n' nodes """
    # generate a random graph with n nodes
    # probability an edge gets created bt two nodes
    p = random.uniform(0.5, 1)

    # create a graph
    G = nx.erdos_renyi_graph(n, p)
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    # array representation of graph
    arr = nx.to_numpy_array(G)
    # assign random weights between 1 to 100 to edges
    weights_generalization = np.random.randint(1, high=100, size=arr.shape, dtype=int)
    arr = np.multiply(arr, weights_generalization)

    return arr, nodes, edges


def generate_plot(df1, df2):
    # plotting a 3d graph for m,n and time_taken
    ax = plt.axes(projection='3d')

    # plotting experimental
    # note: Plotting the logarithmic m_vals and time taken
    ax.plot(df1.n_vals.tolist(),
            df1.m_val.apply(lambda x: math.log(x)).tolist(),
            df1.y_val_scaled.apply(lambda x: math.log(x)).tolist(),
            'green', label="experimental")
    # plotting theoretical
    # note: Plotting the logarithmic m_vals and time taken
    ax.plot(df2.n_vals.tolist(),
            df2.m_val.apply(lambda x: math.log(x)).tolist(),
            df2.y_val.apply(lambda x: math.log(x)).tolist(),
            'red', label="theoretical")
    ax.set_xlabel('n_vals')
    ax.set_ylabel('log(m_val)')
    ax.set_zlabel('exec_time')
    ax.set_title('Kruskals MST - Experimental vs Theoretical')
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.legend()
    plt.show()
    return ax


def main(n_values: list):
    # Test Kruskal's algorithm for different 'n' values
    # n_values = [10, 100, 1000, 10000]
    edge_count = []
    time_taken = []
    for n in n_values:
        graph, nv, ev = generate_random_graph(n)
        edge_count.append(ev)

        start_time = time.time()
        mst = apply_kruskals_algorithm(graph)
        end_time = time.time()

        execution_time = end_time - start_time
        time_taken.append(execution_time)

        print(f"\nnodes = {n}\nedges = {ev}\nExecution time: {execution_time:.6f}\n")
        print("-" * 16)
    print('nvals: ', n_values)
    print('time_take: ', time_taken)

    # Generating the dataframe from Experimental data
    df1 = pd.DataFrame.from_dict({'n_vals': n_values, 'y_val': time_taken, 'm_val': edge_count})
    df1 = df1.assign(hue=['experimental'] * len(df1))

    # Generating the dataframe from theoretical data
    df2 = pd.DataFrame.from_dict({'n_vals': n_values, 'm_val': edge_count})
    # Calculating time taken theoretically and assigning it to y_val
    df2 = df2.assign(y_val=df2.m_val * df2.n_vals.apply(lambda x: math.log(x)))
    df2 = df2.assign(hue=['theoretical'] * len(df2))

    # Determining the scale factor and applying it to the y_val
    dft = df1.merge(df2, on='n_vals', how='left')
    scaling_factor = (dft.y_val_y / dft.y_val_x).mean()
    df1 = df1.assign(y_val_scaled=df1.y_val * scaling_factor)

    ax = generate_plot(df1, df2)

    return 0


if __name__ == "__main__":
    n_values = [10, 100, 1000, 10000]
    main(n_values)
