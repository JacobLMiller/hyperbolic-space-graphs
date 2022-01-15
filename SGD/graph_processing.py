import numpy as np

def get_distance_matrix_nx(graphfile):
    import networkx as nx
    """
    Trivial approach (bfs from every node) implemented in vanilla Python and networkx
    Can be quite slow for large graphs, recommend to use the graph_tool implementation
    for large and dense graphs.
    """

    def bfs(G,start):
        queue = [start]
        discovered = [start]
        distance = {start: 0}

        while len(queue) > 0:
            v = queue.pop()

            for w in G.neighbors(v):
                if w not in discovered:
                    discovered.append(w)
                    distance[w] =  distance[v] + 1
                    queue.insert(0,w)

        myList = []
        for x in G.nodes:
            if x in distance:
                myList.append(distance[x])
            else:
                myList.append(-1)

        return myList

    def all_pairs_shortest_path(G):
        d = [ [ -1 for i in range(len(G.nodes)) ] for j in range(len(G.nodes)) ]

        count = 0
        for node in G.nodes:
            d[count] = bfs(G,node)
            count += 1
        return d

    return all_pairs_shortest_path(nx.drawing.nx_agraph.read_dot(graphfile))


def get_distance_matrix_gt(graphfile,fmt='dot'):
    import graph_tool.all as gt

    #Code taken from **link t-SNET Github
    def get_shortest_path_distance_matrix(g, k=10, weights=None):
        # Used to find which vertices are not connected. This has to be this weird,
        # since graph_tool uses maxint for the shortest path distance between
        # unconnected vertices.
        def get_unconnected_distance():
            g_mock = gt.Graph()
            g_mock.add_vertex(2)
            shortest_distances_mock = gt.shortest_distance(g_mock)
            unconnected_dist = shortest_distances_mock[0][1]
            return unconnected_dist

        # Get the value (usually maxint) that graph_tool uses for distances between
        # unconnected vertices.
        unconnected_dist = get_unconnected_distance()

        # Get shortest distances for all pairs of vertices in a NumPy array.
        X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

        if len(X[X == unconnected_dist]) > 0:
            print('[distance_matrix] There were disconnected components!')

        # Get maximum shortest-path distance (ignoring maxint)
        X_max = X[X != unconnected_dist].max()

        # Set the unconnected distances to k times the maximum of the other
        # distances.
        X[X == unconnected_dist] = k * X_max

        return X


    # Return the distance matrix of g, with the specified metric.
    def distance_matrix(g, distance_metric='shortest_path', normalize=False, k=10.0, verbose=True, weights=None):
        if verbose:
            print('[distance_matrix] Computing distance matrix (metric: {0})'.format(distance_metric))

        if distance_metric == 'shortest_path' or distance_metric == 'spdm':
            X = get_shortest_path_distance_matrix(g, weights=weights)
        elif distance_metric == 'modified_adjacency' or distance_metric == 'mam':
            X = get_modified_adjacency_matrix(g, k)
        else:
            raise Exception('Unknown distance metric.')

        # Just to make sure, symmetrize the matrix.
        X = (X + X.T) / 2

        # Force diagonal to zero
        X[range(X.shape[0]), range(X.shape[1])] = 0

        # Normalize matrix s.t. max is 1.
        if normalize:
            X /= np.max(X)
        if verbose:
            print('[distance_matrix] Done!')

        return X

    return distance_matrix(gt.load_graph(graphfile,fmt=fmt))


def get_distance_matrix(graphfile,fmt='dot',lib='gt'):

    if lib == 'gt':
        return get_distance_matrix_gt(graphfile,fmt=fmt)
    elif lib == 'nx':
        assert(fmt == 'dot')
        return get_distance_matrix_nx(graphfile)
    else:
        raise Exception("Unknown library")
