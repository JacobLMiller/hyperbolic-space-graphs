#!/usr/bin/env python3

#Driver skeleton borrowed from: https://github.com/HanKruiger/tsNET
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with HMDS(*).')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--iterations', '-i', type=int, default=15, help='Number of SGD iterations.')
    parser.add_argument('--scaling_factor', '-a', type=float, default=0, help='Scalar to multiply the distance matrix (d) by. If 0, the default 10/max(d) will be used')
    parser.add_argument('--optimize_scale', '-s', type=bool, default=False, help='Whether or not to optimize the output for scaling factor. If True, scaling_factor parameter is ignored.')
    parser.add_argument('--convergence', '-c', type=bool, default=False, help='Whether or not to iterate until convergence. If True, max_iter is set to max(100,iterations).')
    parser.add_argument('--output', '-o', type=str, help='Save layout to the specified file.')

    args = parser.parse_args()

    import os
    import time
    import graph_tool.all as gt
    from graph_processing import distance_matrix
    from HMDS import HMDS, preprocess, postprocess

    # Check for valid input
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]

    #Load graph data
    G = gt.load_graph(args.input_graph)
    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, G.num_vertices(), G.num_edges()))

    #Record time for fun
    start = time.time()

    #Compute shortest path distance matrix to use as input
    d = distance_matrix(G)

    #Call to HMDS class and solve


    #End time
    end = time.time()

    #Convert to Poincare coords


    #Draw to output SVG 
