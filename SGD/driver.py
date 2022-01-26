#!/usr/bin/env python3

#Driver skeleton borrowed from: https://github.com/HanKruiger/tsNET
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with HMDS(*).')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--iterations', '-i', type=int, default=100, help='Number of SGD iterations.')
    parser.add_argument('--scaling_factor', '-a', type=float, default=0, help='Scalar to multiply the distance matrix (d) by. If 0, the default 10/max(d) will be used')
    parser.add_argument('--optimize_scale', '-s', type=bool, default=False, help='Whether or not to optimize the output for scaling factor. If True, scaling_factor parameter is ignored.')
    parser.add_argument('--convergence', '-c', type=bool, default=False, help='Whether or not to iterate until convergence. If True, max_iter is set to max(100,iterations).')
    parser.add_argument('--output', '-o', type=str, help='Save layout to the specified file.')

    args = parser.parse_args()

    import os
    import time
    import graph_tool.all as gt
    import numpy as np
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
    d = distance_matrix(G,normalize=False)

    #Call to HMDS class and solve
    Y = HMDS(d,opt_scale=args.optimize_scale,
               scaling_factor=args.scaling_factor)
    X = Y.solve(num_iter=args.iterations,
                until_conv=args.convergence)

    #End time
    end = time.time()
    print("Completed embedding in {0} seconds".format(end-start))

    #Convert to Poincare coords
    P = np.zeros(X.shape)
    sinh, cosh = np.sinh, np.cosh
    for i in range(X.shape[0]):
        x,y = X[i]
        Rh = np.arccosh(cosh(x)*cosh(y))
        Re = (np.exp(Rh)-1)/(np.exp(Rh)+1)
        theta = 2*np.arctan( sinh(y) / ( sinh(x)*cosh(y) + np.sqrt( pow(cosh(x),2) * pow(cosh(y),2) - 1 ) ) )
        P[i] = [Re * np.cos(theta), Re * np.sin(theta)]


    #Draw to output SVG

    from drawSvg import Drawing
    from hyperbolic import euclid, util
    from hyperbolic.poincare.shapes import *
    from hyperbolic.poincare import Transform

    points = []
    lines = []
    nodeDict = {}
    d = Drawing(2.1,2.1, origin='center')
    d.draw(euclid.shapes.Circle(0, 0, 1), fill='#ddd')

    for node in P:
        print(np.linalg.norm(node))
        points.append(Point(*node))

    for e in G.edges():
        e1,e2 = e
        lines.append(Line.fromPoints(*points[int(e1)], *points[int(e2)],segment=True))


    for i in lines:
        d.draw(i,hwidth=.01,fill='black')

    for i in points:
        d.draw(i,hradius=.05,fill='green')


    d.setRenderSize(w=400)
    d.saveSvg('test.svg')
