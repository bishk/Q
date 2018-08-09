import numpy as np
test_graphs = [np.load("./graphs/graph000005.npz")]

print(test_graphs[0]["y"])
lines = []
b =   [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for i in range(len(test_graphs)):
        graph = dict()
        lines = []
        for j in range(len(test_graphs[i]["y"])):
            if b[j]>0.5:
                i1 = np.where(test_graphs[i]["Ri_cols"] == j)[0][0]
                i2 = np.where(test_graphs[i]["Ro_cols"] == j)[0][0]
                index1 = test_graphs[i]["Ri_rows"][i1]
                index2 = test_graphs[i]["Ro_rows"][i2]
                if index1 in graph:
                    graph[index1].append(index2)
                    if index2 in graph:
                        graph[index2].append(index1)
                    else:
                        graph[index2] = [index1]
                else:
                    graph[index1] = [index2]
                    if index2 in graph:
                        graph[index2].append(index1)
                    else:
                        graph[index2] = [index1]
                x1, y1, _ = test_graphs[i]["X"][index1]
                x2, y2, _ = test_graphs[i]["X"][index2]
                lines.append([(x1, y1), (x2, y2)])
        #print(getRoots(graph))        
        print(lines)