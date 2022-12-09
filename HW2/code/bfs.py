import csv
edgeFile = 'edges.csv'

def bfs(start, end):
    # Begin your code (Part 1)
    """
    1. load the csv file into rows
    2. store them into the dictionary edges in the format:  
        {startID : list of attached (endID, distance)}
    3. implement BFS
    4. backtrack the path by using the list trace
    5. return path, dist, num_visited
    """
    # raise NotImplementedError("To be implemented")
    edges = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        headers = next(rows)
        # edges: {startID : list of attached (endID, distance) s}
        for row in rows:
            if edges.get(int(row[0])) != None:
                edges[int(row[0])].append((int(row[1]), float(row[2])))
            else: edges[int(row[0])] = [(int(row[1]), float(row[2]))]   

    queue = [(start, 0, 0)]
    explored = [start]
    trace = {}
    num_visited = 0
    while (True):
        flag = True
        if queue[0][0] in edges.keys():  # check if it is a leaf node
            for ID, distance in edges[queue[0][0]]: 
                if ID in explored: continue  # check if the node is explored
                if ID == end: 
                    trace[end] = (queue[0][0], distance) 
                    num_visited += 1
                    flag = False
                    break
                queue.append((ID, distance, queue[0][0]))
                explored.append(ID)
                num_visited += 1
        trace[queue[0][0]] = (queue[0][2], queue[0][1])
        if flag == False: break
        queue.pop(0)

    dist = 0
    path = [end]
    dist += trace[path[0]][1]
    while trace[path[0]][0] != 0:
        path.insert(0, trace[path[0]][0])
        dist += trace[path[0]][1]

    return path, dist, num_visited
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
