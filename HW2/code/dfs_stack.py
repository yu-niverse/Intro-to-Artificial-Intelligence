import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
    1. load the csv file into rows
    2. store them into the dictionary edges in the format:  
        {startID : list of attached (endID, distance)}
    3. implement DFS with stack
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

    stack = [(start, 0, 0)]
    explored = [start]
    trace = {}
    num_visited = 0
    while (True):
        flag = True
        top = stack[-1]
        stack.pop()
        if top[0] in edges.keys():  # check if it is a leaf node
            for ID, distance in edges[top[0]]:
                if ID in explored: continue  # check if the node is explored
                if ID == end:
                    num_visited += 1
                    trace[end] = (top[0], distance)
                    flag = False
                    break
                stack.append((ID, distance, top[0]))
                explored.append(ID)
                num_visited += 1
        trace[top[0]] = (top[2], top[1])
        if flag == False: break

    dist = 0
    path = [end]
    dist += trace[path[0]][1]
    while trace[path[0]][0] != 0:
        path.insert(0, trace[path[0]][0])
        dist += trace[path[0]][1]

    return path, dist, num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
