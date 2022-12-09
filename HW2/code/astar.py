import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    """
    1. determine the case number
    2. load the csv files into dists & rows
    3. store them into the dictionary edges in the format:  
        {startID : list of attached (endID, distance)}
    4. implement A*
    5. backtrack the path by using the list trace
    6. return path, dist, num_visited
    """
    # raise NotImplementedError("To be implemented")
    case = 0

    with open(heuristicFile, mode='r') as input:
        reader = csv.reader(input)
        next(reader)
        dists = {int(rows[0]):[float(rows[1]),float(rows[2]),float(rows[3])] for rows in reader}

    edges = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)
        # edges: {startID : list of attached (endID, distance) s}
        for row in rows:
            if edges.get(int(row[0])) != None:
                edges[int(row[0])].append((int(row[1]), float(row[2])))
            else: edges[int(row[0])] = [(int(row[1]), float(row[2]))]   


    
    queue = [[start, dists[start][case], 0, 0]]  # ID, accumulated distance, parent
    trace = {}
    num_visited = 0
    while (True):
        flag = True
        # altered = False
        min = 0  # the index having the minimum accumulated distance
        for i in queue:
            if i[1] < queue[min][1]: min = queue.index(i)

        if queue[min][0] in edges.keys():  # check if it is a leaf node
            for ID, distance in edges[queue[min][0]]: 
                altered = False
                if ID in trace.keys(): continue  # check if the node is explored
                for i in range(len(queue)):
                    if queue[i][0] == ID:
                        if queue[min][2] + distance + dists[ID][case] < queue[i][1]: 
                            queue[i][1] = queue[min][2] + distance + dists[ID][case]
                            queue[i][2] = queue[min][2] + distance
                            queue[i][3] = queue[min][0]
                        altered = True
                if altered == True: continue     
                if ID == end: 
                    trace[end] = queue[min][0]
                    dist = queue[min][2] + distance
                    num_visited += 1
                    flag = False
                    break
                queue.append([ID, queue[min][2] + distance + dists[ID][case], queue[min][2] + distance, queue[min][0]])
                num_visited += 1

        trace[queue[min][0]] = queue[min][3]
        if flag == False: break
        queue.pop(min)

    path = [end]
    while trace[path[0]] != 0:
        path.insert(0, trace[path[0]])

    return path, dist, num_visited
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
