'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

import heapq


def dfs(cost, start_point, goals):
    return []

def ucs(cost, start_point, goals):
    return []

def astar(cost, heuristic, start_point, goals):
    frontier = []
    infront = [0]*(len(cost))
    infront[start_point] = 1
    leastcost = [float('inf')]*len(cost)
    leastparent = [-1]*len(cost)
    #Format: (F, G, NameOfNode)
    frontier.append((heuristic[start_point], 0, start_point))
    leastcost[start_point] = 0
    ptogoals = []
    while(len(frontier) != 0):
        temp = frontier.pop(0)
        infront[temp[2]] = 0
        for i in range(1,len(cost)):
            if(cost[temp[2]][i] > 0):
                cc = temp[1]+cost[temp[2]][i]+heuristic[i]
                if(infront[i] == 0): #The node is not in frontier
                    if(leastcost[i] > cc):
                        leastcost[i] = cc
                        leastparent[i] = temp[2]
                        frontier.append((cc, cc-heuristic[i], i))
                        infront[i] = 1
                else: #The node is already in frontier
                    for j,value in enumerate(frontier):
                        if(value[2] == i):
                            if((cc - heuristic[i]) < value[1]):
                                frontier.pop(j)
                                frontier.append((cc, cc-heuristic[i], i))
                                leastcost[i] = cc
                                leastparent[i] = temp[2]
                if(i in goals):
                    ptogoals.append((cc, cc, i, temp[2]))
                    
            heapq.heapify(frontier)
    n1 = sorted(ptogoals)[0][2]
    n2 = sorted(ptogoals)[0][3]
    path = []
    while(n1 != start_point):
        path.append(n1)
        n1 = n2
        n2 = leastparent[n2]
    path.append(start_point)
    path.reverse()
    return path    
    


def tri_traversal(cost, heuristic, start_point, goals):
    l = []


    t1 = dfs(cost, start_point, goals)
    t2 = ucs(cost, start_point, goals)
    t3 = astar(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

