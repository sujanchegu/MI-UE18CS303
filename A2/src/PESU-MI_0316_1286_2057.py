import heapq
from collections import deque


class Node:
    def __init__(self, parent, node_id, g_value, h_value=0):
        self.parent = parent
        self.node_id = node_id
        self.g_value = g_value
        self.h_value = h_value

    def getFValue(self):
        return self.g_value + self.h_value

    def getGValue(self):
        return self.g_value

    def createFrontierRecord(self):
        return (self.getFValue(), self.node_id, self)

    def getPath(self):
        pass


def A_star_Traversal(cost, heuristic, start_point, goals):
    # these values are given according to the function
    # createFrontierRecord() in Node class
    EVAL_FUNC_INDEX = 0
    NODE_ID_INDEX = 1
    NODE_OBJ_INDEX = 2

    l = []

    # n = total number of nodes + 1
    n = len(cost[0])

    initial_node = Node(None, start_point, 0, heuristic[start_point])

    # The frontier is a min heap that will store the nodes
    frontier = []

    # Push the inital node into the frontier
    heapq.heappush(frontier, initial_node.createFrontierRecord())

    # Explored will store all the nodes already explored.
    explored = set()

    while (True):
        # If the frontier is empty, our search algorithm has failed
        if (len(frontier) == 0):
            return []

        # Pop the node from the heap having the least cost
        popped_node_record = heapq.heappop(frontier)

        # If the popped node is a goal, return
        if (popped_node_record[NODE_ID_INDEX] in goals):
            return popped_node_record[NODE_OBJ_INDEX].getPath()

        # Add the node to the explored set
        explored.add(popped_node_record[NODE_ID_INDEX])

        # Here, we are logically going through all the neighbour nodes only

        # We iterate through all the nodes mentioned in the cost matrix row
        # whether there is a direct path to them or not
        for i in range(1, n):

            # Only if there is a direct path from the current popped node
            # to a node mentioned in the cost matrix row
            # i.e. when it is a neighbour node, do we proceed with
            # any further processing on that node

            # cost[popped_node[2]][i] -> Cost to travel to the neighbour node i
            # If there's an edge from popped node to i and it is not a self loop
            if (cost[popped_node_record[NODE_ID_INDEX]][i] != -1) and \
               (cost[popped_node_record[NODE_ID_INDEX]][i] != 0):

                # Check if the node is in the frontier
                inFrontier = False  # Assume the node is not in the frontier
                for frontier_record in frontier:
                    # If the node is in the frontier
                    if frontier_record[NODE_ID_INDEX] == i:
                        # then set inFrontier to be true
                        inFrontier = True
                        break

                # If the new node is neither in the frontier nor in
                # the explored set, add it to the heap
                if (inFrontier is False) and (i not in explored):
                    # path_to_node_i = popped_node_record[1] + list((i,))
                    g_value = popped_node_record[NODE_OBJ_INDEX].getGValue() + cost[popped_node_record[NODE_ID_INDEX]][i]
                    h_value = heuristic[i]
                    f_value = g_value + h_value
                    heapq.heappush(frontier,
                                   tuple(
                                            f_value,
                                            i,
                                            Node(popped_node_record[NODE_ID_INDEX], i, g_value, h_value)
                                        )
                                   )

                # If the new node is already in the frontier
                elif inFrontier is True:

                    # Finding the node with same value in the frontier
                    for j in frontier:
                        if j[2] == i:

                            # If the current cost is lesser than
                            # the cost of the node currently in the frontier,
                            # then we have to update
                            if (j[3] > popped_node[3] + cost[popped_node[2]][i]):
                                # If the path cost is the same then the path
                                # choosen must be lexicographically smaller,
                                # to maintain lexicographical order
                                # which is enforced here
                                if (j[3] == popped_node[3] + cost[popped_node[2]][i]) \
                                   and (j[1] <= popped_node[1] + list((i,))):
                                    # If the new path is lexicographically
                                    # greater or equal than the path in the
                                    # frontier then break out of the for loop
                                    break

                                # If we reach here that means that either the new path
                                # cost found is lesser than the one in the frontier or the
                                # new path cost found has equal cost but it is lexicographically
                                # smaller
                                # Update the evaluation function at index 0, i.e. f(n)
                                # Formula used:
                                # Actual path cost from the initial node to the popped_node
                                # + Step cost from the popped_node to the neighbour node
                                # + Heuristic of the neighbour node
                                j[0] = popped_node[3] + cost[popped_node[2]][i] + heuristic[i]
                                # Update the path cost (from initial to neighbour node) in the frontier
                                j[3] = popped_node[3] + cost[popped_node[2]][i]
                                # Update the path in the frontier
                                j[1] = popped_node[1] + list((i,))
                                heapq.heapify(frontier)

                            # Once we have modified/handled the node, in the
                            # frontier we can exit the loop
                            # There will be at most one node of a certain
                            # number or ID in the frontier always
                            break

    return l


def UCS_Traversal(cost, start_point, goals):
    l = []

    # n = total number of nodes + 1
    n = len(cost[0])

    # We define our node structure to be a list containing
    # The cost to reach the node from the start_point, i.e. path cost
    # The path considered
    # The node value
    node = [0, [start_point], start_point]

    # The frontier is a min heap that will store the nodes
    frontier = []
    frontier.append(node)

    # Explored will store all the nodes already explored.
    explored = set()

    while (True):
        # If the frontier is empty, our search algorithm has failed
        if (len(frontier) == 0):
            return []

        # print(frontier)
        # Pop the node from the heap having the least cost
        popped_node = heapq.heappop(frontier)

        # If the popped node is a goal, return
        if (popped_node[2] in goals):
            return popped_node[1]

        # Add the node to the explored set
        explored.add(popped_node[2])

        # Here, we are logically going through all the neighbour nodes only

        # We iterate through all the nodes mentioned in the cost matrix row
        # whether there is a direct path to them or not
        for i in range(1, n):

            # Only if there is a direct path from the current popped node
            # to a node mentioned in the cost matrix row
            # i.e. when it is a neighbour node, do we proceed with
            # any further processing on that node

            # cost[popped_node[1]][i] -> Cost to travel to the neighbour node i
            # If there's an edge from popped node to i and it is not a self loop
            if ((cost[popped_node[2]][i] != -1) and (cost[popped_node[2]][i] != 0)):

                # Check if the node is in the frontier
                boo = False
                for j in frontier:
                    if(j[2] == i):
                        boo = True
                        break

                # If the new node is neither in the frontier nor in
                # the explored set, add it to the heap
                if ((boo is False) and (i not in explored)):
                    temp = popped_node[1] + list((i,))
                    heapq.heappush(frontier, list(
                        (popped_node[0] + cost[popped_node[2]][i], temp, i)))

                # If the new node is already in the frontier
                elif (boo is True):

                    # Finding the node with same value in the frontier
                    for j in frontier:
                        if j[2] == i:

                            # If the current cost is lesser than or equal to
                            # the cost of the node currently in the frontier,
                            # then we may have to update
                            if (j[0] >= popped_node[0] + cost[popped_node[2]][i]):
                                # If the path cost is the same then the path
                                # choosen must be lexicographically smaller,
                                # to maintain lexicographical order
                                # which is enforced here
                                if (j[0] == popped_node[0] + cost[popped_node[2]][i]) \
                                   and (j[1] <= popped_node[1] + list((i,))):
                                    # If the new path is lexicographically
                                    # greater or equal than the current path
                                    # then break out of the for loop
                                    break

                                # Update the cost in the frontier
                                j[0] = popped_node[0] + cost[popped_node[2]][i]
                                # Update the path in the frontier
                                j[1] = popped_node[1] + list((i,))
                                heapq.heapify(frontier)

                            # Once we have modified/handled the node, in the
                            # frontier we can exit the loop
                            # There will be only one node of a certain
                            # number or ID in the frontier always
                            break

    return l


# Defining the Goal Test Function
def goalTest(state, goals):
    return state in goals


# Get a list of the neighbours of the popped node.
# We return a list of the indices of the neighbours
def getNeighbours(adjList):
    neighbourList = []

    # NOTE: We ignore the first node (as 1 based indexing for nodes has
    # been used)
    for index, node in enumerate(adjList[1::], start=1):
        # If the value is greater than 0, then there is a path/edge from the
        # popped node to the node
        if node > 0:
            neighbourList.append(index)

    # NOTE: The list is reversed so that we can add the nodes into
    # the stack in reverse lexicographical order, so that
    # while popping from the stack we can retrieve them in
    # lexicographical order
    return neighbourList[::-1]


def DFS_Traversal(cost, start_point, goals):
    # return [1]
    # Frontier for DFS, i.e. the stack
    stack = deque()

    # Push the inital node into the frontier/stack
    stack.append({
        "node": start_point,
        "path": [start_point]
    })

    # Set to hold the list of nodes explored
    exploredSet = set()

    # While the frontier/stack is not empty
    while (stack):
        # Pop a node from the frontier/stack
        poppedNode = stack.pop()

        # Print the popped node
        # print("Popped node:", poppedNode)

        # If the popped node has already been explored
        # then do not do any further processing for it
        if poppedNode["node"] in exploredSet:
            continue

        # Add the node to the explored set
        exploredSet.add(poppedNode["node"])

        # Check if the popped node is one of the goal states
        if goalTest(poppedNode["node"], goals) is True:
            # Printing the path found for Diagnostics
            # print("Path from DFS is:", poppedNode["path"])

            # Return the path found
            return poppedNode["path"]

        # If the popped node is not one of the goal states

        # Expand the node, and get the list of neighbours' indices
        poppedNodeNeighbours = getNeighbours(cost[poppedNode["node"]])

        # Explored set
        # print("exploredSet:", exploredSet)

        # Print the poppedNodeNeighbours
        # print("The poppedNodeNeighbours:", poppedNodeNeighbours)

        # Print the stack
        # print("Stack:", stack)

        # print("---")

        # Add the resulting nodes (child nodes) into the frontier,
        # if they aren't already in the frontier or the explored set
        for poppedNodeNeighbour in poppedNodeNeighbours:
            if poppedNodeNeighbour not in exploredSet:
                poppedNodeNeighbourRecord = {
                    "node": poppedNodeNeighbour,
                    "path": poppedNode["path"] + [poppedNodeNeighbour]
                }
                stack.append(poppedNodeNeighbourRecord)
        # print("Stack:", stack)
    # If we reached here, then that means that the frontier was empty
    # before we reached a goal state, and hence there is no solution so
    # we return an empty list
    return []


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
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''


def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
