import heapq
from collections import deque


class Node:
    def __init__(self, parent, node_id, g_value=0, h_value=0):
        self.parent = parent
        self.node_id = node_id
        self.g_value = g_value
        self.h_value = h_value

    def getFValue(self):
        return self.g_value + self.h_value

    def getGValue(self):
        return self.g_value

    def getNode_ID(self):
        return self.node_id

    def setParent(self, parent):
        self.parent = parent

    def setGValue(self, g_value):
        self.g_value = g_value

    def createFrontierRecord(self):
        return [self.getFValue(), self.node_id, self]

    def getPath(self):
        curr_obj = self
        path_to_node = []
        while(curr_obj is not None):
            path_to_node.insert(0, curr_obj.node_id)
            curr_obj = curr_obj.parent

        return path_to_node


def A_star_Traversal(cost, heuristic, start_point, goals):
    # these values are given according to the function
    # createFrontierRecord() in Node class
    NODE_EVAL_FUNC_VALUE_INDEX = 0
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
        if len(frontier) == 0:
            return []

        # Pop the node from the heap having the least cost
        popped_node_record = heapq.heappop(frontier)

        # If the popped node is a goal, return the path to the goal
        # node
        if popped_node_record[NODE_ID_INDEX] in goals:
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
            if cost[popped_node_record[NODE_ID_INDEX]][i] > 0:

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
                    node = Node(popped_node_record[NODE_OBJ_INDEX], i, g_value, h_value)
                    heapq.heappush(frontier, node.createFrontierRecord())

                # If the new node is already in the frontier
                elif inFrontier is True:

                    # Finding the node with same value in the frontier
                    for j in frontier:
                        if j[NODE_ID_INDEX] == i:
                            # Formula used:
                            # Actual path cost from the initial node to the popped_node
                            # + Step cost from the popped_node to the neighbour node
                            # + Heuristic of the neighbour node
                            g_value_of_node_i = popped_node_record[NODE_OBJ_INDEX].getGValue() + \
                                                cost[popped_node_record[NODE_ID_INDEX]][i]
                            h_value_of_node_i = heuristic[i]
                            f_value_of_node_i = g_value_of_node_i + h_value_of_node_i

                            # If the current cost is lesser than
                            # the cost of the node currently in the frontier,
                            # then we have to update
                            if j[NODE_EVAL_FUNC_VALUE_INDEX] > f_value_of_node_i:
                                # (self.getFValue(), self.node_id, self)

                                # If we reach here that means that the
                                # new path cost found is lesser than the one
                                # in the frontier
                                # Update the evaluation function at index 0, i.e. f(n)
                                j[NODE_EVAL_FUNC_VALUE_INDEX] = f_value_of_node_i
                                # Update the path cost (from initial to neighbour node) in the frontier
                                j[NODE_OBJ_INDEX].setParent(popped_node_record[NODE_OBJ_INDEX])
                                # Update the path in the frontier
                                j[NODE_OBJ_INDEX].setGValue(g_value_of_node_i)

                                # Heapify the frontier, again
                                heapq.heapify(frontier)

                            break

    return l


def UCS_Traversal(cost, start_point, goals):
    # these values are given according to the function
    # createFrontierRecord() in Node class
    NODE_EVAL_FUNC_VALUE_INDEX = 0
    NODE_ID_INDEX = 1
    NODE_OBJ_INDEX = 2

    l = []

    # n = total number of nodes + 1
    n = len(cost[0])

    initial_node = Node(None, start_point)

    # The frontier is a min heap that will store the nodes
    frontier = []

    # Push the inital node into the frontier
    heapq.heappush(frontier, initial_node.createFrontierRecord())

    # Explored will store all the nodes already explored.
    explored = set()

    while (True):
        # If the frontier is empty, our search algorithm has failed
        if len(frontier) == 0:
            return []

        # Pop the node from the heap having the least cost
        popped_node_record = heapq.heappop(frontier)

        # If the popped node is a goal, return the path to the goal
        # node
        if popped_node_record[NODE_ID_INDEX] in goals:
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
            if cost[popped_node_record[NODE_ID_INDEX]][i] > 0:

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
                    node = Node(popped_node_record[NODE_OBJ_INDEX], i, g_value)
                    heapq.heappush(frontier, node.createFrontierRecord())

                # If the new node is already in the frontier
                elif inFrontier is True:

                    # Finding the node with same value in the frontier
                    for j in frontier:
                        if j[NODE_ID_INDEX] == i:
                            # Formula used:
                            # Actual path cost from the initial node to the popped_node
                            # + Step cost from the popped_node to the neighbour node
                            # + Heuristic of the neighbour node (which is 0 for UCS)
                            g_value_of_node_i = popped_node_record[NODE_OBJ_INDEX].getGValue() + \
                                                cost[popped_node_record[NODE_ID_INDEX]][i]
                            h_value_of_node_i = 0
                            f_value_of_node_i = g_value_of_node_i + h_value_of_node_i

                            # If the current cost is lesser than
                            # the cost of the node currently in the frontier,
                            # then we have to update
                            if j[NODE_EVAL_FUNC_VALUE_INDEX] > f_value_of_node_i:
                                # (self.getFValue(), self.node_id, self)

                                # If we reach here that means that the
                                # new path cost found is lesser than the one
                                # in the frontier
                                # Update the evaluation function at index 0, i.e. f(n)
                                j[NODE_EVAL_FUNC_VALUE_INDEX] = f_value_of_node_i
                                # Update the path cost (from initial to neighbour node) in the frontier
                                j[NODE_OBJ_INDEX].setParent(popped_node_record[NODE_OBJ_INDEX])
                                # Update the path in the frontier
                                j[NODE_OBJ_INDEX].setGValue(g_value_of_node_i)

                                # Heapify the frontier, again
                                heapq.heapify(frontier)

                            break

    return l


# Get a list of the neighbours of the popped node.
# We return a list of the indices of the neighbours
def getNeighbours(adjList):
    neighbourList = []

    # NOTE: We ignore the first node (as 1 based indexing for nodes has
    # been used)
    START_ADJUSTED_LIST = adjList[1::]
    for index, node in enumerate(START_ADJUSTED_LIST, start=1):
        # If the value is greater than 0, then there is a path/edge from the
        # popped node to the node and it is not a self loop
        if node > 0:
            neighbourList.append(index)

    # NOTE: The list is reversed so that we can add the nodes into
    # the stack in reverse lexicographical order, so that
    # while popping from the stack we can retrieve them in
    # lexicographical order
    return neighbourList[::-1]


# Check if the node is in the frontier
def isInFrontier(node_id, frontier):
    inFrontier = False  # Assume the node is not in the frontier

    for frontier_record in frontier:
        # If the node is in the frontier
        if frontier_record["Node Object"].getNode_ID() == node_id:
            # then set inFrontier to be true
            inFrontier = True
            break

    return inFrontier


def DFS_Traversal(cost, start_point, goals):

    # Frontier for DFS, i.e. the stack
    stack = deque()

    # Push the inital node into the frontier/stack
    stack.append({
        "Node Object": Node(None, start_point)
    })

    # Set to hold the list of nodes explored
    exploredSet = set()

    # While the frontier/stack is not empty
    while (stack):
        # Pop a node from the frontier/stack
        popped_node_record = stack.pop()

        # Print the popped node
        # print("Popped node:", popped_node_record)

        # If the popped node has already been explored
        # then do not do any further processing for it
        # if popped_node_record["Node Object"].getNode_ID() in exploredSet:
        #     continue

        # Check if the popped node is one of the goal states
        if popped_node_record["Node Object"].getNode_ID() in goals:
            # Printing the path found for Diagnostics
            # print("Path from DFS is:", popped_node_record["path"])

            # Return the path found
            return popped_node_record["Node Object"].getPath()

        NODE_ID = popped_node_record["Node Object"].getNode_ID()
        # Add the node to the explored set
        exploredSet.add(NODE_ID)

        # If the popped node is not one of the goal states

        # Expand the node, and get the list of neighbours' indices
        poppedNodeNeighbours = getNeighbours(cost[NODE_ID])

        # Explored set
        # print("exploredSet:", exploredSet)

        # Print the poppedNodeNeighbours
        # print("The poppedNodeNeighbours:", poppedNodeNeighbours)

        # Print the stack
        # print("Stack:", stack)

        # print("---")

        # Add the resulting nodes (child nodes) into the frontier,
        # if they aren't already in the frontier or the explored set
        for neighbour in poppedNodeNeighbours:
            if (isInFrontier(neighbour, stack) is False) and (neighbour not in exploredSet):
                poppedNodeNeighbourRecord = {
                    "Node Object": Node(popped_node_record["Node Object"], neighbour),
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
