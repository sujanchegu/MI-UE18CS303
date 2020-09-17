def A_star_Traversal(cost, heuristic, start_point, goals):
    # Frontier for DFS, i.e. the stack
    stack = deque()

    # Set to hold the list of nodes explored
    exploredSet = set()

    # Path list to hold the path as it is being built
    # As all paths start withe the start_point, it has
    # been added in by default
    path = [start_point]

    # Push the inital node into the frontier/stack
    stack.append(start_point)

    # While the frontier/stack is not empty
    while (stack):
        # Pop a node from the frontier/stack
        poppedNode = stack.pop()

        # Check if the popped node is one of the goal states
        if goalTest(poppedNode, goals) is True:
            # Return the path found
            return path

        # If the popped node is not one of the goal states

        # Add the node to the explored set
        exploredSet.add(poppedNode)

        # Expand the node, and get the list of neighbours' indices
        poppedNodeNeighbours = getNeighbours(cost[poppedNode])

        # Add the resulting nodes (child nodes) into the frontier,
        # if they aren't already in the frontier or the explored set
        for poppedNodeNeighbour in poppedNodeNeighbours:
            if (poppedNodeNeighbour not in stack) and \
               (poppedNodeNeighbour not in exploredSet):
                stack.append(poppedNodeNeighbour)

    # If we reached here, then that means that the frontier was emtpy
    # before we reached a goal state, and hence there is no solution so
    # we return an empty list
    return []

def UCS_Traversal(cost, start_point, goals):
    l = []

    # n = total number of nodes + 1
    n = len(cost[0])

    # We define our node structure to be a list containing
    # The cost to reach the node from the start_point
    # The node value
    # The path considered
    node = [0, start_point, [start_point]]

    # The frontier is a min heap that will store the nodes
    frontier = []
    frontier.append(node)

    # Explored will store all the nodes already explored.
    explored = set()

    while (True):
        # If the frontier is empty, our search algorithm has failed
        if (len(frontier) == 0):
            return []

        # Pop the node from the heap having the least cost
        popped_node = heapq.heappop(frontier)

        # If the popped node is a goal, return
        if (popped_node[1] in goals):
            return popped_node[2]

        # Add the node to the explored set
        explored.add(popped_node[1])

        # Going through all the nodes
        for i in range(1, n):

            # If there's an edge from popped node to i
            if (cost[popped_node[1]][i] != -1):

                # If the new node is neither in the frontier nor in
                # the explored set, add it to the heap
                if ((i not in frontier) and (i not in explored)):
                    temp = popped_node[2] + list((i,))
                    heapq.heappush(frontier, list(
                        (popped_node[0] + cost[popped_node[1]][i], i, temp)))

                # If the new node is already in the frontier
                elif (i in frontier):

                    # Finding the node with same value in the frontier
                    for j in frontier:
                        if (j[1] == i):

                            # If the current cost is lesser than the cost of the node currently in the frontier, update
                            if (j[0] > popped_node[0] + cost[popped_node[1]][i]):
                                j[0] = popped_node[0] + cost[popped_node[1]][i]
                                j[2] = popped_node[2] + list((i,))
                                heapq.heapify(frontier)

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
        # If the value is not -1, then there is a path/edge from the
        # popped node to the node
        if node != -1:
            neighbourList.append(index)

    # NOTE: The list is reversed so that we can add the nodes into
    # the stack in reverse lexicographical order, so that
    # while popping from the stack we can retrieve them in
    # lexicographical order
    return neighbourList[::-1]


def DFS_Traversal(cost, start_point, goals):
    # Frontier for DFS, i.e. the stack
    stack = deque()

    # Set to hold the list of nodes explored
    exploredSet = set()

    # Path list to hold the path as it is being built
    # As all paths start withe the start_point, it has
    # been added in by default
    path = [start_point]

    # Push the inital node into the frontier/stack
    stack.append(start_point)

    # While the frontier/stack is not empty
    while (stack):
        # Pop a node from the frontier/stack
        poppedNode = stack.pop()

        # Check if the popped node is one of the goal states
        if goalTest(poppedNode, goals) is True:
            # Return the path found
            return path

        # If the popped node is not one of the goal states

        # Add the node to the explored set
        exploredSet.add(poppedNode)

        # Expand the node, and get the list of neighbours' indices
        poppedNodeNeighbours = getNeighbours(cost[poppedNode])

        # Add the resulting nodes (child nodes) into the frontier,
        # if they aren't already in the frontier or the explored set
        for poppedNodeNeighbour in poppedNodeNeighbours:
            if (poppedNodeNeighbour not in stack) and \
               (poppedNodeNeighbour not in exploredSet):
                stack.append(poppedNodeNeighbour)

    # If we reached here, then that means that the frontier was emtpy
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
