# from src.MI_0316_1286_2057 import *
import importlib


temp = importlib.import_module('src.PESU-MI_0316_1286_2057')
locals()['UCS_Traversal'] = temp.UCS_Traversal
locals()['DFS_Traversal'] = temp.DFS_Traversal
locals()['A_star_Traversal'] = temp.A_star_Traversal
locals()['tri_traversal'] = temp.tri_traversal


cost1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]

# Observe here that the goal node 9 has a non-zero heuristic
# Hence, the heuristics here are incorrect
# heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]

# If we make the heurisitcs of the goal node 9 to 0 like the
# other goal nodes, 7 and 10. Then the answers of UCS and A*
# match
heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 0, 0]

astar =  A_star_Traversal
dfs = DFS_Traversal
ucs=UCS_Traversal


def a_start_ucs_combo():
    """
    This function test both UCS and A* and makes sure that their results are the same
    """

    print("Test 16: ", ucs(cost1,1, [7,9,10]) == astar(cost1, heuristic1, 1, [7, 9, 10]))
    print()
    print("What UCS says:\n", ucs(cost1,1, [7,9,10]))
    print()
    print("What A* says:\n", astar(cost1, heuristic1, 1, [7,9,10]))

print("Now, for the final test...")
a_start_ucs_combo()
