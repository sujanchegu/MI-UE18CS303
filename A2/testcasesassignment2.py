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
        
cost2 = [[0,0,0,0,0,0,0,0],
	[0,0,3,-1,-1,-1,-1,2],
	[0,-1,0,5,10,-1,-1,-1],
	[0,-1,-1,0,2,-1,1,-1],
	[0,-1,-1,-1,0,4,-1,-1],
	[0,-1,-1,-1,-1,0,-1,-1],
	[0,-1,-1,-1,-1,3,0,-1],
	[0,-1,-1,1,-1,-1,4,0]] #https://www.geeksforgeeks.org/search-algorithms-in-ai/

cost3 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 6, -1, -1, -1],
    [0, 6, 0, 3, 3, -1],
    [0, -1, 3, 0, 1, 7],
    [0, -1, 3, 1, 0, 8],
    [0, -1, -1, 7, 8, 0],
]

heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0,7,9,4,2,0,3,5]
heuristic3 = [0, 10, 8, 7, 7, 3]

astar =  A_star_Traversal
dfs = DFS_Traversal
ucs=UCS_Traversal



def dfstest():
	print("---------------------------------- TESTS FOR DFS SEARCH----------------------------------------")
	print("Test 1: ", dfs(cost1, 1, [1]) == [1])
	print("Test 2: ", dfs(cost1, 1, [2]) == [1,2])
	print("Test 3: ", dfs(cost1, 1, [3]) == [1,2,3])
	print("Test 4: ", dfs(cost1, 1, [4]) == [1,2,3,4])
	print("Test 5: ", dfs(cost1, 1, [5]) == [1,2,3,4,8,5])
	print("Test 6: ", dfs(cost1, 1, [6]) == [1,2,6])
	print("Test 7: ", dfs(cost1, 1, [7]) == [1,2,3,4,7])
	print("Test 8: ", dfs(cost1, 1, [8]) == [1,2,3,4,8])
	print("Test 9: ", dfs(cost1, 1, [9]) == [1,2,3,4,8,5,9])
	print("Test 10: ", dfs(cost1, 1, [10]) == [1, 2, 3, 4, 8, 5, 9, 10])
	print("Test 11: ", dfs(cost1, 1, [6,7,10]) == [1,2,3,4,7])
    
	print("Test 12: ", dfs(cost1, 1, [3,4,7,10]) == [1,2,3])
	print("Test 13: ", dfs(cost1, 1, [5,9,4]) == [1,2,3,4])
	print("Test 14: ", dfs(cost1, 1, [4,8,10]) == [1,2,3,4])
	print("Test 15: ", dfs(cost1, 1, [2,8,5]) == [1,2])
	print("Test 16: ", dfs(cost1, 1, [7,9,10]) == [1,2,3,4,7])
	print("Test 17: ", dfs(cost1, 1, [10,6,8,4]) == [1,2,3,4])
	print("Test 18: ", dfs(cost1, 1, [9,7,5,10]) == [1,2,3,4,7])

	print("Test 19: ", dfs(cost2, 1, [1]) == [1])
	print("Test 20: ", dfs(cost2, 1, [2]) == [1,2])
	print("Test 21: ", dfs(cost2, 1, [3]) == [1,2,3])
	print("Test 22: ", dfs(cost2, 1, [4]) == [1,2,3,4])
	print("Test 23: ", dfs(cost2, 1, [5]) == [1,2,3,4,5])
	print("Test 24: ", dfs(cost2, 1, [6]) == [1,2,3,6])
	print("Test 25: ", dfs(cost2, 1, [7]) == [1,7])
	print("Test 26: ", dfs(cost2, 1, [4,5,6]) == [1,2,3,4])
	print("Test 27: ", dfs(cost2, 1, [3,6,7]) == [1,2,3])
	print("Test 28: ", dfs(cost2, 1, [4,6]) == [1,2,3,4])
	print("Test 29: ", dfs(cost2, 1, [2,3,7]) == [1,2])

	print("Test 30: ", dfs(cost2, 4, [3]) == [])
	print("Test 31: ", dfs(cost3, 1, [5]) == [1,2,3,4,5])
	
    
    
    
    
    
    
  
def ucstest():        
    print("----------------------------------TESTS FOR UCS SEARCH----------------------------------------")
    print("Test 1: ", ucs(cost1,1, [1]))
    print("Test 2: ", ucs(cost1,1, [2]))
    print("Test 3: ", ucs(cost1,1, [3]))
    print("Test 4: ", ucs(cost1,1, [4]))
    print("Test 5: ", ucs(cost1,1, [5]))
    print("Test 6: ", ucs(cost1,1, [6]))
    print("Test 7: ", ucs(cost1,1, [7]))
    print("Test 8: ", ucs(cost1,1, [8]))
    print("Test 9: ", ucs(cost1,1, [9]))
    print("Test 10: ", ucs(cost1,1, [10]))
    print("Test 11: ", ucs(cost1,1, [6,7,10]))
    print("Test 12: ", ucs(cost1,1, [3,4,7,10]))
    print("Test 13: ", ucs(cost1,1, [5,9,4]))
    print("Test 14: ", ucs(cost1,1, [4,8,10]))
    print("Test 15: ", ucs(cost1,1, [2,8,5]))
    print("Test 16: ", ucs(cost1,1, [7,9,10]))
    print("Test 17: ", ucs(cost1,1, [10,6,8,4]))
    print("Test 18: ", ucs(cost1,1, [9,7,5,10]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def astartest():
	print("----------------------------------TESTS FOR A* SEARCH----------------------------------------")
	print("Test 1: ", astar(cost1, heuristic1, 1, [1]))
	print("Test 2: ", astar(cost1, heuristic1, 1, [2]))
	print("Test 3: ", astar(cost1, heuristic1, 1, [3]))
	print("Test 4: ", astar(cost1, heuristic1, 1, [4]))
	print("Test 5: ", astar(cost1, heuristic1, 1, [5]))
	print("Test 6: ", astar(cost1, heuristic1, 1, [6]))
	print("Test 7: ", astar(cost1, heuristic1, 1, [7]))
	print("Test 8: ", astar(cost1, heuristic1, 1, [8]))
	print("Test 9: ", astar(cost1, heuristic1, 1, [9]))
	print("Test 10: ", astar(cost1, heuristic1, 1, [10]))
	print("Test 11: ", astar(cost1, heuristic1, 1, [6,7,10]))
	print("Test 12: ", astar(cost1, heuristic1, 1, [3,4,7,10]))
	print("Test 13: ", astar(cost1, heuristic1, 1, [5,9,4]))
	print("Test 14: ", astar(cost1, heuristic1, 1, [4,8,10]))
	print("Test 15: ", astar(cost1, heuristic1, 1, [2,8,5]))
	print("Test 16: ", astar(cost1, heuristic1, 1, [7,9,10]))
	print("Test 17: ", astar(cost1, heuristic1, 1, [10,6,8,4]))
	print("Test 18: ", astar(cost1, heuristic1, 1, [9,7,5,10]))

	# print("Test 19: ", astar(cost2, heuristic2, 1, [1]))
	print("Test 20: ", astar(cost2, heuristic2, 1, [2]))
	print("Test 21: ", astar(cost2, heuristic2, 1, [3]))
	print("Test 22: ", astar(cost2, heuristic2, 1, [4]))
	print("Test 23: ", astar(cost2, heuristic2, 1, [5]))
	print("Test 24: ", astar(cost2, heuristic2, 1, [6]))
	print("Test 25: ", astar(cost2, heuristic2, 1, [7]))
	print("Test 26: ", astar(cost2, heuristic2, 1, [4,5,6]))
	print("Test 27: ", astar(cost2, heuristic2, 1, [3,6,7]))
	print("Test 28: ", astar(cost2, heuristic2, 1, [4,6]))
	print("Test 29: ", astar(cost2, heuristic2, 1, [2,3,7]))

	print("Test 30: ", astar(cost2, heuristic2, 4, [3]))
	print("Test 31: ", astar(cost3, heuristic3, 1, [5]))


def a_start_ucs_combo():
    """
    This function test both UCS and A* and makes sure that their results are the same
    """
    print("Test 1: ", ucs(cost1,1, [1]) == astar(cost1, heuristic1, 1, [1]))
    print("Test 2: ", ucs(cost1,1, [2]) == astar(cost1, heuristic1, 1, [2]))
    print("Test 3: ", ucs(cost1,1, [3]) == astar(cost1, heuristic1, 1, [3]))
    print("Test 4: ", ucs(cost1,1, [4]) == astar(cost1, heuristic1, 1, [4]))
    print("Test 5: ", ucs(cost1,1, [5]) == astar(cost1, heuristic1, 1, [5]))
    print("Test 6: ", ucs(cost1,1, [6]) == astar(cost1, heuristic1, 1, [6]))
    print("Test 7: ", ucs(cost1,1, [7]) == astar(cost1, heuristic1, 1, [7]))
    print("Test 8: ", ucs(cost1,1, [8]) == astar(cost1, heuristic1, 1, [8]))
    print("Test 9: ", ucs(cost1,1, [9]) == astar(cost1, heuristic1, 1, [9]))
    print("Test 10: ", ucs(cost1,1, [10]) == astar(cost1, heuristic1, 1, [10]))
    print("Test 11: ", ucs(cost1,1, [6,7,10]) == astar(cost1, heuristic1, 1, [6,7,10]))
    print("Test 12: ", ucs(cost1,1, [3,4,7,10]) == astar(cost1, heuristic1, 1, [3,4,7,10]))
    print("Test 13: ", ucs(cost1,1, [5,9,4]) == astar(cost1, heuristic1, 1, [5,9,4]))
    print("Test 14: ", ucs(cost1,1, [4,8,10]) == astar(cost1, heuristic1, 1, [4,8,10]))
    print("Test 15: ", ucs(cost1,1, [2,8,5]) == astar(cost1, heuristic1, 1, [2,8,5]))
    print("Test 16: ", ucs(cost1,1, [7,9,10]) == astar(cost1, heuristic1, 1, [7,9,10]))
    print("Test 17: ", ucs(cost1,1, [10,6,8,4]) == astar(cost1, heuristic1, 1, [10,6,8,4]))
    print("Test 18: ", ucs(cost1,1, [9,7,5,10]) == astar(cost1, heuristic1, 1, [9,7,5,10]))

#dfstestcheck() #uncomment if you want to check what ur code prints
dfstest()	
astartest()	
ucstest()
print()
print("Now, for the final test...")
a_start_ucs_combo()
