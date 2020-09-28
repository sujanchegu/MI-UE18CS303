# from Assignment2 import *
import importlib
temp = importlib.import_module('src.PESU-MI_0316_1286_2057')
locals()['UCS_Traversal'] = temp.UCS_Traversal
locals()['DFS_Traversal'] = temp.DFS_Traversal
locals()['A_star_Traversal'] = temp.A_star_Traversal
locals()['tri_traversal'] = temp.tri_traversal


cost=[[0,0,0,0,0,0,0,0,0,0,0],[0,0,6,-1,-1,-1,3,-1,-1,-1,-1],[0,6,0,3,2,-1,-1,-1,-1,-1,-1],[0,-1,3,0,1,5,-1,-1,-1,-1,-1],[0,-1,2,1,0,8,-1,-1,-1,-1,-1],[0,-1,-1,5,8,0,-1,-1,-1,5,5],[0,3,-1,-1,-1,-1,0,1,7,-1,-1],[0,-1,-1,-1,-1,-1,1,0,-1,3,-1],[0,-1,-1,-1,-1,-1,7,-1,0,2,-1],[0,-1,-1,-1,-1,5,-1,3,2,0,3],[0,-1,-1,-1,-1,5,-1,-1,-1,3,0]]
heuristic=[0,10,8,5,7,3,6,5,3,1,0]
goals=[10]
ele=tri_traversal(cost, heuristic,1, goals)
if ele[0]==[1, 2, 3, 4, 5, 9, 10]:
    print("passed dfs testcase 1")
else:
    print("failed dfs testcase 1")
if ele[2]==[1,6,7,9,10]:
    print("passed A* testcase 1")
else:
    print("A* FAILED testcase 1")
if ele[1]==[1,6,7,9,10]:
    print("passed UCS testcase 1")
else:
    print("UCS FAILED testcase 1")

cost = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 6, -1, -1, -1],
    [0, 6, 0, 3, 3, -1],
    [0, -1, 3, 0, 1, 7],
    [0, -1, 3, 1, 0, 8],
    [0, -1, -1, 7, 8, 0],
]
heuristic = [0, 10, 8, 7, 7, 3]
goals = [5]
ele = tri_traversal(cost, heuristic, 1, goals)

if ele[2] == [1, 2, 3, 5]:
    print("passed A* testcase2")
else:
    print("A* FAILED testcase 2")
if ele[0]==[1,2,3,4,5]:
    print("passed DFS testcase 2")
else:
    print("FAILED DFS testcase 2")
if ele[1] == [1, 2, 3, 5]:
    print("passed UCS testcase2")
else:
    print("UCS FAILED testcase 2")

cost=[ [0,0,0,0,0,0,0],
       [-1,0,1,-1,-1,-1,10],
       [-1,-1,0,2,1,-1,-1],
       [-1,-1,-1,0,-1,5,-1],
       [-1,-1,-1,-1,0,5,4],
       [-1,-1,-1,-1,-1,0,2],
       [-1,-1,-1,-1,-1,-1,-1]
    ]
heuristic=[0,5,3,4,2,6,0]
goals=[6]
ele = tri_traversal(cost, heuristic, 1, goals)
if ele[0]==[1, 2, 3, 5, 6]:
    print("DFS passed testcase 3")
else:
    print("DFS FAILED testcase 3")
if ele[2]==[1,2,4,6]:
    print("A* passed testcase 3")
else:
    print("A* failed testcase 3")
if ele[1]==[1,2,4,6]:
    print("UCS passed testcase 3")
else:
    print("UCS failed testcase 3")

cost=[[0 if i==j else -1 for j in range(9)] for i in range(9)]
cost[1][3]=3
cost[1][2]=7
cost[3][4]=9
cost[2][5]=6
cost[2][3]=2
cost[4][5]=3
cost[4][8]=13
cost[5][6]=2
cost[5][7]=1
for j in range(9):
    for k in range(9):
        if cost[j][k]>0:
            cost[k][j]=cost[j][k]
heuristic=[0 for i in range(9)]
goals=[7]
ele=ele = tri_traversal(cost, heuristic, 1, goals)
if ele[0]==[1, 2, 3, 4, 5, 7]:
    print("DFS testcase 4 passed")
else:
    print("DFS testcase 4 failed")
if ele[1]==[1,3,2,5,7]:
    print("UCS testcase 4 passed")
else:
    print("UCS testcase 4 failed",ele[1])
goals=[8]
ele=tri_traversal(cost, heuristic, 1, goals)
if ele[0]==[1,3,4,8]:
    print("DFS of testcase 5 passed")
if ele[1]==[1,3,4,8]:
    print("UCS testcase 5 passed ")
else:
    print("UCS testcase 5 failed")
cost=[[-1 for j in range(6)] for i in range(6)]
goals=[5]
cost[1][2]=1
cost[1][3]=4
cost[2][3]=2
cost[2][4]=5
cost[2][5]=12
cost[3][4]=2
cost[4][5]=3

# for row in cost:
#     for c in row:
#         print(f"{c:^4}", end="")
#     print()


heuristic=[0,7,6,2,1,0]
ele=tri_traversal(cost, heuristic, 1, goals)
if ele[0]==[1,2,3,4,5]:
    print("DFS testcase 6 passed")
else:
    print("DFS testcase 6 FAILED")
if ele[2]==[1,3,4,5]:
    print("A* passed testcase 6")
else:
    print("A* FAILED testcase 6")
    print("You got:", ele[2])
if ele[1]==[1,2,3,4,5]:
    print("UCS testcase 6 passed")
else:
    print("UCS testcase 6 FAILED")
cost=[[-1 for j in range(9)] for i in range(9)]
goals=[8]
cost[1][3]=4
cost[1][4]=3
cost[3][6]=12
cost[3][7]=5
cost[4][5]=7
cost[5][6]=2
cost[4][6]=10
cost[7][8]=16
cost[6][8]=5
heuristic=[0,14,0,12,11,6,4,11,0]
ele=tri_traversal(cost, heuristic, 1, goals)
if ele[0]==[1,3,6,8]:
    print("DFS passed testcase 7")
else:
    print("DFS failed testcase 7")
if ele[2]==[1,4,5,6,8]:
    print("A* passed testcase 7")
else:
    print("A* failed testcase 7")

if ele[1]==[1,4,5,6,8]:
    print("UCS passed testcase 7")
else:
    print("UCS failed testcase 7")


cost=[[-1 for j in range(7)] for i in range(7)]
cost[1][2]=5
cost[2][3]=8
cost[3][4]=6
cost[3][1]=2
cost[4][1]=7
cost[4][6]=3
cost[4][5]=2
cost[5][6]=1

# for row in cost:
#     for c in row:
#         print(f"{c:^4}", end="")
#     print()

goals=[6]
heuristic=[0,2,1,8,9,5,3]
ele=tri_traversal(cost, heuristic, 1, goals)
if ele[1]==[1,2,3,4,6]:
    print("passed testcase 8 UCS")
else:
    print("FAILED testcase 8 UCS")
    print("You got:", ele[1])
if ele[0]==[1,2,3,4,5,6]:
    print("passed testcase 8 DFS")
else:
    print("failed testcase 8 DFS")
if ele[2]==[1,2,3,4,6]:
    print("passed testcase 8 A*")
else:
    print("FAILED testcase 8 A*")

    


