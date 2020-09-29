# from src.MI_0316_1286_2057 import *
import importlib


temp = importlib.import_module('src.PESU-MI_0316_1286_2057')
locals()['UCS_Traversal'] = temp.UCS_Traversal
locals()['DFS_Traversal'] = temp.DFS_Traversal
locals()['A_star_Traversal'] = temp.A_star_Traversal
locals()['tri_traversal'] = temp.tri_traversal


def test_case():
    '''size of cost matrix is 11x11
    0th row and 0th column is ALWAYS 0
    Number of nodes is 10
    size of heuristic list is 11
    0th index is always 0'''

    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]

    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[0] == [1, 2, 3, 4, 7]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")

    answer = (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[0]
    print(f"{answer=}")

    try:
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[1] == [1, 5, 4, 7]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")

    try:
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[2] == [1, 5, 4, 7]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")


test_case()


cost = [
        # #, A,  B, C
        [0,  0,  0, 0],   # #
        [0,  0,  5, 10],  # A
        [0, -1,  0, 5],   # B
        [0, -1, -1, 0]    # C
       ]
heuristic = [0, 5, 7, 3]
goals = [3]

answer = tri_traversal(cost, heuristic, 1, goals)[1]
if answer == [1, 3]:
    print("SAMPLE TEST CASE 4 FOR THE  UCS_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 4 FOR THE  UCS_TRAVERSAL FAILED")
# print(f"Answer: {answer=}")  # Caution this is a feature of Python 3.8



# This Test case fails without the lexicographical order check
# added to UCS code
cost = [
        # #, A, B,  C
        [0,  0, 0,  0],  # #
        [0,  0, 10, 5],  # A
        [0, -1, 0, -1],  # B
        [0, -1, 5,  0]   # C
       ]
heuristic = [0, 5, 7, 3]
goals = [2]

answer = tri_traversal(cost, heuristic, 1, goals)[1]
if answer == [1, 2]:
    print("SAMPLE TEST CASE 5 FOR THE  UCS_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 5 FOR THE  UCS_TRAVERSAL FAILED")
# print(f"Answer: {answer=}")  # Caution this is a feature of Python 3.8



# Backtracking example to test the working of the new DFS code
cost = [
        #    A,  B,  C,  D
        [0,  0,  0,  0,  0],   # #
        [0,  0,  10, 5, -1],   # A
        [0, -1,  0, -1, 20],   # B
        [0, -1, -1,  0, -1],   # C
        [0, -1, -1, -1,  0]    # D
       ]
heuristic = [0, 5, 7, 3, 9]
goals = [3]

answer = tri_traversal(cost, heuristic, 1, goals)[1]
if answer == [1, 3]:
    print("SAMPLE TEST CASE 6 FOR THE  DFS_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 6 FOR THE  DFS_TRAVERSAL FAILED")
# print(f"Answer: {answer=}")  # Caution this is a feature of Python 3.8



cost = [
        [0,  0,  0,  0],
        [0,  0,  5, 10],
        [0, -1,  0,  5],
        [0, -1, -1,  0]
       ]
heuristic = [0, 0, 0, 0]
goals = [3]

answerDFS, answerUCS, answerA = tri_traversal(cost, heuristic, 1, goals)
if answerA == [1, 3]:
    print("SAMPLE TEST CASE 7 FOR THE  A_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 7 FOR THE  A_TRAVERSAL FAILED")
# print(f"Answer: {answerA=}")  # Caution this is a feature of Python 3.8

if answerUCS == [1, 3]:
    print("SAMPLE TEST CASE 8 FOR THE  UCS_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 8 FOR THE  UCS_TRAVERSAL FAILED")
# print(f"Answer: {answerUCS=}")  # Caution this is a feature of Python 3.8

if answerDFS == [1, 2, 3]:
    print("SAMPLE TEST CASE 8 FOR THE  DFS_TRAVERSAL PASSED")
else:
    print("SAMPLE TEST CASE 8 FOR THE  DFS_TRAVERSAL FAILED")
# print(f"Answer: {answerDFS=}")  # Caution this is a feature of Python 3.8

def test_case_start_is_goal():
    '''size of cost matrix is 11x11
    0th row and 0th column is ALWAYS 0
    Number of nodes is 10
    size of heuristic list is 11
    0th index is always 0'''

    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]

    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [1]))[0] == [1]:
            print("SAMPLE TEST CASE 1G FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1G FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1G FOR THE DFS_TRAVERSAL FAILED")

    answer = (tri_traversal(cost,heuristic, 1, [1]))[0]
    print(f"{answer=}")

    try:
        if (tri_traversal(cost,heuristic, 1, [1]))[1] == [1]:
            print("SAMPLE TEST CASE 2G FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2G FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2G FOR THE UCS_TRAVERSAL FAILED")

    try:
        if (tri_traversal(cost,heuristic, 1, [1]))[2] == [1]:
            print("SAMPLE TEST CASE 3G FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3G FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3G FOR THE A_star_TRAVERSAL FAILED")


test_case_start_is_goal()
