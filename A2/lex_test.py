# from A2 import *
import importlib
temp = importlib.import_module('src.PESU-MI_0316_1286_2057')
locals()['UCS_Traversal'] = temp.UCS_Traversal
locals()['DFS_Traversal'] = temp.DFS_Traversal
locals()['A_star_Traversal'] = temp.A_star_Traversal

def test_case1():
    cost = [
            [0,  0,  0,  0],
            [0,  0,  5, 10],
            [0, -1,  0,  5],
            [0, -1, -1,  0]
           ]

    print("DFS:", DFS_Traversal(cost,1,[3]))
    print("UCS:", UCS_Traversal(cost,1,[3]))
    # print(A_star_Traversal(cost,1,[3]))


def test_case2():
    cost = [
            #    A   B  C   D
            [0,  0,  0, 0,  0],  # #
            [0,  0, -1, 10, 2],  # A
            [0, -1,  0, 5, -1],  # B
            [0, -1, -1, 0, -1],  # C
            [0, -1, -1, 5,  0]   # D
           ]

    print("DFS", DFS_Traversal(cost,1,[3]))
    print("UCS", UCS_Traversal(cost,1,[3]))
    # print(A_star_Traversal(cost,1,[3]))


test_case1()
print('-'*80)
test_case2()
print('-'*80)


def test_case3():
    cost = [
            #    A   B  C   D
            [0,  0,  0, 0,  0],  # #
            [0,  0, -1, 10, 5],  # A
            [0, -1,  0, 5, -1],  # B
            [0, -1, -1, 0, -1],  # C
            [0, -1, -1, 5,  0]   # D
           ]

    print("DFS", DFS_Traversal(cost,1,[3]))
    print("UCS", UCS_Traversal(cost,1,[3]))
    # print(A_star_Traversal(cost,1,[3]))


test_case3()
