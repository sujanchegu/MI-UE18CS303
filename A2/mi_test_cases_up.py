# I take no responsibility for the correctness of the test cases.
# Test cases may included goals unreachable from start point, in which case the result for each search is an empty list
# ~ST let me know if any of the test cases are wrong :3
import importlib
import os


temp = importlib.import_module('src.PESU-MI_0316_1286_2057')
locals()['UCS_Traversal'] = temp.UCS_Traversal
locals()['DFS_Traversal'] = temp.DFS_Traversal
locals()['A_star_Traversal'] = temp.A_star_Traversal
locals()['tri_traversal'] = temp.tri_traversal


file = open(os.path.join(os.getcwd(), "mi_test_cases_up.txt"), "r")
test_num = 1
failed = []
file.readline()
file.readline()
file.readline()
for i in range(10):
    file.readline()
    size = int(file.readline())
    file.readline()
    cost = [list(map(int, file.readline().split())) for x in range(size)]
    for j in range(100):
        file.readline()
        file.readline()
        file.readline()
        heuristic = list(map(int, file.readline().split()))
        file.readline()
        start_point = int(file.readline())
        file.readline()
        goals = list(map(int, file.readline().split()))
        file.readline()
        correct_answer = [list(map(int, s.split())) for s in file.readline().split(",")]
        correct_answer.pop()
        your_answer = tri_traversal(cost, heuristic, start_point, goals)
        if(your_answer == correct_answer):
            print("Test",test_num,": PASSED!")
        else:
            print("Test",test_num,": FAILED!")
            print("Your answer :", your_answer)
            print("Correct answer :", correct_answer)
            failed.append(test_num)
        test_num += 1
    file.readline()
    print("================================================================")

if(not failed):
    print("ALL TEST CASES PASSED!")
else:
    print("ono FOLLOWING TEST CASES FAILED!")
    print(*failed)
