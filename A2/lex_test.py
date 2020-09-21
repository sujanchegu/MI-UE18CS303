from Assignment2 import *
def test_case1():
	cost = [[0,0,0,0],
			[0,0,5,10],
			[0,-1,0,5],
			[0,-1,-1,0]
			]
	print(UCS_Traversal(cost,1,[3]))

def test_case2():
	cost = [[0,0,0,0,0],
			[0,0,0,10,5],
			[0,-1,0,5,0],
			[0,-1,-1,0,0],
			[0,-1,-1,5,0]
			]
	print(UCS_Traversal(cost,1,[3]))

test_case1()
test_case2()