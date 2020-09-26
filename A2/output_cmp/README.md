# What are the two files different?
## The reason is in the values of the heuristics...
### Refer to files ```A2/testcasesassignment2.py``` and ```A2/testcase16.py```

The heuristic used in the test case gives a non-zero value for the goal node 9, which is incorrect. So there is a difference
in the output, the previous code would make A* decompose into UCS so it would still get the optimal answer, while the new code, models true A* and returns a sub-optimal path as expected in this case.
Also in the new code we return an empty list, i.e. ```[]```, instead of ```None``` in case of no path from inital state to goal state.


You can run the ```diff``` command between the two output files to get a deeper look at the explanation above.
