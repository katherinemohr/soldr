import math
from pyscipopt import Model, quicksum

# ISL, OSL, timestamp
requests = [(6758, 500, 0), (7322, 490, 1)]
W = 4

model = Model("optimal_routing")
routing = {i: model.addVar(f"routing[{i}]", vtype="I", lb=0, ub=W-1) for i in range(len(requests))}