import argparse
import ast
import numpy as np
from util import estimate_param
from algorithm.main import solve

parser = argparse.ArgumentParser(description="Estimate param for EFSM")
parser.add_argument('lowers', help='Lower extremities')
parser.add_argument('uppers', help='Upper extremities')
args = parser.parse_args()

# parse scenario params
lowers = np.array(ast.literal_eval(args.lowers))
uppers = np.array(ast.literal_eval(args.uppers))
n = lowers.size

# estimate param
param = estimate_param(lowers, uppers)
if not param:
    print("Could not estimate param")

# approximate
bids, costs = solve(lowers, uppers, param=param)

print("Estimated lower bound on bids: %r" % bids[0])

# save the results in a file
with open('efsm.out', 'wt') as f:
    labels = ['lowers', 'uppers', 'bids'] + ['costs_{}'.format(i) for i in range(n)]
    labels = ' '.join(labels)
    values = [lowers.tolist(), uppers.tolist(), bids.tolist()] + [c.tolist() for c in costs]
    values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
    f.write(labels)
    f.write('\n')
    f.write(values)
