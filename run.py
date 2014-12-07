import argparse
import numpy as np

from util import estimate_param
from algorithm.main import solve

parser = argparse.ArgumentParser(description="Estimate param for EFSM")
parser.add_argument('w', type=float, help='Price weight')
parser.add_argument('reps', nargs='+', type=float, help='Reputation array')
args = parser.parse_args()

# parse scenario params
w = args.w
reputations = np.array(args.reps)
n = reputations.size

# estimate param
param = estimate_param(w, reputations)
if not param:
    print("Could not estimate param")

# approximate
bids, costs = solve(w, reputations, param=param)

print("Estimated lower bound on bids: %r" % bids[0])

# save the results in a file
with open('efsm.out', 'wt') as f:
    labels = ['w', 'reps', 'bids'] + ['costs_{}'.format(i) for i in range(n)]
    labels = ' '.join(labels)
    values = [w, reputations.tolist(), bids.tolist()] + [c.tolist() for c in costs]
    values = ' '.join(map(lambda x: repr(x).replace(' ', ''), values))
    f.write(labels)
    f.write('\n')
    f.write(values)
