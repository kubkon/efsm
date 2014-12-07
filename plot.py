import argparse
import ast
import csv
from itertools import cycle, chain
from functools import partial
import numpy as np
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from algorithm.util import upper_bound_bids

def best_responses(costs, bids, b_upper, cdfs, step=100):
    # Infer number of bidders
    n = costs.shape[0]
    # Sample bidding space
    sample_indices = np.arange(0, costs.shape[1], step)
    m = sample_indices.size
    # Initialize results arrays
    sampled_costs = np.empty((n, m), dtype=np.float)
    best_responses = np.empty((n, m), dtype=np.float)

    for i in np.arange(n):
        z = 0

        for j in sample_indices:
            # Get current cost
            cost = costs[i][j]
            # Populate sampled costs
            sampled_costs[i][z] = cost
            # Tabulate space of feasible bids for each sampled cost
            feasible_bids = np.linspace(cost, b_upper, 100)
            n_bids = feasible_bids.size
            # Initialize utility array
            utility = np.empty(n_bids, dtype=np.float)

            for k in np.arange(n_bids):
                # Get currently considered bid
                bid = feasible_bids[k]
                # The upper bound on utility given current cost-bid pair
                utility[k] = bid - cost

                if bid >= bids[0]:
                    # Compute probability of winning
                    corr_bid = np.argmin(np.absolute(bids - np.ones(bids.size) * bid))
                    probability = 1
                    for jj in np.arange(n):
                        if jj == i:
                            continue

                        probability *= (1 - cdfs[jj].cdf(costs[jj][corr_bid]))

                    utility[k] *= probability

            
            best_responses[i][z] = feasible_bids[np.argmax(utility)]
            z += 1

    return sampled_costs, best_responses


csv.field_size_limit(1000000000)

rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

### Parse command line arguments
parser = argparse.ArgumentParser(description="Numerical approximation -- sufficiency analyzer")
parser.add_argument('file_name', help='file with approximation results')
args = parser.parse_args()
file_name = args.file_name

# Read data from file
data_in = {}
with open(file_name, 'rt') as f:
    f_reader = csv.DictReader(f, delimiter=' ')
    for row in f_reader:
        for key in row:
            data_in[key] = row[key]

# Parse data common to FSM and PPM methods
lowers = np.array(ast.literal_eval(data_in['lowers']))
uppers = np.array(ast.literal_eval(data_in['uppers']))
n = lowers.size
bids = np.array(ast.literal_eval(data_in['bids']))
costs = np.array([ast.literal_eval(data_in['costs_{}'.format(i)]) for i in range(n)])

# Estimate upper bound on bids
b_upper = upper_bound_bids(lowers, uppers)

# Verify sufficiency
cdfs = []
for l, u in zip(lowers, uppers):
  cdfs.append(ss.uniform(loc=l, scale=u-l))
step = len(bids) // 35
s_costs, s_bids = best_responses(costs, bids, b_upper, cdfs, step=step)

# Plot
styles = ['b', 'r--', 'g:', 'm-.']
colors = ['b.', 'r.', 'g.', 'm.']

plt.figure()
sts = cycle(styles)
for c in costs:
  plt.plot(c, bids, next(sts))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels = ['Network operator {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')

plt.figure()
sts = cycle(styles)
clss = cycle(colors)
for c, sc, sb in zip(costs, s_costs, s_bids):
  plt.plot(c, bids, next(sts))
  plt.plot(sc, sb, next(clss))
plt.grid()
plt.xlabel(r"Cost-hat, $\hat{c}_i$")
plt.ylabel(r"Bid-hat, $\hat{b}_i$")
labels_1 = ['NO {}'.format(i) for i in range(1, n+1)]
labels_2 = ['NO {}: Best response'.format(i) for i in range(1, n+1)]
labels = list(chain.from_iterable(zip(labels_1, labels_2)))
plt.legend(labels, loc='upper left')
plt.savefig('sufficiency.pdf')

for i, c, sc, sb in zip(range(1, n+1), costs, s_costs, s_bids):
  plt.figure()
  plt.plot(c, bids, 'b')
  plt.plot(sc, sb, 'r.')
  plt.grid()
  plt.xlabel(r"Cost-hat, $\hat{c}_i$")
  plt.ylabel(r"Bid-hat, $\hat{b}_i$")
  plt.savefig('sufficiency_{}.pdf'.format(i))
