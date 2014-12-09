from itertools import cycle, chain
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from algorithm.main import EFSM, Params, PyTDist


rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# Set the scenario
params = [
    Params(0, 0, 0.0625, 0.8125, PyTDist.uniform),
    Params(0, 0, 0.125,  0.875,  PyTDist.uniform),
    Params(0, 0, 0.1875, 0.9375, PyTDist.uniform)
]

# Approximate equilibrium bidding strategies
solver = EFSM(params)
bids, costs = solver.solve()

# Verify sufficiency
step = bids.size // 35
_, s_costs, s_bids = solver.verify_sufficiency(costs, bids, step=step)

# Plot
styles = ['b', 'r--', 'g:', 'm-.']
colors = ['b.', 'r.', 'g.', 'm.']
n = len(params)

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
