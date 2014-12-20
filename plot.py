from itertools import cycle, chain
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from algorithm.main import EFSM, Params, Uniform, TruncatedNormal


rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# Set the scenario
# params = [
#     Params(0, 0, 0.0625, 0.8125, Uniform),
#     Params(0, 0, 0.125,  0.875,  Uniform),
#     Params(0, 0, 0.1875, 0.9375, Uniform)
# ]
# params = [
#     Params(0, 0, 0.1125, 0.6625, Uniform),
#     Params(0, 0, 0.225,  0.775,  Uniform),
#     Params(0, 0, 0.3375, 0.8875, Uniform)
# ]
# params = [
#     Params(0.4375, 0.1875, 0.0625, 0.8125, TruncatedNormal),
#     Params(0.5,    0.1875, 0.125,  0.875,  TruncatedNormal),
#     Params(0.5625, 0.1875, 0.1875, 0.9375, TruncatedNormal)
# ]
# params = [
#     Params(0.3875, 0.1375, 0.1125, 0.6625, TruncatedNormal),
#     Params(0.5,    0.1375, 0.225,  0.775,  TruncatedNormal),
#     Params(0.6125, 0.1375, 0.3375, 0.8875, TruncatedNormal)
# ]
# params = [
#     Params(0,   0,      0.0625, 0.8125, Uniform),
#     Params(0.5, 0.1875, 0.125,  0.875,  TruncatedNormal),
#     Params(0,   0,      0.1875, 0.9375, Uniform)
# ]
# params = [
#     Params(0,   0,      0.1125, 0.6625, Uniform),
#     Params(0.5, 0.1375, 0.225,  0.775,  TruncatedNormal),
#     Params(0,   0,      0.3375, 0.8875, Uniform)
# ]
params = [
    Params(6.9375, 0.6875, 5.5625, 8.3125, TruncatedNormal),
    Params(7.5,    0.6875, 6.125,  8.875,  TruncatedNormal),
    Params(8.0625, 0.6875, 6.6875, 9.4375, TruncatedNormal)
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
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels = ['Bidder {}'.format(i) for i in range(1, n+1)]
plt.legend(labels, loc='upper left')
plt.savefig('approximation.pdf')

plt.figure()
sts = cycle(styles)
clss = cycle(colors)
for c, sc, sb in zip(costs, s_costs, s_bids):
  plt.plot(c, bids, next(sts))
  plt.plot(sc, sb, next(clss))
plt.grid()
plt.xlabel(r"Cost, $c_i$")
plt.ylabel(r"Bid, $b_i$")
labels_1 = ['Bidder {}'.format(i) for i in range(1, n+1)]
labels_2 = ['Bidder {}: Best response'.format(i) for i in range(1, n+1)]
labels = list(chain.from_iterable(zip(labels_1, labels_2)))
plt.legend(labels, loc='upper left')
plt.savefig('sufficiency.pdf')

for i, c, sc, sb in zip(range(1, n+1), costs, s_costs, s_bids):
  plt.figure()
  plt.plot(c, bids, 'b')
  plt.plot(sc, sb, 'r.')
  plt.grid()
  plt.xlabel(r"Cost, $c_i$")
  plt.ylabel(r"Bid, $b_i$")
  plt.savefig('sufficiency_{}.pdf'.format(i))
