import os
import sys
import numpy as np
from algorithm.util import upper_bound_bids
import algorithm.internal as internal


def solve(lowers, uppers, granularity=10000, param=1e-6):
    # infer number of bidders
    n = lowers.size

    # estimate the upper bound on bids
    b_upper = upper_bound_bids(lowers, uppers)

    # set initial conditions for the FSM algorithm
    low = lowers[1]
    high = b_upper
    epsilon = 1e-6
    num = granularity
    cond1 = np.empty(num, dtype=np.bool)
    cond2 = np.empty(num, dtype=np.bool)
    cond3 = np.empty(num-1, dtype=np.bool)

    # run the FSM algorithm until the estimate of the lower bound
    # on bids is found
    while high - low > epsilon:
        guess = 0.5 * (low + high)
        bids = np.linspace(guess, b_upper-param, num=num, endpoint=False)

        # solve the system
        try:
            costs = internal.solve(lowers, uppers, bids).T

        except Exception:
            if param >= 1e-3:
                raise Exception("Exceeded maximum iteration limit.")
            param += 1e-6
            continue

        # modify array of lower extremities to account for the bidding
        # extension
        initial = costs[0,:]

        for i in np.arange(n):
            for j in np.arange(num):
                x = costs[i][j]
                cond1[j] = initial[i] <= x and x <= b_upper
                cond2[j] = bids[j] > x

        for i in np.arange(1, num):
            cond3[i-1] = bids[i-1] < bids[i]

        if np.all(cond1) and np.all(cond2) and np.all(cond3):
            high = guess
        else:
            low = guess

    try:
        return bids, costs

    except UnboundLocalError:
        raise Exception("Algorithm failed to converge.")
        

if __name__ == "__main__":
    param = float(sys.argv[1])

    # set the scenario
    lowers = np.array([0.0125,0.015,0.0375])
    uppers = np.array([0.9625,0.965,0.9875])
    n = lowers.size

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
