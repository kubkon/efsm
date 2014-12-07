import numpy as np
import scipy.stats as stats
from algorithm.util import upper_bound_bids
from algorithm.main import solve


def verify_sufficiency(costs, bids, b_upper, cdfs, step=100):
    # Infer number of bidders
    n = costs.shape[0]
    # Sample bidding space
    sample_indices = np.arange(0, costs.shape[1], step)
    m = sample_indices.size
    # Initialize results arrays
    sampled_bids = np.empty((n, m), dtype=np.float)
    best_responses = np.empty((n, m), dtype=np.float)

    for i in np.arange(n):
        z = 0

        for j in sample_indices:
            # Get current cost
            cost = costs[i][j]
            # Populate sampled bids
            sampled_bids[i][z] = bids[j]
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

    return sampled_bids, best_responses

def estimate_param(lowers, uppers):
    # get number of bidders
    n = lowers.size

    # estimate upper bound on bids
    b_upper = upper_bound_bids(lowers, uppers)

    # approximate
    param = 1e-6

    while True:
        if param > 1e-4:
            return None

        try:
            bids, costs = solve(lowers, uppers, param=param)
        except Exception:
            param += 1e-6
            continue

        # verify sufficiency
        cdfs = []
        for l,u in zip(lowers, uppers):
          cdfs.append(stats.uniform(loc=l, scale=u-l))

        step = len(bids) // 35
        sampled_bids, best_responses = verify_sufficiency(costs, bids, b_upper, cdfs, step=step)

        # calculate average error
        errors = []
        m = sampled_bids.shape[1]
        for i in range(n):
            error = 0
            for b,br in zip(sampled_bids[i], best_responses[i]):
                error += abs(b-br)

            errors.append(error / m)

        # Check if average is low for each bidder
        if all([e < 1e-2 for e in errors]):
            break

        # Update param
        param += 1e-6

    return param
