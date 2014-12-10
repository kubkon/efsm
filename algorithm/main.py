import os
import sys
import numpy as np

try:
    import algorithm.internal as internal
    from algorithm.dists import SupportedDistributions, py_get_distribution
except ImportError:
    raise Exception("No module algorithm.internal. Perhaps you forgot to run 'make'?")

class Params:
    def __init__(self, loc, scale, a, b, dist_id):
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
        self.dist_id = dist_id

class EFSM:
    def __init__(self, params, granularity=10000):
        # Infer number of bidders
        self.num_bidders = len(params)
        # Save distribution parameters
        self.params = params
        # Set solution granularity
        self.granularity = granularity
        # Populate cdfs of cost distributions
        self.cdfs = []
        for param in params:
            self.cdfs.append(py_get_distribution(param.dist_id)(param.loc, param.scale, param.a, param.b))
        # Calculate upper bound on bids
        self.b_upper = self._upper_bound_bids()

    def solve(self):
        conv_param = self._estimate_convergence_param()
        if not conv_param:
            raise Exception("Algorithm failed to converge. Could not estimate 'conv_param'.")
        return self._run_ode_solver(conv_param)

    def verify_sufficiency(self, costs, bids, step=100):
        # Sample bidding space
        sample_indices = np.arange(0, costs.shape[1], step)
        m = sample_indices.size
        # Initialize results arrays
        sampled_bids = np.empty((self.num_bidders, m), dtype=np.float)
        sampled_costs = np.empty((self.num_bidders, m), dtype=np.float)
        best_responses = np.empty((self.num_bidders, m), dtype=np.float)
        for i in np.arange(self.num_bidders):
            z = 0
            for j in sample_indices:
                # Get current cost
                cost = costs[i][j]
                # Populate sampled bids and costs
                sampled_bids[i][z] = bids[j]
                sampled_costs[i][z] = cost
                # Tabulate space of feasible bids for each sampled cost
                feasible_bids = np.linspace(cost, self.b_upper, 100)
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
                        for jj in np.arange(self.num_bidders):
                            if jj == i:
                                continue
                            probability *= (1 - self.cdfs[jj].cdf(costs[jj][corr_bid]))
                        utility[k] *= probability
                best_responses[i][z] = feasible_bids[np.argmax(utility)]
                z += 1
        return sampled_bids, sampled_costs, best_responses
    
    def _upper_bound_bids(self):
        """Returns an estimate on upper bound on bids.
        """
        # tabulate the range of permissible values
        num = 10000
        vals = np.linspace(self.params[0].b, self.params[1].b, num)
        tabulated = np.empty(num, dtype=np.float)
        # solve the optimization problem in Eq. (1.8) in the thesis
        for i in np.arange(num):
            v = vals[i]
            probs = 1
            for j in np.arange(1, self.num_bidders):
                probs *= 1 - self.cdfs[j].cdf(v)
            tabulated[i] = (v - self.params[0].b) * probs

        return vals[np.argmax(tabulated)]

    def _estimate_convergence_param(self):
        # approximate
        param = 1e-6
        while True:
            if param > 1e-4:
                return None
            try:
                bids, costs = self._run_ode_solver(param)
            except Exception:
                param += 1e-6
                continue
            # verify sufficiency
            step = bids.size // 35
            sampled_bids, _, best_responses = self.verify_sufficiency(costs, bids, step=step)
            # calculate average error
            errors = []
            m = sampled_bids.shape[1]
            for i in range(self.num_bidders):
                error = 0
                for b,br in zip(sampled_bids[i], best_responses[i]):
                    error += abs(b-br)
                errors.append(error / m)
            # Check if average is low for each bidder
            if np.all([e < 1e-2 for e in errors]):
                break
            # Update param
            param += 1e-6
        return param

    def _run_ode_solver(self, conv_param):
        # set initial conditions for the FSM algorithm
        low = self.params[1].a
        high = self.b_upper
        epsilon = 1e-6
        cond1 = np.empty(self.granularity, dtype=np.bool)
        cond2 = np.empty(self.granularity, dtype=np.bool)
        cond3 = np.empty(self.granularity-1, dtype=np.bool)
        # run the FSM algorithm until the estimate of the lower bound
        # on bids is found
        while high - low > epsilon:
            guess = 0.5 * (low + high)
            bids = np.linspace(guess, self.b_upper-conv_param, num=self.granularity, endpoint=False)
            # solve the system
            try:
                costs = internal.solve(self.params, bids).T
            except Exception:
                if conv_param >= 1e-3:
                    raise Exception("Exceeded maximum iteration limit.")
                conv_param += 1e-6
                continue
            # modify array of lower extremities to account for the bidding
            # extension
            initial = costs[0,:]
            for i in np.arange(self.num_bidders):
                for j in np.arange(self.granularity):
                    x = costs[i][j]
                    cond1[j] = initial[i] <= x and x <= self.b_upper
                    cond2[j] = bids[j] > x
            for i in np.arange(1, self.granularity):
                cond3[i-1] = bids[i-1] < bids[i]
            if np.all(cond1) and np.all(cond2) and np.all(cond3):
                high = guess
            else:
                low = guess
        try:
            return bids, costs
        except UnboundLocalError:
            raise Exception("Algorithm failed to converge.")
        
