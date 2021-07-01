import consumerUtility as cu
import numpy as np



def  tatonnement(prices_0, valuations, budgets, utility_params, num_iters, learning_rate):
        """
        Function that runs the tatonnement process
        """
        iter = 0
        prices = np.copy(prices_0) 
        prices_hist = []
        obj_hist = []

        for iter in range(num_iters):
            prices_hist.append(prices)

            if (iter % 10 == 0):
                print(f"ITERATION {iter}\n")
            
            demands = np.zeros(valuations.shape)
            
            obj = np.sum(prices)
            for buyer in range(budgets.shape[0]):    
                if (utility_params[buyer] == -np.inf):
                    demands[buyer,:] = cu.get_leontief_marshallian_demand(prices, budgets[buyer], valuations[buyer,:])
                    util_val = budgets[buyer] * np.log(cu.get_leontief_indirect_util(prices, 1, valuations[buyer,:]))
                    obj += util_val
                elif (utility_params[buyer] == 0):
                    demands[buyer,:] = cu.get_CD_marshallian_demand(prices, budgets[buyer], valuations[buyer,:])
                    util_val = budgets[buyer] * np.log(cu.get_CD_indirect_util(prices, 1, valuations[buyer,:]))
                    obj += util_val
                elif (utility_params[buyer] == 1):
                    demands[buyer,:] = cu.get_linear_marshallian_demand(prices, budgets[buyer], valuations[buyer,:])
                    util_val = budgets[buyer] * np.log(cu.get_linear_indirect_util(prices, 1, valuations[buyer,:]))
                    obj += util_val
                else:
                    rho = utility_params[buyer]
                    demands[buyer,:] = cu.get_ces_marshallian_demand(prices, budgets[buyer], valuations[buyer,:], rho)
                    util_val = budgets[buyer] * np.log(cu.get_ces_indirect_util(prices, 1, valuations[buyer,:], rho))
                    # print(f"ces = {util_val}")
                    obj += util_val
            obj_hist.append(obj)
            # demands_hist.append(demands)
            demand = np.sum(demands, axis = 0)
            excess = demand - 1
            prices *= np.exp(excess/learning_rate)
            
        return (prices, prices_hist, obj_hist)