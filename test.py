#%%
# Import libraries
import tatonnement as t
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)



if __name__ == '__main__':
    # Declare number of goods, consumers and elasticity of substitution (rho)
    num_goods = 5
    num_buyers = 10
    num_iters = 50
    learning_rate = 2.9
    num_experiments = 10000
    obj_hist_avg = np.zeros(num_iters)    
    
    for experiment in range(num_experiments):
        print(f"Experiment {experiment}")
        budgets = np.random.rand(num_buyers) + 2
        valuations = np.random.rand(num_buyers, num_goods) + 2
        util_params = []

        for i in range(num_buyers):   
            util_type = np.random.randint(0,3)
            if(util_type == 0):
                util_params.append(0)
            elif(util_type == 1):
                util_params.append(-np.inf)
            else:
                is_complem = np.random.randint(0,1)
                if(is_complem):
                    util_params.append(-(np.random.rand()*100 + 1))
                else:
                    util_params.append(np.random.rand()/2 +0.25)
        prices_0 = np.random.rand(num_goods) + 2

        prices, prices_hist, obj_hist = t.tatonnement(prices_0, valuations, budgets, util_params, num_iters, learning_rate)
        
        obj_hist_avg += np.array(obj_hist)

    obj_hist_avg /= num_experiments

    x = np.linspace(1, len(obj_hist_avg) +1, len(obj_hist_avg) + 1)
    plt.plot(list(range(len(obj_hist_avg))), obj_hist_avg, color = "red")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-1)) + obj_hist_avg[-1], color='blue', linestyle='dashed', label = "1/t")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-2)) + obj_hist_avg[-1], color='green', linestyle='dashed', label = "1/t^2")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-3)) + obj_hist_avg[-1], color='orange', linestyle='dashed', label = "1/t^3")
    plt.xlabel("Number of iterations")
    plt.yticks([], [])
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.savefig("./graphs/obj_change.png")
    plt.show()

    obj_hist_avg = np.zeros(num_iters)    
    num_not_conv = 0

    for experiment in range(num_experiments):
        print(f"Experiment {experiment}")
        budgets = np.random.rand(num_buyers) + 2
        valuations = np.random.rand(num_buyers, num_goods) + 2
        util_params = []

        for i in range(num_buyers):   
            util_type = np.random.randint(0,4)
            if(util_type == 0):
                util_params.append(0)
            elif(util_type == 1):
                util_params.append(1)
            elif(util_type == 2):
                util_params.append(-np.inf)
            else:
                is_complem = np.random.randint(0,1)
                if(is_complem):
                    util_params.append(-(np.random.rand()*100 + 1))
                else:
                    util_params.append(np.random.rand()/2 +0.25)
        prices_0 = np.random.rand(num_goods) + 2

        prices, prices_hist, obj_hist = t.tatonnement(prices_0, valuations, budgets, util_params, num_iters, learning_rate)
        
        num_not_conv += np.any(np.diff(obj_hist) > 0)
        obj_hist_avg += np.array(obj_hist)

    obj_hist_avg /= num_experiments

    x = np.linspace(1, len(obj_hist_avg) +1, len(obj_hist_avg) + 1)
    plt.plot(list(range(len(obj_hist_avg))), obj_hist_avg, color = "red")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-1)) + obj_hist_avg[-1], color='blue', linestyle='dashed', label = "1/t")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-2)) + obj_hist_avg[-1], color='green', linestyle='dashed', label = "1/t^2")
    plt.plot(x-1, (obj_hist_avg[0] - obj_hist_avg[-1])*(x**(-3)) + obj_hist_avg[-1], color='orange', linestyle='dashed', label = "1/t^3")
    plt.xlabel("Number of iterations")
    plt.yticks([], [])
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.savefig("./graphs/obj_change_with_linear.png")
    plt.show()    

    print(f"Number of experiments not converging {num_not_conv}\Percentage of experiments not converging {num_not_conv/num_experiments}")

# %%
