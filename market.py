import numpy as np


class Market:
    """
    A class that models an instance of a Fisher market
    """
    def __init__(self, consumers, firms, num_commods):
        self.consumers = consumers
        self.firms = firms
        self.num_commods = num_commods


    def get_demand(self, prices):
        """
        Function that calculates the total demand (i.e. for all consumers) for commodities at given prices
        """
        demand = np.zeros(self.num_commods)
        for consumer in self.consumers:
            demand = demand + consumer.get_demand(prices)

        return demand
    
    def get_excess(self, prices):
        """
        Function that calculates the excess demand at given prices
        """
        return self.get_demand(prices) - 1

    def tatonnement(self, learning_rate):
        """
        Function that runs the tatonnement process
        """
        # Initialize prices uniformly 
        prices = np.repeat(1/self.num_commods, self.num_commods)*100
        excess = self.get_excess(prices)
        iter = 0

        while (np.sum(np.abs(excess)) > 0.01 and iter <= 1000):

            if (iter % 100 == 0):
                print(f"ITERATION {iter}")
                print(f"\tThe excess demand is:\n\t{excess}\n\tPrices are:\n\t{prices}")
                print(f"\tSum of the excess demands {np.sum(np.abs(excess))}")

            excess = self.get_excess(prices)
            prices += learning_rate*excess
            prices = np.clip(prices, 0, 10000) + 0.001
            prices = prices/np.sum(prices)*10000
            prices = np.round(prices, decimals = 3) 
            iter += 1
            
        return prices