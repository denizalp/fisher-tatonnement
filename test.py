#%%
# Import libraries
import market as m
import consumer as c
import numpy as np
np.set_printoptions(precision=3)


#%%
# Declare number of goods, consumers and elasticity of substitution (rho)
num_raw_commods = 10 
num_commods = 5
num_firms = 10
num_consumers = 100
rho_consumer = -1

# Create consumers
consumers = []

for i in range(num_consumers):
    budgets = np.random.randint(4, size= 1) + 2
    valuation = np.random.randint(5, size=num_commods) + 2
    consumers.append(c.Consumer(endowment, valuation, rho_consumer))

# Create economy
economy = m.Market(consumers, firms, num_commods)

economy.tatonnement(learning_rate = 0.01)