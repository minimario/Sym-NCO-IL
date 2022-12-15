import numpy as np
import pickle

with open('results/tsp/tsp20_test_seed1234/tsp20_test_seed1234-lkh.pkl', 'rb') as f:  
    data = pickle.load(f)

sol_list = []

for i in range(len(data[0])):
    sol = data[0][i][1]
    sol_list.append(sol)

