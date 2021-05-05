import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf
from pathlib import Path
import os
import mplfinance as mpf
from time import sleep
from tqdm import tqdm

download_link = os.path.join(Path.home(),"Downloads","Stocks_test.csv")

print(f'{"Welcome":=^100}')
list_of_stocks = re.sub(r" ","",input("Enter the tickers here, separated by ',': "))
list_of_stocks.upper()
lists_of_stocks = re.split(r",",list_of_stocks)
list_of_stocks = re.sub(r","," ",list_of_stocks)
print("You have requested for the following stocks:")
print(list_of_stocks)
df_stock = yf.download(list_of_stocks,period="max",interval='1d',auto_adjust=True).pct_change().dropna()
df_stocks = df_stock["Close"]
df_stocks.to_csv(download_link,sep=",")

def geomean(returns_arr=df_stocks):
    try:
        n, r = returns_arr.size, np.prod(returns_arr + 1)
    except:
        n, r = np.divide(returns_arr.size, len(returns_arr)), np.prod(returns_arr + 1, axis=1)
    finally:
        r1 = np.power(r, (1. / n)) - 1
    return r1 * 252

def minimize_variance():
    def objective_stocks(x):
        variance = np.matmul(x.T, (np.matmul(cov_var, x)))
        return variance
    
    def constrains(x):
        return 1.0-np.sum(x)
    
    weights_guess = [1/cov_var.shape[0] for stock_tickers in range(cov_var.shape[0])]
    con_1 = {'type':'eq','fun':constrains}
    boundaries = (0.0, 1.0)
    stock_bnd = [boundaries for i in range(cov_var.shape[0])]
    
    solution_stock = minimize(objective_stocks, x0=weights_guess, bounds=stock_bnd, method='SLSQP', constraints=con_1)
    return solution_stock

means_arr = geomean(returns_arr=df_stocks)  # Annualized

if len(lists_of_stocks)>1:
    cov_var = df_stocks.cov()*252  # Annualized
elif len(lists_of_stocks)==1:
    mpf.plot(df_stock[["Open","High","Low","Close","Volume"]], type='candle', style='charles',title='',ylabel='',ylabel_lower='',volume=True,mav=(10))
    exit()

def calculateportfolio(weights):
    returns = np.matmul(weights.T, means_arr)
    variance = np.matmul(weights.T, (np.matmul(cov_var, weights)))
    standard_deviation = np.sqrt(variance)
    return returns, standard_deviation

optimal_weights = minimize_variance()
optimal_results = calculateportfolio(optimal_weights.x)

def simulate(counter=100000):
    def temp_simulate():
        a = np.random.rand(cov_var.shape[0])
        weights = a/np.sum(a)
        return weights
    # a = np.random.rand(counter,cov_var.shape[0])
    # b = np.sum(a, axis=1)
    # b = np.resize(b, (cov_var.shape[0], counter)).T
    # c = a/b
    simulated_lst = []
    print()
    print("Running Simulation for Efficient Frontier:")
    for i in tqdm(range(counter)):
        simulated_lst.append(calculateportfolio(temp_simulate()))
    simulated_lst = np.asarray(simulated_lst)
    return simulated_lst

fig, ax = plt.subplots()
fig.suptitle("The efficient Frontier for {}".format(", ".join(lists_of_stocks)))
ax.set_xlabel("Risk")
ax.set_ylabel("Return (%)")
simulated_arr = simulate()
sns.scatterplot(x=simulated_arr[:,1],y=simulated_arr[:,0])
ax.scatter(optimal_results[1],optimal_results[0],linewidth=15,marker="+")
plt.show()
df_results = pd.DataFrame(list(zip(lists_of_stocks,optimal_weights.x)),columns=["Tickers","Recommended Allocation"]).sort_values(["Recommended Allocation"],ascending=False).reset_index(drop=True)
df_results.to_csv(os.path.join(Path.home(),"Downloads","Stocks_test_weights.csv"), index=False)
print()
print()
print("The Recommended Allocation for the Stocks Are:")
print(df_results)
print(f'{"End":=^100}')