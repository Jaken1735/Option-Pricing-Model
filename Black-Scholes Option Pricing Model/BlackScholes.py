import numpy as np
from scipy.stats import norms

# Define Variables
r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

def blackScholes(S, K, T, sigma, type='C'):
    """
    Calculate the Black-Scholes Option Price for a put/call
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == 'C':
            price = S*norms.cdf(d1) - K*np.exp(-r*T)*norms.cdf(d2)
        elif type == 'P':
            price = K*np.exp(-r*T)*norms.cdf(-d2) - S*norms.cdf(-d1)
        else:
            raise ValueError('Type must be either "C" or "P"')
    except ValueError as e:
        print(e)
    
    return price

# Calculate the Call and Put Option Prices
call_price = blackScholes(S, K, T, sigma, type='C')
put_price = blackScholes(S, K, T, sigma, type='P')

print(f'Call Price: {call_price}')
print(f'Put Price: {put_price}')


