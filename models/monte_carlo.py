import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import math

def run_monte_carlo(df, simulations=10000, days=30):
    '''
    Run a Monte Carlo simulation to forecast stock prices.
    Parameters:
    - data (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    - simulations (int): Number of simulations to run.
    - days (int): Number of days to forecast.
    '''

    data = df.copy()
    last_date = data["Date"].max()
    data = data[['Close']]

    data['Return'] = data['Close'].pct_change()
    data = data.dropna()

    returns = data['Return']

    mu = returns.mean()
    sigma = returns.std()

    last_price = data['Close'].iloc[-1]
    
    forecast = np.zeros((days, simulations))

    for i in range(simulations):
        prices = [last_price]
        for d in range(days):
            # simulate daily return
            daily_return = np.random.normal(mu, sigma)

            # calculate new price
            price = prices[-1] * (1 + daily_return)
            prices.append(price)
        forecast[:, i] = prices[1:]
    
    mean_prices = forecast.mean(axis=1)
    percentile_5 = np.percentile(forecast, 5, axis=1)
    percentile_95 = np.percentile(forecast, 95, axis=1)
    day_count = np.arange(days)

    chart_data = [
        {
            "day": (last_date + BDay(day)).strftime('%Y-%m-%d'),
            "mean": float(mean),
            "p5": float(p5),
            "p95": float(p95)
        }
        for day, mean, p5, p95 in zip(day_count, mean_prices, percentile_5, percentile_95)
    ]

    final_prices = forecast[-1]
    counts, bin_edges = np.histogram(final_prices, bins=math.floor(math.sqrt(len(final_prices))))
    histogram_data = [
        {
            "price": float(bin_edges[i]),
            "count": int(counts[i])
        }
        for i in range(len(counts))
    ]

    lower_95_CI = np.percentile(final_prices, 2.5)
    upper_95_CI = np.percentile(final_prices, 97.5)

    return chart_data, histogram_data, lower_95_CI, upper_95_CI