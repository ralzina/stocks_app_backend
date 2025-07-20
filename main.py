from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from models.time_series import run_time_series
from models.decision_tree import run_decision_tree
from models.monte_carlo import run_monte_carlo
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stockdata")
def get_stock_data(
    ticker: str = Query(...), 
    period: str = Query("6mo"),
    days: int = Query(30)
):
    
    '''
    Fetches stock data for a given ticker symbol and period.
    :param ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    :param period: Period for which to fetch data (default is '1mo').
    :return: JSON response containing stock data.
    '''
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    data = hist.reset_index()

    data['Date'] = data['Date'].astype(str).str[:10]
    data['Date'] = pd.to_datetime(data['Date'])

    ts_forecast, ts_mape = run_time_series(data)
    dt_accuracy, dt_prediciton = run_decision_tree(data)
    mc_simulations, mc_histogram, mc_lower_95_CI, mc_upper_95_CI = run_monte_carlo(df=data, days=days)

    '''
    ts_forecast['Date'] = ts_forecast['Date'].astype(str)
    ts_forecast['Predicted_Close'] = ts_forecast['Predicted_Close'].astype(float)

    ts_forecast = ts_forecast.to_dict(orient='records')
    '''

    ts_mape = float(ts_mape)
    dt_accuracy = float(dt_accuracy)
    dt_prediciton = int(dt_prediciton)

    # mc_simulations = mc_simulations.tolist()

    response = {
        "ts_forecast": ts_forecast,
        "ts_mape": ts_mape,
        "mc_simulations": mc_simulations,
        "mc_histogram": mc_histogram,
        "mc_lower_95_CI": mc_lower_95_CI,
        "mc_upper_95_CI": mc_upper_95_CI,
        "dt_accuracy": dt_accuracy,
        "dt_prediction": dt_prediciton
    }

    return JSONResponse(content=response)
    