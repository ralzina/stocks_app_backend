from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
from models.time_series import run_time_series
from models.decision_tree import run_decision_tree
from models.monte_carlo import run_monte_carlo
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

client = Groq(api_key=GROQ_API_KEY)

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

    try:
        prompt = f"""
        You are a helpful financial assistant. 
        
        Give your own personal opinion of the current status of this stock.
        
        Don't write this as a list, but rather a concise continuous paragraph, using beginner friendly language.
        Don't name the specific names of ts_forecast, dt_accuracy, etc. Just summarize the results.
        ts_forecast means time series forecast, containing historical data along with predictions.
        ts_mape is the MAPE for the time series.
        mc_simulations is all the simulations of the montecarlo model.
        mc_histogram is the histogram of the final day of all the simulations.
        mc_lower_95_CI and mc_upper_95_CI are the 95% confidence intervals of the monte carlo model.
        dt_accuracy is the accuracy of the decision tree model.
        dt_prediction is the predicted closing price for the next day from the decision tree model, where -1 is sell, 0 is hold, and 1 is buy

        Stock model JSON:
        {json.dumps(response)}
        """

        gpt_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        summary = gpt_response.choices[0].message.content

    except Exception as e:
        summary = f"Could not generate summary: {str(e)}"

    response["summary"] = summary

    return JSONResponse(content=response)    