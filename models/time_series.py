# Imports
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

def run_time_series(data):
    data = data.drop(columns=['Stock Splits','Dividends'])

    '''
    Graphs and printing statements for debugging

    print(data.head())
    print(data.info())
    print(data.describe())

    # Initial data visualization
    # Plot 1 - Open and Close Prices over time
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'],data['Open'], label="Open", color="blue")
    plt.plot(data['Date'],data['Close'], label="Close", color="red")
    plt.title("Open-Close Price over Time")
    plt.legend()
    # plt.show()

    # Plot 2 - Trading Volume (check for outliers)
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Volume'], label="Volume", color="orange")
    plt.title("Stock Volume over Time")
    # plt.show()

    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    # Plot 3 - Check for correlation between features
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    def cut_data(data, low, high):
        start_index = int(len(data) * low)
        end_index = int(len(data) * high)
        return data.iloc[start_index:end_index]

    prediction = cut_data(data, 0.2, 0.8)

    plt.figure(figsize=(12,6))
    plt.plot(data['Date'],data['Close'], color="blue")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title("Close Price over Time")
    '''

    # Prepare for the LSTM Model (Sequential)
    features = ['Open','High','Low','Close','Volume']
    stock_close = data[features].copy()

    dataset = stock_close.values # convert to numpy array

    training_data_length = int(np.ceil(len(dataset) * 0.80))

    # Preprocessing Stages
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    training_data = scaled_data[:training_data_length] # 95% of all data

    X_train, y_train = [], []

    # Create a sliding window for stock (60 days)
    for i in range(60, len(training_data)):
        X_train.append(training_data[i-60:i]) # Store the last 60 days
        y_train.append(training_data[i][3]) # Store the current day

    X_train, y_train = np.array(X_train), np.array(y_train)

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the model
    model = keras.models.Sequential()

    # First Layer
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))   

    # Second Layer
    model.add(keras.layers.LSTM(64, return_sequences=False))   

    # Third Layer
    model.add(keras.layers.Dense(128, activation='relu'))

    # Fourth Layer
    model.add(keras.layers.Dropout(0.2))

    # Final Output Layer
    model.add(keras.layers.Dense(1))

    model.summary()
    model.compile(optimizer="adam",
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError()])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stop])

    # Prep the test data
    # Start at 95% date - 60 since the model uses 60 days to predict the next and then go till the end with :
    test_data = scaled_data[training_data_length - 60:]
    X_test = []

    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i])

    X_test = np.array(X_test)
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make a Prediction
    predictions = model.predict(X_test)
    # predictions = scaler.inverse_transform(predictions)

    pred_full = np.zeros((len(predictions), 5))
    pred_full[:, 3] = predictions[:, 0]  # only fill the "Close" column
    pred_full = scaler.inverse_transform(pred_full)
    predictions_close = pred_full[:, 3]

    # Checking accuracy
    test = data[training_data_length:].copy()
    test['Predictions'] = predictions_close

    mape = np.mean(np.abs((test['Close'] - test['Predictions']) / test['Close'])) * 100

    '''
    Graphs and printing statements for debugging

    train = data[:training_data_length]
    
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    plt.figure(figsize=(12,8))
    plt.plot(train['Date'], train['Close'], label='Train (Actual)', color='blue')
    plt.plot(test['Date'], test['Close'], label='Test (Actual)', color='orange')
    plt.plot(test['Date'], test['Predictions'], label='Test (Predictions)', color='red')
    plt.title("Stock Predictions")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()
    '''

    # Forecast future days
    days = 30
    # input_sequence = scaled_data[-60:].flatten().tolist()
    predictions = []
    input_sequence = scaled_data[-60:].copy()
    
    for _ in range(days):
        x_input = np.array(input_sequence).reshape(1, 60, input_sequence.shape[1])
        y_pred = model.predict(x_input, verbose=0)
        predictions.append(y_pred[0][0])

        next_row = input_sequence[-1].copy()  # shape: (5,)
        next_row[3] = y_pred[0][0]            # overwrite 'Close' (index 3)

        # Append to input sequence
        input_sequence = np.vstack([input_sequence, next_row])[1:]  # keep length at 60

    predictions = np.array(predictions)

    noise = np.random.normal(0, 0.1, size=predictions.shape)
    predictions_noise = predictions + predictions * noise

    # Convert back to original scale
    pred_full = np.zeros((days, 5))
    pred_full[:, 3] = predictions_noise
    predictions_close = scaler.inverse_transform(pred_full)[:, 3]
    # predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    last_date = data['Date'].iloc[-1]

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')

    forecast = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions_close.flatten()
    })

    actual = data[['Date', 'Close']].tail(60).copy()
    actual.rename(columns={'Close': 'Actual_Close'}, inplace=True)

    combined = pd.merge(actual, forecast, on='Date', how='outer')

    combined.sort_values(by='Date', inplace=True)

    result = []
    for _, row in combined.iterrows():
        result.append({
            "date": row["Date"].strftime('%Y-%m-%d'),
            "actual": float(row["Actual_Close"]) if pd.notnull(row["Actual_Close"]) else None,
            "predicted": float(row["Predicted_Close"]) if pd.notnull(row["Predicted_Close"]) else None
        })

    '''
    # Graphs and printing statements for debugging

    # Plot last 60 actual + 30 predicted values
    plt.figure(figsize=(12,6))

    # Plot actual close prices (last 60 days)
    plt.plot(data['Date'], data['Close'], label='Actual Close', color='blue')

    # Plot predictions
    plt.plot(forecast['Date'], forecast['Predicted_Close'], label='Predicted Close', color='red')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    return result, mape

    

    