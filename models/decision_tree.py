import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def add_action(change, threshold=0.05):
    if change > threshold:
        return 1 # Buy
    elif change < threshold:
        return -1 # Sell
    else:
        return 0 # Hold

def run_decision_tree(data, test_size=0.2):
    data = data.drop(columns=['Stock Splits', 'Dividends', 'Date'])

    data['price_change'] = data['Close'].pct_change().shift(-1) # Add next day's price change for prediction
    data = data.dropna()

    # Add features
    data['daily_return'] = (data['Close'] - data['Open']) / data['Open']
    data['prev_close_diff'] = data['Close'].pct_change()
    data['3_day_ma'] = data['Close'].rolling(window=3).mean()
    data['5_day_ma'] = data['Close'].rolling(window=5).mean()
    data['volume_change'] = data['Volume'].pct_change()

    # Add target and remove price change
    data['target'] = data['price_change'].apply(add_action)
    data = data.drop(columns=['price_change'])
    data = data.dropna()

    # Prepare training data
    X = data.iloc[:, :-1]   # All columns except target
    y = data.iloc[:, -1]    # Target column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # Create and train decision tree
    dtc = DecisionTreeClassifier(max_depth=10)
    dtc.fit(X_train, y_train)

    # Make predictions
    y_pred = dtc.predict(X_test)

    # Determine accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Get information from the latest day
    latest_data  = data.iloc[-1:].copy()
    latest_data = latest_data.drop(columns=['target'])

    # Make prediction
    next_day_prediction = dtc.predict(latest_data)[0]

    return accuracy, next_day_prediction