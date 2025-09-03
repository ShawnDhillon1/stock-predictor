import os
from openai import OpenAI
from dotenv import load_dotenv
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import LSTM, Dense, Dropout  

class Agent:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_msg = "You are a helpful assistant."
        self.messages = []
        self.messages.append({"role": "system", "content": self.system_msg})
        os.makedirs("models", exist_ok=True)  # ensure models dir exists
    # build an LSTM model using TensorFlow/Keras
    def build_lstm_model(self, input_shape):
        model = Sequential([ # builds a stacked neural network
            # first LSTM layer (returns sequences for next LSTM)
            LSTM(64, return_sequences=True, input_shape=input_shape),# 50 neurons, R_S=T needed when stacking another LSTM
            Dropout(0.3),  # dropout helps prevent overfitting
            # second LSTM layer (final LSTM layer)
            LSTM(32, return_sequences=False),
            Dropout(0.3),# 20% of random neurons turned off,forces model to learn more patterns than just memorizing
            #fully connected layers for final prediction
            Dense(16,activation="relu"), # layer has 16 neurons
            Dense(1)  # Output: predicted stock price
        ])
        # compile the model with Adam optimizer and MSE loss
        model.compile(optimizer='adam', loss='mse',metrics=['mae']) # MSE tells model how wrong predictions are
        return model

    def predict_stock(self, ticker, days_ahead=5):
        # keeps uppercase
        ticker = ticker.upper()
        seq_len = 252 # 1 full trading year 
        model_path  = f"models/{ticker}_lstm_price.keras"      
        scaler_path = f"models/{ticker}_scaler_price.pkl"

        # download daily history & get current live price
        data = yf.download(ticker, period="max", interval="1d", auto_adjust=True)[["Close"]].dropna()
        try:
            # gets latest current price fastest
            current_price = yf.Ticker(ticker).fast_info.get("last_price", None)
            # if no price available (market closed)
            if current_price is None:
                raise ValueError("fast_info did not return a price")
            #converts price into a float 
            current_price = float(current_price)
        # fallback if fastest method fails
        except Exception:
            intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True)
            # make sure its float so its easier to do the math later on
            current_price = float(intraday["Close"].iloc[-1].item()) # check notes
        print(f"[INFO] Current {ticker} price: ${current_price:.2f}") # log current stock price

       # N = number of historical prices 
        close = data["Close"].values  # shape (N,)
        # make sure we have more data points than sequence length, cannot train LSTM w/o
        if len(close) <= seq_len:
            raise ValueError(f"Not enough data: have {len(close)}, need > {seq_len}")
        # prepare input X and output Y sequences
        X, y = [], []
        for i in range(seq_len, len(close)):
            X.append(close[i - seq_len:i])  # last seq_len closes
            y.append(close[i])              # next day's close
        X= np.array(X)                 # (samples, seq_len)
        y= np.array(y)                 # (samples,)

        # train / test 
        split = int(len(X) * 0.8) # 80% train the rest for testing 
        X_train, X_test = X[:split], X[split:] # 80 then rest 20 for testing
        y_train, y_test = y[:split], y[split:]
        # keeps a copy of raw target values for later evaluation
        y_train_raw = y_train.copy()
        y_test_raw  = y_test.copy()

        # if saved scaler for stock already exist then load it 
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f: # open the saved pickle file in read-binary mode
                scaler_y = pickle.load(f) # load into object memory
        # else create new scaler and fit into training target prices
        else:
            # MinMaxScaler to fit values between 0 and 1
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            # fir scaler only on y train (2D input)
            scaler_y.fit(y_train.reshape(-1, 1))
            # save for future predictions
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler_y, f)

        # scale target prices using fitted scaler
        # ensures that both y_train and y_test are in the range [0, 1]
        y_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_test  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # scale input equences (closing prices) with same scaler
        # keeps input and output on same scale for LSTM
        X_train = scaler_y.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_test  = scaler_y.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        # LSTM expects 3D input shape (samples, time steps, features)
        # samples = number of sequences, time_steps = seq_len, features = 1 (just closing price)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

        # load or train model (don’t compile on load)
        if os.path.exists(model_path): # if trained LSTM exist load it 
            model = load_model(model_path, compile=False) # False means we wont try to load the saved optimizer state or loss function
        else:
            # train the model on the scaled training data
            # epochs=15 - go through the training dataset 15 times
            # batch_size=32 - update model weights after every 32 samples
            # verbose=1 - show progress bar and training loss after each epoch
            model = self.build_lstm_model((seq_len, 1))
            history = model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
            model.save(model_path, include_optimizer=False) # we dont save optimizer state, which makes it easier to load

        # 6) Compile in current environment (robust to version changes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # 0.001
            loss=tf.keras.losses.MeanSquaredError(), # LF - what model tries to minimize
            metrics=[tf.keras.metrics.MeanAbsoluteError()] # average of preidcted - actual
        )
        # evaluate (scaled space)
        # unseen test data for performance measure
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"[EVAL] Test MSE (scaled): {loss:.6f} | Test MAE (scaled): {mae:.6f}") # normalized scale not real dollar amounts

        y_pred_scaled = model.predict(X_test, verbose=0) # predict on the (scaled) test inputs → outputs are scaled
        y_pred = scaler_y.inverse_transform(y_pred_scaled)  # convert scaled predictions back to real dollar prices
        y_true = y_test_raw.reshape(-1, 1)  # already real prices (already unscaled)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE in dollars (penalizes big errors more)
        mae2 = mean_absolute_error(y_true, y_pred) # MAE in dollars (average absolute error)
        print(f"[EVAL] Test RMSE ($): {rmse:.4f} | Test MAE ($): {mae2:.4f}")

        # forecast next N daily prices, anchored to the live current price
        # start from the last historical window, but replace its last value with the live price
        last_seq = X[-1].copy()                # shape (seq_len,)
        last_seq[-1] = current_price               # anchor seed
        last_seq = scaler_y.transform(last_seq.reshape(-1, 1)).reshape(seq_len, 1) # scale the window to match what the model was trained on

        preds = [] # stores future predicted prices in dollars
        last_price = current_price
        for _ in range(days_ahead):
            #predict next days scaled price for current scaled window
            pred_scaled = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0][0]
            #conver scaled prediction back to dollar price
            pred_price  = scaler_y.inverse_transform([[pred_scaled]])[0][0]

            # limit day-over-day change to ±5% to avoid explosive drift
            upper = last_price * 1.05
            lower = last_price * 0.95
            pred_price = max(min(pred_price, upper), lower)

            preds.append(pred_price) # store the price 
            last_price = pred_price # update rolling reference

            # slide window with the **scaled** predicted price, keep 2D shape
            pred_scaled_for_seq = scaler_y.transform([[pred_price]])[0][0]
            last_seq = np.append(last_seq[1:], [[pred_scaled_for_seq]], axis=0)

        # 8) Build output
        result = f"\nPredicted closing prices for {ticker} (next {days_ahead} days):\n"
        for i, p in enumerate(preds, 1):
            result += f"  Day {i}: ${p:.2f}\n" # 2 deciaml places
        return result 

    def send_message(self, message):
        self.messages.append({"role": "user", "content": message})
        
        # check if the message is a stock prediction request
        if message.lower().startswith("predict stock"):
            parts = message.strip().split()
            # check if at least the ticker symbol is provided  "predict stock AAPL"
            if len(parts) >= 3:
                ticker = parts[2].upper()  # extract the stock symbol and make it uppercase
                # call your own method to predict stock prices and return the result
                return self.predict_stock(ticker)
           
        # if not a stock command call OpenAI's chat model
        response=self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply
       
if __name__ == "__main__":
    agent = Agent()

    while True:
        user_input = input("Shawn: ")
        if user_input.lower() == "exit":
            break
        response = agent.send_message(user_input)
        print("AI Assistant:", response)

