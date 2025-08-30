Stock Predictor
A simple LSTM-based stock price predictor built with TensorFlow and yfinance.
Features
Downloads stock data with yfinance
Uses an LSTM model trained on the last 90 days of closing prices
Predicts the next 5 days of prices
Anchors forecasts to the current live price
Saves models and scalers for each stock in the models/ folder

How to Run
Clone this repo and enter the folder:
git clone https://github.com/ShawnDhillon1/stock-predictor.git
cd stock-predictor

Create a virtual environment and install dependencies:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Run the program:
python agent.py

Type a stock prediction command:
Shawn: predict stock TSLA

