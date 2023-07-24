import numpy as np
import pandas as pd 
import ccxt 
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 시드 값 설정
seed_value = 11
set_seed(seed_value)

binance = ccxt.binance({
    'apiKey': YOUR_API_KEY,
    'secret': YOUR_SECRET_KEY,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'future'
    }
})
balance = binance.fetch_balance()

# window_size = 180
# forecast_steps = 12

window_size = 700
forecast_steps = 5

def get_USDT_tickers():
    binance = ccxt.binance()
    markets = binance.fetch_markets()
    USDT_tickers = [market['symbol'] for market in markets if market['quote'] == 'USDT' and market['active']]
    
    return USDT_tickers



def cryptoGenerator(crypto_list, feature, timeframe, limit):
    crypto_df = pd.DataFrame()
    
    for crypto in crypto_list:
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(f"{crypto}", timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = pd.DataFrame(df[f'{feature}'])
        df.rename(columns={f'{feature}':f'{crypto}'}, inplace=True)
        
        crypto_df = pd.concat([crypto_df, df], axis=1)
        crypto_df.dropna(inplace=True)
        
    return crypto_df



def trainDataGenerator(crypto_data, ticker, test):
    if test:
        data = crypto_data[[f'{ticker}']]
        data = data.values.reshape(-1, 1)[:-forecast_steps]
    else:
        data = crypto_data[[f'{ticker}']]
        data = data.values.reshape(-1, 1)

    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, Y = [], []
    for i in range(len(data_scaled) - window_size - forecast_steps + 1):
        X.append(data_scaled[i : (i + window_size), 0])
        Y.append(data_scaled[(i + window_size) : (i + window_size + forecast_steps), 0])

    X_train, Y_train = np.array(X), np.array(Y)

    # PyTorch 데이터 형태로 변환
    X_train = torch.FloatTensor(X_train).view([-1, window_size, 1])
    Y_train = torch.FloatTensor(Y_train)

    return X_train, Y_train



def cryptoGRU(crypto_data, X_train, Y_train, ticker, test):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GRU 모델 정의
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out
        
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = forecast_steps
    learning_rate = 0.001
    num_epochs = 50

    model = model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()

            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

    if test:
        trade_point = crypto_data[[f'{ticker}']][-forecast_steps - 1:-forecast_steps]
        trade_point = float(trade_point[f'{ticker}'])

        real_data = crypto_data[[f'{ticker}']][-forecast_steps:]
        hist_data = crypto_data[[f'{ticker}']][:-forecast_steps]
        hist_data = hist_data.values.reshape(-1, 1)

        input_data = hist_data[-window_size:]
        fit_data = hist_data[:-window_size]

        scaler = MinMaxScaler(feature_range=(0, 1))
        _ = scaler.fit_transform(fit_data)

        input_data_scaled = scaler.transform(input_data)
        input_data_tensor = torch.FloatTensor(input_data_scaled).view(1, window_size, 1).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(input_data_tensor).cpu().numpy()

        predictions = scaler.inverse_transform(predictions)
        predictions = list(predictions[0])
        real_data = list(real_data[f'{ticker}'])

        point_with_pred = [trade_point] + predictions
        point_with_real = [trade_point] + real_data

        plt.figure(figsize=(14,7))
        plt.plot(point_with_pred, label='predicted')
        plt.plot(point_with_real, label='real')
        plt.scatter(0, point_with_real[0], color='red', label='Trade Point')
        plt.legend()

        if (predictions[-1] - trade_point) > 0 and (real_data[-1] - trade_point) > 0:
            return 1
        elif (predictions[-1] - trade_point) < 0 and (real_data[-1] - trade_point) < 0:
            return 1
        else:
            return 0


    else:
        trade_point = crypto_data[[f'{ticker}']][-2:-1]
        trade_point = float(trade_point[f'{ticker}'])

        hist_data = crypto_data[[f'{ticker}']]
        hist_data = hist_data.values.reshape(-1, 1)

        input_data = hist_data[-window_size:]
        fit_data = hist_data[:-window_size]

        scaler = MinMaxScaler(feature_range=(0, 1))
        _ = scaler.fit_transform(fit_data)

        input_data_scaled = scaler.transform(input_data)
        input_data_tensor = torch.FloatTensor(input_data_scaled).view(1, window_size, 1).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(input_data_tensor).cpu().numpy()

        predictions = scaler.inverse_transform(predictions)
        predictions = list(predictions[0])

        final_return = (predictions[-1] - trade_point) / trade_point        
        
        return final_return
    


def cryptoNLinear(crypto_data, X_train, Y_train, ticker, test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class LTSF_NLinear(torch.nn.Module):
        def __init__(self, window_size, forecast_steps, individual, feature_size):
            super(LTSF_NLinear, self).__init__()
            self.window_size = window_size
            self.forecast_size = forecast_steps
            self.individual = individual
            self.channels = feature_size
            if self.individual:
                self.Linear = torch.nn.ModuleList()
                for i in range(self.channels):
                    self.Linear.append(torch.nn.Linear(self.window_size, self.forecast_size))
            else:
                self.Linear = torch.nn.Linear(self.window_size, self.forecast_size)

        def forward(self, x):
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
            if self.individual:
                output = torch.zeros([x.size(0), self.forecast_size, x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
            else:
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            x = x + seq_last
            return x
        
    learning_rate = 0.001
    num_epochs = 100

    model = LTSF_NLinear(
    window_size=window_size, 
    forecast_steps=forecast_steps, 
    individual=False, 
    feature_size=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()

            Y_pred = model(X_batch)
            Y_pred = Y_pred.squeeze(-1)  # 마지막 차원 제거(LTSF-Linear에만 사용)

            loss = criterion(Y_pred, Y_batch)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

    if test:
        trade_point = crypto_data[[f'{ticker}']][-forecast_steps - 1:-forecast_steps]
        trade_point = float(trade_point[f'{ticker}'])

        real_data = crypto_data[[f'{ticker}']][-forecast_steps:]
        hist_data = crypto_data[[f'{ticker}']][:-forecast_steps]
        hist_data = hist_data.values.reshape(-1, 1)

        input_data = hist_data[-window_size:]
        fit_data = hist_data[:-window_size]

        scaler = MinMaxScaler(feature_range=(0, 1))
        _ = scaler.fit_transform(fit_data)

        input_data_scaled = scaler.transform(input_data)
        input_data_tensor = torch.FloatTensor(input_data_scaled).view(1, window_size, 1).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(input_data_tensor).cpu().numpy()
        
        predictions = predictions.reshape(predictions.shape[0], -1) # LTSF-Linear에만 추가
        predictions = scaler.inverse_transform(predictions)
        predictions = list(predictions[0])
        real_data = list(real_data[f'{ticker}'])

        point_with_pred = [trade_point] + predictions
        point_with_real = [trade_point] + real_data

        plt.figure(figsize=(14,7))
        plt.plot(point_with_pred, label='predicted')
        plt.plot(point_with_real, label='real')
        plt.scatter(0, point_with_real[0], color='red', label='Trade Point')
        plt.legend()

        if (predictions[-1] - trade_point) > 0 and (real_data[-1] - trade_point) > 0:
            return 1
        elif (predictions[-1] - trade_point) < 0 and (real_data[-1] - trade_point) < 0:
            return 1
        else:
            return 0


    else:
        trade_point = crypto_data[[f'{ticker}']][-2:-1]
        trade_point = float(trade_point[f'{ticker}'])

        hist_data = crypto_data[[f'{ticker}']]
        hist_data = hist_data.values.reshape(-1, 1)

        input_data = hist_data[-window_size:]
        fit_data = hist_data[:-window_size]

        scaler = MinMaxScaler(feature_range=(0, 1))
        _ = scaler.fit_transform(fit_data)

        input_data_scaled = scaler.transform(input_data)
        input_data_tensor = torch.FloatTensor(input_data_scaled).view(1, window_size, 1).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(input_data_tensor).cpu().numpy()

        predictions = predictions.reshape(predictions.shape[0], -1) # LTSF-Linear에만 추가
        predictions = scaler.inverse_transform(predictions)
        predictions = list(predictions[0])

        final_return = (predictions[-1] - trade_point) / trade_point
        
        return final_return