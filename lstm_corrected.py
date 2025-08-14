# Mashiat Tabassum Khan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform
import os

# Load dataset
file_path = '/home/mashiat/Downloads/airline-passengers.csv'  
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_csv(file_path)
data.columns = ['Month', 'Passengers']  # Rename columns

# Convert Month to datetime and set as index
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data, label='Passengers')
plt.title('Monthly Airline Passengers')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Convert data to sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 12  # Sequence length (1 year)
X, y = create_sequences(data_scaled, TIME_STEPS)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def build_lstm_model(units, activation, optimizer, dropout_rate, num_layers):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))

    # Add LSTM layers based on num_layers
    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False
        model.add(LSTM(units if i < num_layers - 1 else units // 2,
                       activation=activation,
                       return_sequences=return_seq))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Learning rate scheduler
def scheduler(epoch, lr):
    return lr * 0.9 if epoch > 50 else lr

lr_scheduler = LearningRateScheduler(scheduler)

# Hyperparameter tuning
param_grid = {
    'units': [128, 256],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': uniform(0.1, 0.3),
    'batch_size': [16, 32],
    'epochs': [100, 200],
    'num_layers': [2, 3] 
}

n_combinations = 15
random_params = list(ParameterSampler(param_grid, n_iter=n_combinations, random_state=42))

best_model = None
best_loss = float('inf')
best_params = None

for params in random_params:
    print(f"\nTesting params: {params}")
    model = build_lstm_model(
        units=params['units'],
        activation=params['activation'],
        optimizer=params['optimizer'],
        dropout_rate=params['dropout_rate'],
        num_layers=params['num_layers']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=0,
        callbacks=[early_stopping, lr_scheduler]
    )

    val_loss = min(history.history['val_loss'])

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model
        best_params = params

print("\nBest Hyperparameters:")
print(best_params)

results_df = pd.DataFrame(random_params)
results_df.to_csv('mashiat_lstm_output.csv', index=False)

test_loss = best_model.evaluate(X_test, y_test)
print(f'\nTest Loss: {test_loss:.4f}')

predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='orange')
plt.legend()
plt.xlim(0, len(y_test_actual))
plt.title('LSTM Predictions on Airline Passengers Data')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.show()

