import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_file = "usdzar_data.csv"
temp_file = "temp_usdzar_data.csv"

correct_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

print("📥 Downloading latest 90 days of USD/ZAR data to temporary file...")

# Download fresh 90 days data every run to temp file
temp_df = yf.download("USDZAR=X", period="90d", interval="1h")
temp_df.dropna(inplace=True)

# Flatten columns if MultiIndex
if isinstance(temp_df.columns, pd.MultiIndex):
    temp_df.columns = temp_df.columns.get_level_values(0)

temp_df.to_csv(temp_file, index=True)
print(f"✅ Temp data saved with {len(temp_df)} rows.")

if os.path.exists(data_file):
    print(f"📂 Loading existing master data from {data_file}...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Master data has {len(df)} rows.")

    # Fix columns if needed
    if len(df.columns) == len(correct_cols):
        df.columns = correct_cols
    elif len(df.columns) == len(correct_cols) - 1:
        df.columns = correct_cols[:-1]
    else:
        print(f"⚠️ Warning: Unexpected number of columns ({len(df.columns)}). Using existing columns as-is.")

    # Identify new rows in temp_df not in df
    new_rows = temp_df.loc[~temp_df.index.isin(df.index)]
    print(f"Found {len(new_rows)} new rows to append.")

    if len(new_rows) > 0:
        df = pd.concat([df, new_rows])
        df.sort_index(inplace=True)
        df.to_csv(data_file, index=True)
        print(f"💾 Appended new rows and saved master data. Now total {len(df)} rows.")
    else:
        print("No new rows found. Master data is up to date.")
else:
    print("No existing master data found, using temp data as master.")
    df = temp_df.copy()
    df.to_csv(data_file, index=True)
    print(f"💾 Saved initial data with {len(df)} rows.")

print(f"Final dataset has {len(df)} rows.")

# --- Processing and feature engineering ---

df['Close'] = df['Close'].astype(float)

print("📊 Calculating indicators...")

fast_period = 12
slow_period = 26
signal_period = 9

ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
df['macd'] = ema_fast - ema_slow
df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

window = 20
rolling_mean = df['Close'].rolling(window=window).mean()
rolling_std = df['Close'].rolling(window=window).std()
df['bb_high'] = rolling_mean + (rolling_std * 2)
df['bb_low'] = rolling_mean - (rolling_std * 2)

df.dropna(inplace=True)

print("🎯 Creating prediction targets...")
df['target_6h'] = (df['Close'].shift(-6) > df['Close']).astype(int)
df['target_24h'] = (df['Close'].shift(-24) > df['Close']).astype(int)
df['target_48h'] = (df['Close'].shift(-48) > df['Close']).astype(int)
df['target_120h'] = (df['Close'].shift(-120) > df['Close']).astype(int)  # 5 days = 120 hours

df.dropna(inplace=True)

features = ['macd', 'macd_signal', 'rsi', 'bb_high', 'bb_low']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_6h, y_test_6h = train_test_split(X_scaled, df['target_6h'], test_size=0.2, shuffle=False)
_, _, y_train_24h, y_test_24h = train_test_split(X_scaled, df['target_24h'], test_size=0.2, shuffle=False)
_, _, y_train_48h, y_test_48h = train_test_split(X_scaled, df['target_48h'], test_size=0.2, shuffle=False)
_, _, y_train_120h, y_test_120h = train_test_split(X_scaled, df['target_120h'], test_size=0.2, shuffle=False)

print("🧠 Training models...")

model_params = dict(
    eval_metric='logloss',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8
)

model_6h = XGBClassifier(**model_params)
model_6h.fit(X_train, y_train_6h)

model_24h = XGBClassifier(**model_params)
model_24h.fit(X_train, y_train_24h)

model_48h = XGBClassifier(**model_params)
model_48h.fit(X_train, y_train_48h)

model_120h = XGBClassifier(**model_params)
model_120h.fit(X_train, y_train_120h)

def evaluate_model(model, X_test, y_test, timeframe):
    y_pred = model.predict(X_test)
    print(f"\n📈 {timeframe} Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
evaluate_model(model_6h, X_test, y_test_6h, "6-hour")
print("\n" + "="*50)
evaluate_model(model_24h, X_test, y_test_24h, "24-hour")
print("\n" + "="*50)
evaluate_model(model_48h, X_test, y_test_48h, "48-hour")
print("\n" + "="*50)
evaluate_model(model_120h, X_test, y_test_120h, "120-hour (5-day)")

latest = X_scaled[-1:].reshape(1, -1)

pred_6h = model_6h.predict(latest)[0]
prob_6h = model_6h.predict_proba(latest)[0][1]

pred_24h = model_24h.predict(latest)[0]
prob_24h = model_24h.predict_proba(latest)[0][1]

pred_48h = model_48h.predict(latest)[0]
prob_48h = model_48h.predict_proba(latest)[0][1]

pred_120h = model_120h.predict(latest)[0]
prob_120h = model_120h.predict_proba(latest)[0][1]

print("\n🔮 Future Price Predictions:")
print(f"Next 6 hours: {'📈 UP' if pred_6h == 1 else '📉 DOWN'} (Confidence: {prob_6h:.1%})")
print(f"Next 24 hours: {'📈 UP' if pred_24h == 1 else '📉 DOWN'} (Confidence: {prob_24h:.1%})")
print(f"Next 48 hours: {'📈 UP' if pred_48h == 1 else '📉 DOWN'} (Confidence: {prob_48h:.1%})")
print(f"Next 5 days: {'📈 UP' if pred_120h == 1 else '📉 DOWN'} (Confidence: {prob_120h:.1%})")

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.plot(df.index, df['bb_high'], label='BB High', linestyle='--', alpha=0.6)
plt.plot(df.index, df['bb_low'], label='BB Low', linestyle='--', alpha=0.6)
plt.title("USD/ZAR Price with Bollinger Bands")
plt.legend()
plt.tight_layout()
plt.show()