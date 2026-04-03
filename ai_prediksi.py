import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("⏳ Mengambil data historis dari Binance untuk 'sekolah' AI...")

# Inisialisasi Bersih (Tanpa verify: False)
exchange = ccxt.binance()

# Kita ambil 1000 candle terakhir (1m) untuk bahan belajar
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1000)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print("🧮 Menghitung Indikator Teknikal (Sebagai bahan ajar/Fitur)...")
# 1. FITUR (X) - Data yang dipelajari AI
df['RSI'] = ta.rsi(df['close'], length=14)
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
df['MACD_Line'] = macd['MACD_12_26_9']
df['MACD_Hist'] = macd['MACDh_12_26_9']
df['EMA_200'] = ta.ema(df['close'], length=200)

df['Jarak_ke_EMA'] = df['close'] - df['EMA_200'] 

# 2. TARGET (Y) - Kunci Jawaban!
df['Target_Naik'] = (df['close'].shift(-1) > df['close']).astype(int)

fitur_kolom = ['RSI', 'MACD_Line', 'MACD_Hist', 'Jarak_ke_EMA', 'volume']

# Simpan candle TERAKHIR (yang sedang live) ke variabel khusus SEBELUM dropna
data_live_untuk_ditebak = df.iloc[-1:][fitur_kolom].copy()

# Hapus baris kosong agar AI bisa belajar dari data masa lalu yang lengkap
df.dropna(inplace=True)

# Pisahkan antara Data Belajar (X) dan Kunci Jawaban (Y)
X = df[fitur_kolom]
y = df['Target_Naik']

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ==========================================
# 3. PROSES TRAINING (SEKOLAH AI)
# ==========================================
print("🧠 AI sedang belajar mencari pola tersembunyi...")
model_ai = RandomForestClassifier(n_estimators=100, random_state=42)
model_ai.fit(X_train, y_train)

prediksi_ujian = model_ai.predict(X_test)
akurasi = accuracy_score(y_test, prediksi_ujian) * 100
print(f"🎯 Akurasi Ujian AI di masa lalu: {akurasi:.2f}% (Tebakan benar dari {len(X_test)} kasus)")

# ==========================================
# 4. PREDIKSI LIVE (MENEBAK MASA DEPAN)
# ==========================================
print("\n" + "="*50)
print("🔮 PREDIKSI CANDLE BERIKUTNYA OLEH AI")
print("="*50)

# Gunakan data_live yang sudah kita selamatkan dari dropna tadi
tebakan = model_ai.predict(data_live_untuk_ditebak)[0]
probabilitas = model_ai.predict_proba(data_live_untuk_ditebak)[0]

prob_turun = probabilitas[0] * 100
prob_naik = probabilitas[1] * 100

arah = "📈 NAIK (BULLISH)" if tebakan == 1 else "📉 TURUN (BEARISH)"

print(f"Harga BTC Terakhir : ${bars[-1][4]:.2f}") 
print(f"Prediksi AI        : {arah}")
print(f"Keyakinan Naik     : {prob_naik:.1f}%")
print(f"Keyakinan Turun    : {prob_turun:.1f}%")
print("="*50)