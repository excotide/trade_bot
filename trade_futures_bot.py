import os
import csv
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import signal
import sys
import urllib3
import feedparser
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# Matikan peringatan SSL untuk koneksi stabil
urllib3.disable_warnings()

# ==========================================
# 0. MEMUAT AI SENTIMEN (FINBERT)
# ==========================================
print("⏳ Memuat AI FinBERT (NLP Brain)...")
try:
    ai_sentimen = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("✅ AI Sentimen Siap!")
except Exception as e:
    print(f"⚠️ Gagal memuat FinBERT: {e}. Bot akan berjalan tanpa Gate 3.")
    ai_sentimen = None

# ==========================================
# 1. KONFIGURASI API (FUTURES DEMO)
# ==========================================
load_dotenv()

# Koneksi ke Mainnet (Hanya Baca Data)
exchange_data = ccxt.binance({'options': {'defaultType': 'future'}})

# Koneksi ke Demo Trading (Eksekusi Order)
exchange_trade = ccxt.binance({
    'apiKey': os.getenv('BINANCE_FUTURES_API_KEY'),
    'secret': os.getenv('BINANCE_FUTURES_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'test': True  # Mengarah ke Demo Trading Endpoint
    }
})

# ==========================================
# 2. PARAMETER STRATEGI
# ==========================================
SIMBOL = 'BTC/USDT'
TIMEFRAME = '1m'       
BATAS_RISIKO_USD = 5.0      # Jumlah margin per posisi (dalam USDT)
LEVERAGE = 10               # Daya ungkit
AMBANG_BATAS_TEKNIKAL = 70  # Gate 1 Minimal %
AMBANG_BATAS_AI = 60.0      # Gate 2 Minimal %

# Variabel State
sedang_ada_posisi = False
tipe_posisi = None # 'LONG' atau 'SHORT'
harga_entry_sekarang = 0
lot_sekarang, tp_sekarang, sl_sekarang = 0, 0, 0
fitur_ai = ['RSI', 'MACD_Line', 'MACD_Hist', 'Jarak_ke_EMA', 'volume']

# ==========================================
# 3. FUNGSI UTILITAS (LOG & WIN RATE)
# ==========================================
def simpan_ke_csv(data_trade):
    file_name = 'history_futures_v9.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Waktu', 'Simbol', 'Side', 'Entry', 'Exit', 'Lot', 'PnL_USD', 'Status'])
        if not file_exists: writer.writeheader()
        writer.writerow(data_trade)

def hitung_win_rate():
    file_name = 'history_futures_v9.csv'
    if not os.path.isfile(file_name): return 0, 0, 0.0
    try:
        df_history = pd.read_csv(file_name)
        if df_history.empty: return 0, 0, 0.0
        total = len(df_history)
        win = len(df_history[df_history['PnL_USD'] > 0])
        return total, win, (win/total)*100
    except: return 0, 0, 0.0

def tangani_keluar(sig, frame):
    print("\n\n🛑 BOT DIMATIKAN. Membersihkan sisa order...")
    sys.exit(0)
signal.signal(signal.SIGINT, tangani_keluar)

# ==========================================
# 4. DATA ENGINE & AI TRAINING
# ==========================================
def latih_ai_diawal():
    print("🧠 AI sedang mempelajari pola market 1000 menit terakhir...")
    try:
        bars = exchange_data.fetch_ohlcv(SIMBOL, timeframe=TIMEFRAME, limit=1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        df['EMA_200'] = ta.ema(df['close'], length=200)
        df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[fitur_ai], df['Target'])
        print("✅ Otak AI Berhasil Dilatih!")
        return model
    except Exception as e:
        print(f"❌ Gagal melatih AI: {e}")
        sys.exit(1)

def ambil_data_live():
    try:
        bars = exchange_data.fetch_ohlcv(SIMBOL, timeframe=TIMEFRAME, limit=250)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA_200'] = ta.ema(df['close'], length=200)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        macd = ta.macd(df['close'])
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
        return df
    except: return None

# ==========================================
# 5. LOGIKA ANALISIS DUAL-DIRECTION
# ==========================================
def hitung_skor_dual(bar, df, model_ai):
    h, ema200, rsi = bar['close'], bar['EMA_200'], bar['RSI']
    m_line, m_sig = bar['MACD_Line'], bar['MACD_Signal']
    
    # Gate 2: Analisis AI
    data_ai = df.iloc[-2:][fitur_ai].head(1)
    prob = model_ai.predict_proba(data_ai)[0]
    p_naik, p_turun = prob[1] * 100, prob[0] * 100

    # Gate 1: Skor Teknikal LONG
    s_long = 0
    if m_line > m_sig: s_long += 35
    if h > ema200: s_long += 35
    if 40 < rsi < 70: s_long += 30

    # Gate 1: Skor Teknikal SHORT
    s_short = 0
    if m_line < m_sig: s_short += 35
    if h < ema200: s_short += 35
    if 30 < rsi < 60: s_short += 30

    return s_long, s_short, p_naik, p_turun

def analisis_berita_kilat():
    if not ai_sentimen: return "NEUTRAL", []
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        titles = [e.title for e in feed.entries[:3]]
        skor = 0
        rincian = []
        for t in titles:
            res = ai_sentimen(t)[0]
            label = res['label'].upper()
            if label == 'POSITIVE': skor += 1; icon = "🟢"
            elif label == 'NEGATIVE': skor -= 1; icon = "🔴"
            else: icon = "⚪"
            rincian.append(f"   {icon} {t[:50]}... -> {label}")
        res_sentimen = "BULLISH" if skor > 0 else "BEARISH" if skor < 0 else "NEUTRAL"
        return res_sentimen, rincian
    except: return "NEUTRAL", ["   ⚠️ Gagal akses RSS Berita"]

# ==========================================
# 6. ENGINE EKSEKUSI ORDER
# ==========================================
def buka_posisi(side, lot, tp, sl):
    try:
        # 1. Market Order
        order = exchange_trade.create_market_order(SIMBOL, 'buy' if side == 'LONG' else 'sell', lot)
        entry_price = order['average'] if order['average'] else exchange_data.fetch_ticker(SIMBOL)['last']
        
        # 2. Stop Loss (Reduce Only)
        exchange_trade.create_order(SIMBOL, 'STOP_MARKET', 'sell' if side == 'LONG' else 'buy', lot, 
                                    params={'stopPrice': sl, 'reduceOnly': True})
        
        # 3. Take Profit (Reduce Only)
        exchange_trade.create_order(SIMBOL, 'TAKE_PROFIT_MARKET', 'sell' if side == 'LONG' else 'buy', lot, 
                                    params={'stopPrice': tp, 'reduceOnly': True})
        
        print(f"✅ {side} Berhasil Dibuka di ${entry_price:.2f}")
        return entry_price
    except Exception as e:
        print(f"❌ Gagal Eksekusi: {e}")
        return None

# ==========================================
# 7. LOOP UTAMA (ULTIMATE DASHBOARD)
# ==========================================
model_ai_global = latih_ai_diawal()
print(f"\n🚀 Bot v9.0 DASHBOARD Aktif di {SIMBOL}...")

while True:
    try:
        df = ambil_data_live()
        try:
            bal = exchange_trade.fetch_balance()
            free_usdt = bal['USDT']['free']
            wallet_total = bal['USDT']['total']
        except: free_usdt, wallet_total = 0.0, 0.0
            
        total_trade, win_trade, win_rate = hitung_win_rate()
        
        if df is not None:
            harga_skrg = df.iloc[-1]['close']
            
            # --- MONITORING POSISI AKTIF ---
            if sedang_ada_posisi:
                pos = exchange_trade.fetch_positions([SIMBOL])
                ukuran = 0
                for p in pos:
                    if p['symbol'] == SIMBOL:
                        ukuran = abs(float(p['info']['positionAmt']))
                        unrealized_pnl = float(p['info']['unRealizedProfit'])
                
                if ukuran == 0:
                    print("\n🔔 POSISI CLOSED (TP/SL Terpukul)!")
                    exchange_trade.cancel_all_orders(SIMBOL)
                    sedang_ada_posisi = False
                else:
                    tanda = "+" if unrealized_pnl >= 0 else ""
                    print(f"\n" + "🔥" * 15)
                    print(f"   POSISI {tipe_posisi} AKTIF")
                    print(f"   PnL: {tanda}${unrealized_pnl:.2f} | Live: ${harga_skrg:.2f}")
                    print(f"   Target: TP ${tp_sekarang:.2f} | SL ${sl_sekarang:.2f}")
                    print(" " + "🔥" * 15)

            # --- MENCARI SINYAL (RADAR MODE) ---
            if not sedang_ada_posisi:
                bar = df.iloc[-2]
                s_long, s_short, p_naik, p_turun = hitung_skor_dual(bar, df, model_ai_global)
                
                print(f"\n" + "═"*65)
                print(f"📡 RADAR FUTURES HYBRID | {datetime.now().strftime('%H:%M:%S')} | {SIMBOL}")
                print(f"💰 BALANCE  : ${free_usdt:.2f} / ${wallet_total:.2f}")
                print(f"🏆 STATS    : Win Rate {win_rate:.1f}% ({win_trade}/{total_trade} Trade)")
                print(f"📈 PRICE    : ${harga_skrg:.2f}")
                print(f"─"*65)

                # Gate 1 & 2 Display
                arah_dom = "LONG" if s_long >= s_short else "SHORT"
                skor_dom = s_long if s_long >= s_short else s_short
                prob_dom = p_naik if s_long >= s_short else p_turun
                
                status_g1 = "✅" if skor_dom >= AMBANG_BATAS_TEKNIKAL else "⏳"
                status_g2 = "✅" if prob_dom >= AMBANG_BATAS_AI else "⏳"

                print(f"1️⃣ GATE 1: TEKNIKAL {arah_dom} [{status_g1}] Skor: {skor_dom}%")
                print(f"2️⃣ GATE 2: AI PREDICTION [{status_g2}] Prob: {prob_dom:.1f}%")
                print(f"─"*65)

                # EKSEKUSI LOGIC
                if skor_dom >= AMBANG_BATAS_TEKNIKAL and prob_dom >= AMBANG_BATAS_AI:
                    print(f"🚀 SINYAL {arah_dom} TERDETEKSI! Memicu Gate 3 Berita...")
                    sentimen_berita, rincian = analisis_berita_kilat()
                    for r in rincian: print(r)
                    
                    # Konfirmasi Akhir (Gate 3)
                    if (arah_dom == "LONG" and sentimen_berita == "BEARISH") or \
                       (arah_dom == "SHORT" and sentimen_berita == "BULLISH"):
                        print(f"❌ DIBATALKAN: Sentimen berita melawan arah trading!")
                    else:
                        # KALKULASI LOT & TARGET
                        atr = bar['ATR']
                        sl = harga_skrg - (1.5 * atr) if arah_dom == "LONG" else harga_skrg + (1.5 * atr)
                        tp = harga_skrg + (3.0 * atr) if arah_dom == "LONG" else harga_skrg - (3.0 * atr)
                        
                        lot = round((BATAS_RISIKO_USD * LEVERAGE) / harga_skrg, 3)
                        if lot < 0.001: lot = 0.001
                        
                        print(f"🔥 CONFLUENCE! Membuka Posisi {arah_dom}...")
                        entry = buka_posisi(arah_dom, lot, round(tp, 2), round(sl, 2))
                        if entry:
                            harga_entry_sekarang, lot_sekarang = entry, lot
                            tp_sekarang, sl_sekarang = tp, sl
                            tipe_posisi, sedang_ada_posisi = arah_dom, True
                else:
                    print("💤 STATUS: Menunggu Sinyal Confluence (Teknikal + AI)...")
                print("═"*65)

        time.sleep(10)
    except Exception as e:
        print(f"⚠️ Radar Error: {e}"); time.sleep(5)