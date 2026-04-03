import os
import csv
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import signal
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import urllib3
import feedparser
from transformers import pipeline

urllib3.disable_warnings()

# ==========================================
# 0. MEMUAT AI SENTIMEN (FINBERT)
# ==========================================
print("⏳ Sedang memuat AI FinBERT (Membaca Bahasa Manusia)...")
ai_sentimen = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ Otak Kanan (AI Sentimen) Siap!")

# ==========================================
# 1. KONFIGURASI KONEKSI (HANYA BACA DATA MAINNET)
# ==========================================
# Kita buang exchange_trade dan API Keys. Kita hanya butuh data publik asli.
exchange_data = ccxt.binance({
    'options': {'defaultType': 'future'}
})

# ==========================================
# 2. PARAMETER STRATEGI & DOMPET VIRTUAL (SIMULATOR)
# ==========================================
SIMBOL = 'BTC/USDT'
TIMEFRAME = '1m'       
BATAS_RISIKO_USD = 2.0  
LEVERAGE = 10               
AMBANG_BATAS_TEKNIKAL = 70  
AMBANG_BATAS_AI = 60.0      

# --- SISTEM DOMPET VIRTUAL ---
SALDO_VIRTUAL = 100.0  # Modal awal mainan Anda: $100

sedang_ada_posisi = False
harga_entry_sekarang = 0
lot_sekarang, tp_sekarang, sl_sekarang = 0, 0, 0
model_ai_global = None 
fitur_ai = ['RSI', 'MACD_Line', 'MACD_Hist', 'Jarak_ke_EMA', 'volume']

# ==========================================
# 3. FUNGSI LOG, WIN RATE & PROTEKSI
# ==========================================
def simpan_ke_csv(data_trade):
    file_name = 'history_simulasi_futures.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Waktu', 'Simbol', 'Side', 'Entry', 'Exit', 'Lot', 'PnL_USD', 'Saldo_Akhir', 'Status'])
        if not file_exists: writer.writeheader()
        writer.writerow(data_trade)

def hitung_win_rate():
    file_name = 'history_simulasi_futures.csv'
    if not os.path.isfile(file_name): return 0, 0, 0.0
    try:
        df_history = pd.read_csv(file_name)
        if df_history.empty: return 0, 0, 0.0
        total_trade = len(df_history)
        win_trade = len(df_history[df_history['PnL_USD'] > 0])
        win_rate = (win_trade / total_trade) * 100
        return total_trade, win_trade, win_rate
    except: return 0, 0, 0.0

def tangani_keluar(sig, frame):
    global sedang_ada_posisi, lot_sekarang, harga_entry_sekarang, SALDO_VIRTUAL
    print("\n\n🛑 MENERIMA SINYAL BERHENTI...")
    if sedang_ada_posisi:
        print(f"⚠️ Menutup paksa posisi SIMULASI {SIMBOL}...")
        try:
            harga_akhir = exchange_data.fetch_ticker(SIMBOL)['last']
            pnl_final = (harga_akhir - harga_entry_sekarang) * lot_sekarang
            SALDO_VIRTUAL += pnl_final
            simpan_ke_csv({'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Simbol': SIMBOL, 'Side': 'LONG', 'Entry': harga_entry_sekarang, 'Exit': harga_akhir, 'Lot': lot_sekarang, 'PnL_USD': round(pnl_final, 2), 'Saldo_Akhir': round(SALDO_VIRTUAL, 2), 'Status': 'FORCE_CLOSE'})
            print(f"💰 POSISI DITUTUP. PnL: ${pnl_final:.2f} | Saldo: ${SALDO_VIRTUAL:.2f}")
        except Exception as e: print(f"❌ Gagal tutup otomatis: {e}")
    sys.exit(0)
signal.signal(signal.SIGINT, tangani_keluar)

# ==========================================
# 4. FUNGSI AI PREDIKSI (OTAK KIRI)
# ==========================================
def latih_ai_diawal():
    print("⏳ Menyiapkan Otak Kiri (AI Angka). Mengambil 1000 data historis Futures...")
    bars = exchange_data.fetch_ohlcv(SIMBOL, timeframe=TIMEFRAME, limit=1000)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_Line'] = macd['MACD_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
    df['Target_Naik'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[fitur_ai], df['Target_Naik'])
    print("✅ Otak Kiri (AI Angka) Siap!")
    return model

# ==========================================
# 5 & 6. FUNGSI ANALISIS BERITA & LIVE
# ==========================================
def analisis_berita_kilat():
    url_rss = "https://cointelegraph.com/rss"
    try:
        feed = feedparser.parse(url_rss)
        berita_terkini = feed.entries[:3] 
        skor_total = 0
        rincian_berita = []
        for item in berita_terkini:
            hasil = ai_sentimen(item.title)[0]
            label = hasil['label'].upper()
            if label == 'POSITIVE': skor_total += 1; simbol = "🟢"
            elif label == 'NEGATIVE': skor_total -= 1; simbol = "🔴"
            else: simbol = "⚪"
            rincian_berita.append(f"   {simbol} {item.title[:50]}... -> {label}")
        if skor_total > 0: return "BULLISH", rincian_berita
        elif skor_total < 0: return "BEARISH", rincian_berita
        else: return "NEUTRAL", rincian_berita
    except Exception as e: return "NEUTRAL", [f"   ⚠️ Gagal mengambil berita: {e}"]

def ambil_data_live():
    try:
        bars = exchange_data.fetch_ohlcv(SIMBOL, timeframe=TIMEFRAME, limit=250)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA_200'] = ta.ema(df['close'], length=200)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None: df = pd.concat([df, bbands], axis=1)
        sar_data = ta.psar(df['high'], df['low'], df['close'])
        if sar_data is not None: df['SAR'] = sar_data.iloc[:, 0]
        macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_data is not None: df = pd.concat([df, macd_data], axis=1)
        
        df['RSI'] = df['RSI_14']
        df['MACD_Line'] = df.get('MACD_12_26_9', 0)
        df['MACD_Hist'] = df.get('MACDh_12_26_9', 0)
        df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
        return df
    except: return None

def hitung_skor_teknikal(bar):
    rincian, total_skor, h, ema200 = {}, 0, bar['close'], bar['EMA_200']
    macd_ok = (bar.get('MACD_Line', 0) > bar.get('MACDs_12_26_9', 0)) and (bar.get('MACD_Hist', 0) > 0)
    if macd_ok: total_skor += 25
    rincian['MACD (25%)'] = "✅ BULLISH" if macd_ok else "❌ BEARISH"
    ema_ok = h > ema200
    if ema_ok: total_skor += 25
    rincian['EMA 200 (25%)'] = "✅ BULLISH" if ema_ok else "❌ BEARISH"
    col_bbm = [c for c in bar.index if 'BBM' in c]
    bb_ok = col_bbm and h > bar[col_bbm[0]]
    if bb_ok: total_skor += 20
    rincian['Bollinger Mid (20%)'] = "✅ DI ATAS" if bb_ok else "❌ DI BAWAH"
    rsi_val = bar['RSI_14']
    rsi_ok = 40 < rsi_val < 70
    if rsi_ok: total_skor += 15
    rincian[f'RSI {rsi_val:.1f} (15%)'] = "✅ STABIL" if rsi_ok else "❌ OVER"
    sar_ok = bar['SAR'] < h
    if sar_ok: total_skor += 15
    rincian['P. SAR (15%)'] = "✅ POSITIF" if sar_ok else "❌ NEGATIF"
    return total_skor, rincian

# ==========================================
# 7. LOOP UTAMA (MESIN SIMULATOR FUTURES)
# ==========================================
model_ai_global = latih_ai_diawal()
print(f"\n🚀 Bot v8.1 PAPER TRADING EDITION Aktif di {SIMBOL}...")

while True:
    try:
        df = ambil_data_live()
        
        if df is not None:
            harga_skrg = df.iloc[-1]['close']
            
            # --- PENGECEKAN POSISI SIMULATOR ---
            if sedang_ada_posisi:
                # 1. Hitung PnL Berjalan (Unrealized)
                unrealized_pnl = (harga_skrg - harga_entry_sekarang) * lot_sekarang
                
                # 2. Cek apakah harga menyentuh Target Profit atau Stop Loss
                if harga_skrg >= tp_sekarang or harga_skrg <= sl_sekarang:
                    if harga_skrg >= tp_sekarang:
                        status_exit = 'TAKE_PROFIT'
                        harga_exit = tp_sekarang
                    else:
                        status_exit = 'STOP_LOSS'
                        harga_exit = sl_sekarang
                        
                    pnl_final = (harga_exit - harga_entry_sekarang) * lot_sekarang
                    SALDO_VIRTUAL += pnl_final # Update saldo dompet virtual
                    
                    print(f"\n🔔 Order Tereksekusi: {status_exit}!")
                    simpan_ke_csv({'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Simbol': SIMBOL, 'Side': 'LONG', 'Entry': harga_entry_sekarang, 'Exit': harga_exit, 'Lot': lot_sekarang, 'PnL_USD': round(pnl_final, 2), 'Saldo_Akhir': round(SALDO_VIRTUAL, 2), 'Status': status_exit})
                    sedang_ada_posisi = False
                else:
                    tanda = "+" if unrealized_pnl >= 0 else ""
                    print(f"\n⏳ [SIMULASI] POSISI AKTIF | PnL: {tanda}${unrealized_pnl:.2f}")
                    print(f"   Live  : ${harga_skrg:.2f}  (Entry: ${harga_entry_sekarang:.2f})")
                    print(f"   TP    : ${tp_sekarang:.2f}  |  SL: ${sl_sekarang:.2f}")
                    time.sleep(1) 
            
            if not sedang_ada_posisi:
                bar_terakhir = df.iloc[-2] 
                
                skor_teknikal, rincian = hitung_skor_teknikal(bar_terakhir)
                
                data_untuk_ai = df.iloc[-2:][fitur_ai].head(1)
                probabilitas = model_ai_global.predict_proba(data_untuk_ai)[0]
                prob_naik = probabilitas[1] * 100

                total_trade, win_trade, win_rate = hitung_win_rate()

                print(f"\n" + "="*65)
                print(f"📡 RADAR SIMULASI ({TIMEFRAME}) | Lev: {LEVERAGE}x | Live: ${harga_skrg:.2f}")
                print(f"💰 Saldo Virtual: ${SALDO_VIRTUAL:.2f}")
                print(f"🏆 Win Rate     : {win_rate:.1f}% ({win_trade} Win / {total_trade} Trade)")
                print(f"="*65)
                print(f"1️⃣ GATE 1: TEKNIKAL MANUAL")
                for k, v in rincian.items(): print(f"   {k.ljust(22)} : {v}")
                print(f"   SKOR TEKNIKAL        : {skor_teknikal}% (Target: {AMBANG_BATAS_TEKNIKAL}%)")
                
                print(f"\n2️⃣ GATE 2: AI MACHINE LEARNING")
                print(f"   PROBABILITAS NAIK    : {prob_naik:.1f}% (Target: {AMBANG_BATAS_AI}%)")
                print("-" * 65)

                if skor_teknikal >= AMBANG_BATAS_TEKNIKAL and prob_naik >= AMBANG_BATAS_AI:
                    print(f"3️⃣ GATE 3: AI SENTIMEN BERITA")
                    sentimen, rincian_berita = analisis_berita_kilat()
                    for rincian_teks in rincian_berita: print(rincian_teks)
                    print(f"   KESIMPULAN SENTIMEN  : {sentimen}")
                    print("-" * 65)
                    
                    if sentimen == "BEARISH":
                        print("⛔ BATAL BUY: Sentimen Berita Buruk!")
                    else:
                        print(f"🔥 KETIGA GATE LULUS. Eksekusi LONG (BUY) Virtual...")
                        atr = bar_terakhir['ATR']
                        sl = round(harga_skrg - (1.5 * atr), 2)
                        tp = round(harga_skrg + (3.0 * atr), 2)
                        
                        # Hitung Lot dan Margin
                        lot_uang = BATAS_RISIKO_USD * LEVERAGE
                        lot = round(lot_uang / harga_skrg, 3)
                        if lot < 0.001: lot = 0.001
                        
                        margin_dibutuhkan = (lot * harga_skrg) / LEVERAGE
                        
                        if margin_dibutuhkan > SALDO_VIRTUAL:
                            print(f"⚠️ Saldo virtual tidak cukup! Butuh ${margin_dibutuhkan:.2f}, Saldo ${SALDO_VIRTUAL:.2f}")
                        else:
                            # Eksekusi Virtual (Pencatatan Variabel Saja)
                            harga_entry_sekarang = harga_skrg
                            lot_sekarang = lot
                            tp_sekarang = tp
                            sl_sekarang = sl
                            sedang_ada_posisi = True
                            print(f"✅ Posisi Virtual Terbuka pada harga ${harga_entry_sekarang:.2f}")
                else:
                    print("💤 Menunggu formasi yang valid...")
                print("="*65)

        time.sleep(10) # Cek harga setiap 10 detik
    except Exception as e:
        print(f"⚠️ Error Utama: {e}"); time.sleep(5)