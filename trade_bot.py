import os
import csv
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import signal
import sys
from dotenv import load_dotenv
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import urllib3

# Matikan peringatan SSL
urllib3.disable_warnings()

# ==========================================
# 0. MEMUAT AI SENTIMEN (FINBERT)
# ==========================================
print("⏳ Sedang memuat AI FinBERT (Membaca Bahasa Manusia)...")
import feedparser
from transformers import pipeline
ai_sentimen = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ Otak Kanan (AI Sentimen) Siap!")

# ==========================================
# 1. KONFIGURASI API & KONEKSI GANDA
# ==========================================
load_dotenv() 
exchange_data = ccxt.binance()
exchange_trade = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
})
exchange_trade.set_sandbox_mode(True) 

# ==========================================
# 2. PARAMETER STRATEGI
# ==========================================
SIMBOL = 'BTC/USDT'
TIMEFRAME = '5m'       
UKURAN_LOT = 0.005         # <-- Setel lot manual
AMBANG_BATAS_TEKNIKAL = 70 
AMBANG_BATAS_AI = 60.0      

sedang_ada_posisi = False
harga_entry_sekarang = 0
lot_sekarang = 0

# --- VARIABEL HYBRID TRAILING SL ---
id_order_sl_server = None
harga_sl_server_saat_ini = 0
harga_tertinggi_posisi = 0
atr_posisi = 0

model_ai_global = None 
fitur_ai = ['RSI', 'MACD_Line', 'MACD_Hist', 'Jarak_ke_EMA', 'volume']

# ==========================================
# FUNGSI BARU: PASANG SL DI SERVER BINANCE
# ==========================================
def pasang_sl_server(harga_sl):
    try:
        # Menggunakan STOP_LOSS_LIMIT untuk Spot Market Binance
        order = exchange_trade.create_order(
            symbol=SIMBOL,
            type='STOP_LOSS_LIMIT',
            side='sell',
            amount=lot_sekarang,
            price=harga_sl, # Limit price
            params={'stopPrice': harga_sl, 'timeInForce': 'GTC'}
        )
        return order['id']
    except Exception as e:
        print(f"❌ Gagal pasang SL di server: {e}")
        return None

# ==========================================
# 3. FUNGSI LOG, WIN RATE & PROTEKSI
# ==========================================
def simpan_ke_csv(data_trade):
    file_name = 'history_transaksi.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Waktu', 'Simbol', 'Side', 'Entry', 'Exit', 'Lot', 'PnL_USD', 'Status'])
        if not file_exists: writer.writeheader()
        writer.writerow(data_trade)

def hitung_win_rate():
    file_name = 'history_transaksi.csv'
    if not os.path.isfile(file_name):
        return 0, 0, 0.0 
    try:
        df_history = pd.read_csv(file_name)
        if df_history.empty: return 0, 0, 0.0
        
        total_trade = len(df_history)
        win_trade = len(df_history[df_history['PnL_USD'] > 0])
        win_rate = (win_trade / total_trade) * 100
        return total_trade, win_trade, win_rate
    except:
        return 0, 0, 0.0

def tangani_keluar(sig, frame):
    global sedang_ada_posisi, lot_sekarang, harga_entry_sekarang
    print("\n\n🛑 MENERIMA SINYAL BERHENTI...")
    if sedang_ada_posisi:
        print(f"⚠️ Menutup paksa posisi {SIMBOL}...")
        try:
            exchange_trade.cancel_all_orders(SIMBOL)
            harga_akhir = exchange_data.fetch_ticker(SIMBOL)['last']
            exchange_trade.create_market_sell_order(SIMBOL, lot_sekarang)
            pnl_final = (harga_akhir - harga_entry_sekarang) * lot_sekarang
            simpan_ke_csv({'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Simbol': SIMBOL, 'Side': 'LONG', 'Entry': harga_entry_sekarang, 'Exit': harga_akhir, 'Lot': lot_sekarang, 'PnL_USD': round(pnl_final, 2), 'Status': 'FORCE_CLOSE'})
            print(f"💰 POSISI DITUTUP. PnL: ${pnl_final:.2f}")
        except Exception as e: print(f"❌ Gagal tutup otomatis: {e}")
    sys.exit(0)
signal.signal(signal.SIGINT, tangani_keluar)

# ==========================================
# 4. FUNGSI AI PREDIKSI (OTAK KIRI)
# ==========================================
def latih_ai_diawal():
    print("⏳ Menyiapkan Otak Kiri (AI Angka). Mengambil 1000 data historis...")
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
# 5. FUNGSI ANALISIS BERITA (GATE 3)
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
            
            if label == 'POSITIVE': 
                skor_total += 1
                simbol = "🟢"
            elif label == 'NEGATIVE': 
                skor_total -= 1
                simbol = "🔴"
            else: 
                simbol = "⚪"
                
            rincian_berita.append(f"   {simbol} {item.title[:50]}... -> {label}")
            
        if skor_total > 0: kesimpulan = "BULLISH"
        elif skor_total < 0: kesimpulan = "BEARISH"
        else: kesimpulan = "NEUTRAL"
        
        return kesimpulan, rincian_berita
    except Exception as e:
        return "NEUTRAL", [f"   ⚠️ Gagal mengambil berita: {e}"]

# ==========================================
# 6. FUNGSI ANALISIS LIVE (GATE 1)
# ==========================================
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
# 7. LOOP UTAMA (TRIPLE ENGINE + HYBRID TRAILING)
# ==========================================
model_ai_global = latih_ai_diawal()
print(f"\n🚀 Bot v7.2 HYBRID TRAILING EDITION Aktif di {SIMBOL}...")

while True:
    try:
        df = ambil_data_live()
        try: current_usdt = exchange_trade.fetch_balance()['total']['USDT']
        except: current_usdt = 0.0
        
        if df is not None:
            harga_skrg = df.iloc[-1]['close']
            
            # --- JIKA SEDANG ADA POSISI (MODE PELACAKAN) ---
            if sedang_ada_posisi:
                # 1. Cek apakah SL di server sudah tereksekusi
                if id_order_sl_server:
                    try:
                        order_info = exchange_trade.fetch_order(id_order_sl_server, SIMBOL)
                        if order_info['status'] == 'closed' or order_info['status'] == 'canceled':
                            harga_exit = order_info['average'] if order_info.get('average') else harga_sl_server_saat_ini
                            pnl_akhir = (harga_exit - harga_entry_sekarang) * lot_sekarang
                            simpan_ke_csv({'Waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Simbol': SIMBOL, 'Side': 'LONG', 'Entry': harga_entry_sekarang, 'Exit': harga_exit, 'Lot': lot_sekarang, 'PnL_USD': round(pnl_akhir, 2), 'Status': 'CLOSED_BY_TRAILING_SL'})
                            
                            print(f"\n🔔 POSISI CLOSED! Terkena SL di Server (Profit Terakhir: ${pnl_akhir:.2f})")
                            sedang_ada_posisi = False
                            id_order_sl_server = None
                            continue
                    except Exception as e:
                        pass # Abaikan jika gagal cek API sesaat

                # 2. Update Harga Tertinggi & Geser SL (Trailing)
                if harga_skrg > harga_tertinggi_posisi:
                    harga_tertinggi_posisi = harga_skrg
                    
                    # Calon SL Baru: Puncak dikurangi 0.5x ATR
                    calon_sl_baru = round(harga_tertinggi_posisi - (0.5 * atr_posisi), 2)
                    
                    if calon_sl_baru > harga_sl_server_saat_ini:
                        print(f"🔄 Menggeser Garis Jaring SL Server naik ke: ${calon_sl_baru}")
                        try:
                            # Cancel SL lama
                            if id_order_sl_server:
                                exchange_trade.cancel_order(id_order_sl_server, SIMBOL)
                            # Pasang SL Baru
                            id_baru = pasang_sl_server(calon_sl_baru)
                            if id_baru:
                                id_order_sl_server = id_baru
                                harga_sl_server_saat_ini = calon_sl_baru
                        except Exception as e:
                            print(f"⚠️ Gagal update SL di Binance: {e}")

                profit_usd = (harga_skrg - harga_entry_sekarang) * lot_sekarang
                tanda = "+" if profit_usd >= 0 else ""
                
                print(f"\n🛡️ HYBRID TRAILING AKTIF | Profit: {tanda}${profit_usd:.2f}")
                print(f"   Live  : ${harga_skrg:.2f}  (Entry: ${harga_entry_sekarang:.2f})")
                print(f"   Puncak: ${harga_tertinggi_posisi:.2f}  |  SL Server: ${harga_sl_server_saat_ini:.2f}")
                time.sleep(3) 
            
            # --- JIKA TIDAK ADA POSISI (MODE RADAR) ---
            if not sedang_ada_posisi:
                bar_terakhir = df.iloc[-2] 
                
                skor_teknikal, rincian = hitung_skor_teknikal(bar_terakhir)
                
                data_untuk_ai = df.iloc[-2:][fitur_ai].head(1)
                probabilitas = model_ai_global.predict_proba(data_untuk_ai)[0]
                prob_naik = probabilitas[1] * 100

                total_trade, win_trade, win_rate = hitung_win_rate()

                print(f"\n" + "="*65)
                print(f"📡 RADAR 3-GATE PASAR ({TIMEFRAME}) | Live: ${harga_skrg:.2f}")
                print(f"💰 Balance : ${current_usdt:.2f}")
                print(f"🏆 Win Rate: {win_rate:.1f}% ({win_trade} Win / {total_trade} Trade)")
                print(f"="*65)
                print(f"1️⃣ GATE 1: TEKNIKAL MANUAL")
                for k, v in rincian.items(): print(f"   {k.ljust(22)} : {v}")
                print(f"   SKOR TEKNIKAL        : {skor_teknikal}% (Target: {AMBANG_BATAS_TEKNIKAL}%)")
                
                print(f"\n2️⃣ GATE 2: AI MACHINE LEARNING")
                print(f"   PROBABILITAS NAIK    : {prob_naik:.1f}% (Target: {AMBANG_BATAS_AI}%)")
                print("-" * 65)

                if skor_teknikal >= AMBANG_BATAS_TEKNIKAL and prob_naik >= AMBANG_BATAS_AI:
                    
                    print(f"3️⃣ GATE 3: AI SENTIMEN BERITA (Tugas Terakhir)")
                    print("   📡 Sinyal Terdeteksi! Mengecek berita global detik ini...")
                    sentimen, rincian_berita = analisis_berita_kilat()
                    
                    for rincian_teks in rincian_berita: print(rincian_teks)
                    print(f"   KESIMPULAN SENTIMEN  : {sentimen}")
                    print("-" * 65)
                    
                    if sentimen == "BEARISH":
                        print("⛔ BATAL BUY: Teknikal & ML mendukung, tapi BERITA SEDANG BURUK!")
                    else:
                        print(f"🔥 ULTIMATE CONFLUENCE! KETIGA GATE LULUS. Eksekusi BUY...")
                        lot = UKURAN_LOT
                        
                        if (lot * harga_skrg) > current_usdt:
                            print(f"⚠️ Saldo simulasi tidak cukup!")
                        else:
                            try:
                                order_buy = exchange_trade.create_market_buy_order(SIMBOL, lot)
                                
                                # SETUP AWAL HYBRID TRAILING
                                harga_entry_sekarang = order_buy['average'] if order_buy.get('average') else harga_skrg
                                lot_sekarang = lot
                                atr_posisi = bar_terakhir['ATR']
                                
                                # Stop Loss keras awal diatur 0.8x ATR ke bawah dari titik entry
                                harga_sl_awal = round(harga_entry_sekarang - (0.8 * atr_posisi), 2)
                                
                                # Pasang SL ke Server
                                id_sl = pasang_sl_server(harga_sl_awal)
                                
                                if id_sl:
                                    id_order_sl_server = id_sl
                                    harga_sl_server_saat_ini = harga_sl_awal
                                    harga_tertinggi_posisi = harga_entry_sekarang
                                    sedang_ada_posisi = True
                                    print(f"✅ Beli Berhasil & SL Pengaman Server Aktif di ${harga_sl_awal}")
                                else:
                                    print("⚠️ Gagal pasang SL server, posisi ditutup otomatis demi keamanan.")
                                    exchange_trade.create_market_sell_order(SIMBOL, lot)
                            except Exception as e:
                                print(f"❌ Gagal Eksekusi: {e}")
                else:
                    if skor_teknikal < AMBANG_BATAS_TEKNIKAL:
                        print("💤 Gate 1 Gagal. Menunggu formasi teknikal...")
                    else:
                        print("💤 Gate 2 Gagal. AI Prediksi merasa pasar akan turun.")
                print("="*65)

        time.sleep(10)
    except Exception as e:
        print(f"⚠️ Error Utama: {e}"); time.sleep(5)