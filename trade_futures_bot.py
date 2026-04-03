import os
import time
import requests
import pandas as pd
import pandas_ta as ta
import signal
import sys
from dotenv import load_dotenv
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from supabase import create_client, Client
import urllib3

urllib3.disable_warnings()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ==========================================
# 1. IMPORT BINANCE FUTURES CONNECTOR
# ==========================================
# pip install binance-futures-connector supabase
from binance.um_futures import UMFutures
from binance.error import ClientError

load_dotenv()

TESTNET_URL    = "https://testnet.binancefuture.com"
API_KEY        = os.getenv('BINANCE_DEMO_API_KEY')
API_SECRET     = os.getenv('BINANCE_DEMO_SECRET_KEY')
SUPABASE_URL   = os.getenv('SUPABASE_URL')
SUPABASE_KEY   = os.getenv('SUPABASE_KEY')

# Supabase client — data history tersimpan permanen di cloud
# Tidak akan hilang saat Railway redeploy / restart
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase terhubung!")
except Exception as e:
    print(f"❌ Gagal koneksi Supabase: {e}")
    print("   Pastikan SUPABASE_URL dan SUPABASE_KEY sudah diisi di .env")
    sys.exit(1)

client_data  = UMFutures(base_url=TESTNET_URL)
client_trade = UMFutures(key=API_KEY, secret=API_SECRET, base_url=TESTNET_URL)

# ==========================================
# 2. PARAMETER STRATEGI
# ==========================================
SIMBOL         = 'BTCUSDT'
SIMBOL_DISPLAY = 'BTC/USDT'
TIMEFRAME      = '5m'     # ← kembali ke 5 menit
LEVERAGE       = 10
MARGIN_USDT    = 10.0

AMBANG_TEKNIKAL = 65
AMBANG_AI       = 60.0

# SL & Trailing dikembalikan ke nilai 5m
TRAILING_ATR_MULTIPLIER = 0.5   # was 0.3 di 1m
SL_AWAL_ATR_MULTIPLIER  = 0.8   # was 0.5 di 1m

# RSI konsistensi — di 5m cukup 3 candle (= 15 menit)
RSI_KONSISTENSI_CANDLE = 3      # was 5 di 1m

# EMA setting — di 5m pakai EMA lebih panjang
EMA_PENDEK  = 50    # was 20 di 1m
EMA_PANJANG = 200   # was 89 di 1m

# MACD setting — kembali ke default yang cocok untuk 5m
MACD_FAST   = 12    # was 5 di 1m
MACD_SLOW   = 26    # was 13 di 1m
MACD_SIGNAL = 9     # was 4 di 1m

# Stochastic setting — kembali ke standard
STOCH_K = 14        # was 5 di 1m
STOCH_D = 3
STOCH_S = 3

# Loop delay — di 5m cukup 10 detik
LOOP_DELAY = 10     # was 2 di 1m

# ── Threshold Gate 3 (Sentimen Realtime) ─────────────────────────────────────
# Fear & Greed: 0-100 (0=Extreme Fear, 100=Extreme Greed)
FNG_GREED_THRESHOLD = 65   # > ini = pasar terlalu greedy → SHORT lebih aman
FNG_FEAR_THRESHOLD  = 35   # < ini = pasar terlalu takut → LONG lebih aman

# Funding Rate: positif = longs bayar shorts (bullish berlebihan)
FUNDING_BULLISH_THRESHOLD = 0.0005   # > ini = pasar overbullish → SHORT lebih aman
FUNDING_BEARISH_THRESHOLD = -0.0005  # < ini = pasar overbearish → LONG lebih aman

# Long/Short Ratio: > 1 artinya lebih banyak LONG, < 1 lebih banyak SHORT
LS_RATIO_LONG_HEAVY  = 1.5   # > ini = terlalu banyak long → SHORT lebih aman
LS_RATIO_SHORT_HEAVY = 0.7   # < ini = terlalu banyak short → LONG lebih aman

# ── Cache sentimen (biar tidak di-fetch setiap 10 detik) ─────────────────────
_cache_sentimen = {
    'data'        : None,
    'last_fetch'  : 0,
    'ttl_detik'   : 60    # refresh setiap 60 detik
}

# --- STATE ---
sedang_ada_posisi        = False
arah_posisi              = None
harga_entry_sekarang     = 0
lot_sekarang             = 0
id_order_sl_server       = None
harga_sl_server_saat_ini = 0
harga_ekstrem_posisi     = 0
atr_posisi               = 0
model_ai_global          = None

fitur_ai = ['RSI', 'MACD_Line', 'MACD_Hist', 'Jarak_ke_EMA', 'volume', 'OBV', 'STOCH_K']

# ==========================================
# HELPER
# ==========================================
def parse_resp(resp):
    if isinstance(resp, dict):
        return resp
    try:
        return resp.json()
    except Exception:
        return {}

def ohlcv_ke_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

# ==========================================
# SETUP LEVERAGE + CEK KONEKSI
# ==========================================
def setup_leverage():
    print("🔌 Menghubungkan ke Binance Futures Testnet...")
    if not API_KEY or not API_SECRET:
        print("\n❌ API KEY tidak ditemukan di .env!")
        print("   BINANCE_DEMO_API_KEY=napikeykamu")
        print("   BINANCE_DEMO_SECRET_KEY=nsecretkamu")
        print("   → https://testnet.binancefuture.com (login GitHub)")
        sys.exit(1)

    print(f"   [DEBUG] Key: {API_KEY[:6]}...{API_KEY[-4:]}")
    try:
        resp = client_trade.account()
        data = parse_resp(resp)
        usdt = 0.0
        for asset in data.get('assets', []):
            if asset.get('asset') == 'USDT':
                usdt = float(asset.get('walletBalance', 0))
                break
        print(f"✅ Koneksi berhasil! Saldo USDT: ${usdt:.2f}")
    except ClientError as e:
        print(f"\n❌ AUTENTIKASI GAGAL: {e}")
        print("   → Key harus dari https://testnet.binancefuture.com (login GitHub)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Gagal koneksi: {e}")
        sys.exit(1)

    try:
        client_trade.change_leverage(symbol=SIMBOL, leverage=LEVERAGE)
        print(f"✅ Leverage: {LEVERAGE}x")
    except Exception as e:
        print(f"⚠️ Set leverage: {e} (lanjut...)")

# ==========================================
# HITUNG LOT & SL
# ==========================================
def hitung_lot(harga):
    return round((MARGIN_USDT * LEVERAGE) / harga, 3)

def pasang_sl_server(harga_sl, arah):
    try:
        resp = client_trade.new_order(
            symbol=SIMBOL,
            side='SELL' if arah == 'LONG' else 'BUY',
            type='STOP_MARKET',
            quantity=lot_sekarang,
            stopPrice=harga_sl,
            positionSide='LONG' if arah == 'LONG' else 'SHORT',
            timeInForce='GTC',
            reduceOnly='true'
        )
        return parse_resp(resp).get('orderId')
    except Exception as e:
        print(f"❌ Gagal pasang SL: {e}")
        return None

# ==========================================
# 3. LOG & PROTEKSI — pakai Supabase
# ==========================================
def simpan_ke_db(data_trade):
    """
    Simpan record trade ke Supabase (PostgreSQL).
    Data permanen — tidak hilang saat Railway redeploy/restart.
    """
    try:
        supabase.table('history_transaksi').insert({
            'waktu'      : data_trade['Waktu'],
            'simbol'     : data_trade['Simbol'],
            'side'       : data_trade['Side'],
            'entry'      : data_trade['Entry'],
            'exit'       : data_trade['Exit'],
            'lot'        : data_trade['Lot'],
            'leverage'   : data_trade['Leverage'],
            'margin_usdt': data_trade['Margin_USDT'],
            'pnl_usd'    : data_trade['PnL_USD'],
            'pnl_persen' : data_trade['PnL_Persen'],
            'status'     : data_trade['Status'],
        }).execute()
    except Exception as e:
        # Kalau Supabase gagal, fallback ke CSV lokal sebagai backup
        print(f"⚠️ Gagal simpan ke Supabase: {e} — backup ke CSV lokal")
        try:
            import csv
            file_name   = 'history_backup.csv'
            file_exists = os.path.isfile(file_name)
            with open(file_name, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data_trade.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_trade)
        except Exception as e2:
            print(f"❌ Backup CSV juga gagal: {e2}")

def hitung_win_rate():
    """
    Hitung win rate dari data di Supabase.
    """
    try:
        resp  = supabase.table('history_transaksi').select('pnl_usd').execute()
        data  = resp.data
        if not data:
            return 0, 0, 0.0
        total = len(data)
        win   = sum(1 for row in data if float(row['pnl_usd']) > 0)
        return total, win, (win / total) * 100
    except Exception as e:
        print(f"⚠️ Gagal ambil win rate dari Supabase: {e}")
        return 0, 0, 0.0

def hitung_pnl(harga_exit, harga_entry, lot, arah):
    return (harga_exit - harga_entry) * lot if arah == 'LONG' else (harga_entry - harga_exit) * lot

def tutup_posisi_darurat(harga_akhir, status='FORCE_CLOSE'):
    global sedang_ada_posisi, id_order_sl_server, arah_posisi
    try:
        client_trade.cancel_open_orders(symbol=SIMBOL)
        client_trade.new_order(
            symbol=SIMBOL,
            side='SELL' if arah_posisi == 'LONG' else 'BUY',
            type='MARKET', quantity=lot_sekarang,
            positionSide='LONG' if arah_posisi == 'LONG' else 'SHORT',
            reduceOnly='true'
        )
        pnl = hitung_pnl(harga_akhir, harga_entry_sekarang, lot_sekarang, arah_posisi)
        pct = (pnl / MARGIN_USDT) * 100
        simpan_ke_db({
            'Waktu'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Simbol'     : SIMBOL_DISPLAY,
            'Side'       : arah_posisi,
            'Entry'      : harga_entry_sekarang,
            'Exit'       : harga_akhir,
            'Lot'        : lot_sekarang,
            'Leverage'   : LEVERAGE,
            'Margin_USDT': MARGIN_USDT,
            'PnL_USD'    : round(pnl, 2),
            'PnL_Persen' : round(pct, 2),
            'Status'     : status
        })
        print(f"💰 Posisi Ditutup. PnL: ${pnl:.2f} ({pct:+.1f}%)")
        sedang_ada_posisi = False; id_order_sl_server = None; arah_posisi = None
    except Exception as e:
        print(f"❌ Gagal tutup paksa: {e}")

def tangani_keluar(sig, frame):
    print("\n\n🛑 BERHENTI...")
    if sedang_ada_posisi:
        try:
            harga_akhir = float(parse_resp(client_data.ticker_price(symbol=SIMBOL)).get('price', harga_entry_sekarang))
        except Exception:
            harga_akhir = harga_entry_sekarang
        tutup_posisi_darurat(harga_akhir)
    sys.exit(0)

signal.signal(signal.SIGINT, tangani_keluar)

# ==========================================
# 4. AI PREDIKSI (OTAK KIRI)
# ==========================================
def latih_ai_diawal():
    print(f"⏳ Melatih AI... Mengambil 1000 data historis ({TIMEFRAME})...")
    klines = client_data.klines(symbol=SIMBOL, interval=TIMEFRAME, limit=1000)
    df     = ohlcv_ke_dataframe(klines)

    # Gunakan setting yang sudah disesuaikan untuk 1m
    df['RSI']          = ta.rsi(df['close'], length=14)
    macd               = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd_col           = f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    macd_hist_col      = f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    df['MACD_Line']    = macd[macd_col]
    df['MACD_Hist']    = macd[macd_hist_col]
    df['EMA_200']      = ta.ema(df['close'], length=EMA_PANJANG)
    df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
    df['OBV']          = ta.obv(df['close'], df['volume'])
    stoch_key          = f'STOCHk_{STOCH_K}_{STOCH_D}_{STOCH_S}'
    stoch              = ta.stoch(df['high'], df['low'], df['close'], k=STOCH_K, d=STOCH_D, smooth_k=STOCH_S)
    df['STOCH_K']      = stoch[stoch_key] if stoch is not None and stoch_key in stoch.columns else 50
    df['Target_Naik']  = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(df[fitur_ai], df['Target_Naik'])
    print("✅ AI Siap!")
    return model

# ==========================================
# 5. GATE 3 — SENTIMEN REALTIME (BARU)
# ==========================================
# Mengganti RSS+FinBERT yang lag berjam-jam dengan 3 sumber data realtime:
#   A) Fear & Greed Index (CoinyBubble) — update tiap ~1 menit, gratis, no key
#   B) Funding Rate (Binance) — update tiap beberapa menit, no key
#   C) Long/Short Ratio (Binance) — update tiap 5 menit, no key
#
# Logika:
#   LONG cocok  → pasar Fear (orang takut = buy opportunity) + funding negatif + banyak short
#   SHORT cocok → pasar Greed (orang serakah = koreksi datang) + funding positif + banyak long

def ambil_fear_greed():
    """
    Ambil Fear & Greed Index dari CoinyBubble.
    Update ~1 menit. Gratis, tanpa API key.
    Return: (nilai int 0-100, label str, status str)
    """
    try:
        resp = requests.get(
            "https://coinybubble.com/api/v1/fear-greed",
            timeout=5
        )
        if resp.status_code == 200:
            data  = resp.json()
            nilai = int(data.get('value', 50))
            label = data.get('value_classification', 'Neutral')
            return nilai, label, "OK"
    except Exception:
        pass

    # Fallback: Alternative.me (update harian tapi tetap berguna sebagai konfirmasi makro)
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=5
        )
        if resp.status_code == 200:
            item  = resp.json()['data'][0]
            nilai = int(item['value'])
            label = item['value_classification']
            return nilai, label, "OK (fallback alternative.me)"
    except Exception:
        pass

    return 50, "Neutral", "GAGAL"

def ambil_funding_rate():
    """
    Ambil funding rate terbaru dari Binance Futures.
    Positif  → longs bayar shorts → pasar overbullish
    Negatif  → shorts bayar longs → pasar overbearish
    Return: (rate float, status str)
    """
    try:
        resp = client_data.mark_price(symbol=SIMBOL)
        data = parse_resp(resp)
        rate = float(data.get('lastFundingRate', 0))
        return rate, "OK"
    except Exception as e:
        return 0.0, f"GAGAL: {e}"

def ambil_long_short_ratio():
    """
    Ambil Global Long/Short Account Ratio dari Binance.
    > 1 = lebih banyak akun LONG, < 1 = lebih banyak SHORT.
    Return: (ratio float, long_pct float, short_pct float, status str)
    """
    try:
        resp = client_data.global_long_short_account_ratio(
            symbol=SIMBOL,
            period='5m',
            limit=1
        )
        if resp and len(resp) > 0:
            item      = resp[0]
            ls_ratio  = float(item.get('longShortRatio', 1.0))
            long_pct  = float(item.get('longAccount', 0.5)) * 100
            short_pct = float(item.get('shortAccount', 0.5)) * 100
            return ls_ratio, long_pct, short_pct, "OK"
    except Exception as e:
        pass
    return 1.0, 50.0, 50.0, "GAGAL"

def analisis_sentimen_realtime(arah_sinyal):
    """
    Gate 3 baru: gabungkan 3 sumber data realtime.

    Sistem poin:
      Setiap sumber bisa kasih +1 (mendukung arah) / -1 (berlawanan) / 0 (netral)
      Skor akhir: -3 sampai +3
        >= +1 → COCOK (lanjut trade)
        <= -1 → BERLAWANAN (batal)
        0    → NETRAL (lanjut trade, tapi warning)

    Return: (cocok bool, skor int, rincian list)
    """
    global _cache_sentimen

    sekarang = time.time()

    # Gunakan cache kalau belum expired
    if (_cache_sentimen['data'] is not None and
            sekarang - _cache_sentimen['last_fetch'] < _cache_sentimen['ttl_detik']):
        fng_nilai, fng_label, fng_status, \
        fund_rate, fund_status, \
        ls_ratio, long_pct, short_pct, ls_status = _cache_sentimen['data']
    else:
        fng_nilai, fng_label, fng_status       = ambil_fear_greed()
        fund_rate, fund_status                  = ambil_funding_rate()
        ls_ratio, long_pct, short_pct, ls_status = ambil_long_short_ratio()

        _cache_sentimen['data'] = (
            fng_nilai, fng_label, fng_status,
            fund_rate, fund_status,
            ls_ratio, long_pct, short_pct, ls_status
        )
        _cache_sentimen['last_fetch'] = sekarang

    skor    = 0
    rincian = []

    # ── A. FEAR & GREED INDEX ────────────────────────────────────────────────
    if arah_sinyal == 'LONG':
        if fng_nilai <= FNG_FEAR_THRESHOLD:
            skor += 1
            fng_simbol = "🟢"
            fng_note   = "FEAR → Peluang LONG (beli saat orang takut)"
        elif fng_nilai >= FNG_GREED_THRESHOLD:
            skor -= 1
            fng_simbol = "🔴"
            fng_note   = "GREED → Kurang ideal untuk LONG"
        else:
            fng_simbol = "⚪"
            fng_note   = "NEUTRAL"
    else:  # SHORT
        if fng_nilai >= FNG_GREED_THRESHOLD:
            skor += 1
            fng_simbol = "🟢"
            fng_note   = "GREED → Pasar overbullish, SHORT ideal"
        elif fng_nilai <= FNG_FEAR_THRESHOLD:
            skor -= 1
            fng_simbol = "🔴"
            fng_note   = "FEAR → Kurang ideal untuk SHORT"
        else:
            fng_simbol = "⚪"
            fng_note   = "NEUTRAL"

    rincian.append(
        f"   {fng_simbol} F&G Index : {fng_nilai}/100 ({fng_label}) → {fng_note}"
        + (f" [{fng_status}]" if fng_status != "OK" else "")
    )

    # ── B. FUNDING RATE ──────────────────────────────────────────────────────
    fund_pct = fund_rate * 100
    if arah_sinyal == 'LONG':
        if fund_rate <= FUNDING_BEARISH_THRESHOLD:
            skor += 1
            fr_simbol = "🟢"
            fr_note   = f"Negatif ({fund_pct:.4f}%) → shorts bayar longs, LONG menguntungkan"
        elif fund_rate >= FUNDING_BULLISH_THRESHOLD:
            skor -= 1
            fr_simbol = "🔴"
            fr_note   = f"Positif tinggi ({fund_pct:.4f}%) → LONG harus bayar, kurang ideal"
        else:
            fr_simbol = "⚪"
            fr_note   = f"Netral ({fund_pct:.4f}%)"
    else:  # SHORT
        if fund_rate >= FUNDING_BULLISH_THRESHOLD:
            skor += 1
            fr_simbol = "🟢"
            fr_note   = f"Positif ({fund_pct:.4f}%) → longs bayar shorts, SHORT menguntungkan"
        elif fund_rate <= FUNDING_BEARISH_THRESHOLD:
            skor -= 1
            fr_simbol = "🔴"
            fr_note   = f"Negatif ({fund_pct:.4f}%) → SHORT harus bayar, kurang ideal"
        else:
            fr_simbol = "⚪"
            fr_note   = f"Netral ({fund_pct:.4f}%)"

    rincian.append(
        f"   {fr_simbol} Funding Rate: {fr_note}"
        + (f" [{fund_status}]" if fund_status != "OK" else "")
    )

    # ── C. LONG/SHORT RATIO ──────────────────────────────────────────────────
    if arah_sinyal == 'LONG':
        if ls_ratio <= LS_RATIO_SHORT_HEAVY:
            skor += 1
            ls_simbol = "🟢"
            ls_note   = f"Banyak SHORT ({short_pct:.1f}%) → potensi short squeeze, LONG oke"
        elif ls_ratio >= LS_RATIO_LONG_HEAVY:
            skor -= 1
            ls_simbol = "🔴"
            ls_note   = f"Terlalu banyak LONG ({long_pct:.1f}%) → crowded, risiko"
        else:
            ls_simbol = "⚪"
            ls_note   = f"Seimbang (Long {long_pct:.1f}% / Short {short_pct:.1f}%)"
    else:  # SHORT
        if ls_ratio >= LS_RATIO_LONG_HEAVY:
            skor += 1
            ls_simbol = "🟢"
            ls_note   = f"Terlalu banyak LONG ({long_pct:.1f}%) → crowded, SHORT ideal"
        elif ls_ratio <= LS_RATIO_SHORT_HEAVY:
            skor -= 1
            ls_simbol = "🔴"
            ls_note   = f"Banyak SHORT ({short_pct:.1f}%) → crowded short, kurang ideal"
        else:
            ls_simbol = "⚪"
            ls_note   = f"Seimbang (Long {long_pct:.1f}% / Short {short_pct:.1f}%)"

    rincian.append(
        f"   {ls_simbol} L/S Ratio  : {ls_ratio:.2f} → {ls_note}"
        + (f" [{ls_status}]" if ls_status != "OK" else "")
    )

    # ── Kesimpulan ────────────────────────────────────────────────────────────
    if skor >= 1:
        kesimpulan = "COCOK ✅"
        cocok      = True
    elif skor <= -1:
        kesimpulan = "BERLAWANAN ❌"
        cocok      = False
    else:
        kesimpulan = "NETRAL ⚠️ (lanjut dengan hati-hati)"
        cocok      = True   # Netral = boleh lanjut tapi dengan warning

    rincian.append(f"   📊 Skor Sentimen: {skor:+d}/3 → {kesimpulan}")
    return cocok, skor, rincian

# ==========================================
# 6. ANALISIS LIVE (GATE 1)
# ==========================================
def ambil_data_live():
    try:
        klines = client_data.klines(symbol=SIMBOL, interval=TIMEFRAME, limit=500)
        df     = ohlcv_ke_dataframe(klines)

        # EMA disesuaikan untuk 1m
        df['EMA_50']  = ta.ema(df['close'], length=EMA_PENDEK)
        df['EMA_200'] = ta.ema(df['close'], length=EMA_PANJANG)
        df['RSI_14']  = ta.rsi(df['close'], length=14)
        df['ATR']     = ta.atr(df['high'], df['low'], df['close'], length=14)

        # MACD cepat untuk 1m
        macd_col      = f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
        macd_sig_col  = f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
        macd_hist_col = f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
        macd_data = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd_data is not None:
            df = pd.concat([df, macd_data], axis=1)
        df['MACD_12_26_9']  = df.get(macd_col,      pd.Series(0, index=df.index))
        df['MACDs_12_26_9'] = df.get(macd_sig_col,  pd.Series(0, index=df.index))
        df['MACDh_12_26_9'] = df.get(macd_hist_col, pd.Series(0, index=df.index))

        # OBV — satu-satunya volume indicator yang dipertahankan
        df['OBV']     = ta.obv(df['close'], df['volume'])
        df['OBV_EMA'] = ta.ema(df['OBV'], length=20)

        # Alias untuk AI
        df['RSI']          = df['RSI_14']
        df['MACD_Line']    = df['MACD_12_26_9']
        df['MACD_Hist']    = df['MACDh_12_26_9']
        df['Jarak_ke_EMA'] = df['close'] - df['EMA_200']
        df['STOCH_K']      = 50   # placeholder agar fitur_ai tidak error
        return df
    except Exception as e:
        print(f"⚠️ Gagal ambil data: {e}")
        return None

def hitung_skor_teknikal(bar, df, idx):
    """
    Gate 1 — Lean 3-Indicator System (sinkron dengan ML)

    Dipilih berdasarkan konsistensi dengan ML prediction:

    BULLISH (LONG):              BEARISH (SHORT):
    ─────────────────────        ──────────────────────
    1. EMA Trend  (30-40)        1. EMA Trend  (30-40)
    2. MACD       (25-35)        2. MACD       (25-35)
    3. OBV        (20-25)        3. RSI OB↓    (25-35)

    Total base max = 100, confluence bonus bisa tambah hingga +30.

    Kenapa 3 saja?
    - ML belajar dari pola harga & volume — EMA, MACD, OBV sangat dekat
      dengan fitur yang ML gunakan, jadi keduanya cenderung agree
    - RSI OB hanya untuk SHORT karena ML sangat akurat di kondisi
      overbought reversal (pola sangat jelas di data historis)
    - Stochastic, Ichimoku, Divergence dihapus — terlalu noise di 1m
      dan sering jadi sumber konflik dengan ML
    """
    rincian_long  = {}
    rincian_short = {}
    skor_long     = 0
    skor_short    = 0
    confluence_long  = 0
    confluence_short = 0

    h      = bar['close']
    ema50  = bar['EMA_50']
    ema200 = bar['EMA_200']

    macd_line = bar.get('MACD_12_26_9',  0)
    macd_sig  = bar.get('MACDs_12_26_9', 0)
    macd_hist = bar.get('MACDh_12_26_9', 0)
    rsi       = bar['RSI_14']
    rsi_c1    = df.iloc[idx - 1]['RSI_14'] if idx >= 1 else rsi
    rsi_cn    = df.iloc[idx - RSI_KONSISTENSI_CANDLE]['RSI_14'] if idx >= RSI_KONSISTENSI_CANDLE else rsi
    obv       = bar.get('OBV',     0)
    obv_ema   = bar.get('OBV_EMA', 0)

    MARGIN_EMA      = 0.0015
    jarak_h_ema50   = ((h - ema50)  / ema50)  * 100
    jarak_ema50_200 = ((ema50 - ema200) / ema200) * 100

    # Rata-rata histogram MACD 20 candle terakhir sebagai baseline intensity
    col_hist   = 'MACDh_12_26_9'
    hist_mean  = df[col_hist].iloc[max(0, idx-20):idx].abs().mean() if col_hist in df.columns and idx >= 20 else 1
    hist_ratio = abs(macd_hist) / hist_mean if hist_mean > 0 else 1.0

    obv_diverge_pct = ((obv - obv_ema) / abs(obv_ema)) * 100 if obv_ema != 0 else 0

    # =========================================================================
    # BULLISH — 3 indikator
    # =========================================================================

    # 1. EMA Trend LONG (bobot 30-40)
    ema_long = (h > ema50 * (1 + MARGIN_EMA)) and (ema50 > ema200 * (1 + MARGIN_EMA))
    if ema_long:
        d = abs(jarak_h_ema50)
        if d > 1.0:   bobot = 40; lvl = "💪 SANGAT KUAT"
        elif d > 0.5: bobot = 35; lvl = "✊ KUAT"
        else:         bobot = 30; lvl = "👌 NORMAL"
        skor_long += bobot; confluence_long += 1
        rincian_long['📈 EMA'] = f"🟢 UPTREND {lvl} (+{bobot}) | H+{jarak_h_ema50:.2f}% EMA50+{jarak_ema50_200:.2f}%"
    else:
        rincian_long['📈 EMA'] = f"⚪ Tidak aktif | H-EMA50: {jarak_h_ema50:+.2f}%"

    # 2. MACD LONG (bobot 25-35)
    macd_long = macd_line > macd_sig and macd_hist > 0
    if macd_long:
        if hist_ratio > 2.0:   bobot = 35; lvl = f"💪 SANGAT KUAT ({hist_ratio:.1f}x)"
        elif hist_ratio > 1.0: bobot = 30; lvl = f"✊ KUAT ({hist_ratio:.1f}x)"
        else:                  bobot = 25; lvl = f"👌 NORMAL ({hist_ratio:.1f}x)"
        skor_long += bobot; confluence_long += 1
        rincian_long['📈 MACD'] = f"🟢 BULLISH {lvl} (+{bobot})"
    else:
        rincian_long['📈 MACD'] = f"⚪ Tidak aktif | hist: {macd_hist:.2f}"

    # 3. OBV LONG (bobot 20-25)
    obv_long = obv > obv_ema
    if obv_long:
        d = abs(obv_diverge_pct)
        if d > 5:   bobot = 25; lvl = f"💪 KUAT (+{obv_diverge_pct:.1f}%)"
        elif d > 2: bobot = 22; lvl = f"✊ SEDANG (+{obv_diverge_pct:.1f}%)"
        else:       bobot = 20; lvl = f"👌 LEMAH (+{obv_diverge_pct:.1f}%)"
        skor_long += bobot; confluence_long += 1
        rincian_long['📈 OBV'] = f"🟢 TEKANAN BELI {lvl} (+{bobot})"
    else:
        rincian_long['📈 OBV'] = f"⚪ Tidak aktif | OBV-EMA: {obv_diverge_pct:.1f}%"

    # =========================================================================
    # BEARISH — 3 indikator
    # =========================================================================

    # 1. EMA Trend SHORT (bobot 30-40)
    ema_short = (h < ema50 * (1 - MARGIN_EMA)) and (ema50 < ema200 * (1 - MARGIN_EMA))
    if ema_short:
        d = abs(jarak_h_ema50)
        if d > 1.0:   bobot = 40; lvl = "💪 SANGAT KUAT"
        elif d > 0.5: bobot = 35; lvl = "✊ KUAT"
        else:         bobot = 30; lvl = "👌 NORMAL"
        skor_short += bobot; confluence_short += 1
        rincian_short['📉 EMA'] = f"🔴 DOWNTREND {lvl} (+{bobot}) | H{jarak_h_ema50:.2f}% EMA50{jarak_ema50_200:.2f}%"
    else:
        rincian_short['📉 EMA'] = f"⚪ Tidak aktif | H-EMA50: {jarak_h_ema50:+.2f}%"

    # 2. MACD SHORT (bobot 25-35)
    macd_short = macd_line < macd_sig and macd_hist < 0
    if macd_short:
        if hist_ratio > 2.0:   bobot = 35; lvl = f"💪 SANGAT KUAT ({hist_ratio:.1f}x)"
        elif hist_ratio > 1.0: bobot = 30; lvl = f"✊ KUAT ({hist_ratio:.1f}x)"
        else:                  bobot = 25; lvl = f"👌 NORMAL ({hist_ratio:.1f}x)"
        skor_short += bobot; confluence_short += 1
        rincian_short['📉 MACD'] = f"🔴 BEARISH {lvl} (+{bobot})"
    else:
        rincian_short['📉 MACD'] = f"⚪ Tidak aktif | hist: {macd_hist:.2f}"

    # 3. RSI Overbought SHORT (bobot 25-35)
    # Ini indikator paling aligned dengan ML untuk SHORT
    rsi_turun_konsisten = rsi < rsi_c1 and rsi_c1 < rsi_cn
    rsi_short_ok = rsi > 65 and rsi_turun_konsisten
    if rsi_short_ok:
        if rsi > 80:   bobot = 35; lvl = f"💪 EKSTREM ({rsi:.1f})"
        elif rsi > 75: bobot = 30; lvl = f"✊ TINGGI ({rsi:.1f})"
        else:          bobot = 25; lvl = f"👌 OB ({rsi:.1f})"
        skor_short += bobot; confluence_short += 1
        rincian_short[f'📉 RSI'] = f"🔴 OVERBOUGHT TURUN {lvl} (+{bobot}) | {rsi_cn:.1f}→{rsi_c1:.1f}→{rsi:.1f}"
    else:
        rincian_short[f'📉 RSI'] = f"⚪ Tidak aktif | {rsi_cn:.1f}→{rsi_c1:.1f}→{rsi:.1f}"

    # =========================================================================
    # CONFLUENCE BONUS (maks 3 indikator per arah)
    # 1 agree → +0, 2 agree → +15, 3 agree → +30
    # =========================================================================
    TABEL_BONUS  = {1: 0, 2: 15, 3: 30}
    bonus_long   = TABEL_BONUS.get(confluence_long,  0)
    bonus_short  = TABEL_BONUS.get(confluence_short, 0)

    if confluence_long > confluence_short and bonus_long > 0:
        skor_long += bonus_long
        bonus_info = f"🟢 {confluence_long}/3 LONG agree → +{bonus_long}"
    elif confluence_short > confluence_long and bonus_short > 0:
        skor_short += bonus_short
        bonus_info = f"🔴 {confluence_short}/3 SHORT agree → +{bonus_short}"
    else:
        bonus_info = f"⚪ Terbagi ({confluence_long}L/{confluence_short}S) → +0"

    # =========================================================================
    # SUSUN OUTPUT
    # =========================================================================
    rincian = {}
    rincian['─── BULLISH ───'] = ''
    for k, v in rincian_long.items():
        rincian[k] = v
    rincian['─── BEARISH ───'] = ''
    for k, v in rincian_short.items():
        rincian[k] = v
    rincian['🔥 Confluence'] = bonus_info
    rincian['📊 Skor'] = (
        f"LONG {skor_long} (base {skor_long - bonus_long if confluence_long > confluence_short else skor_long}) | "
        f"SHORT {skor_short} (base {skor_short - bonus_short if confluence_short > confluence_long else skor_short})"
    )

    return min(skor_long, 130), min(skor_short, 130), rincian

# ==========================================
# HELPER SALDO
# ==========================================
def ambil_saldo_usdt():
    try:
        data = parse_resp(client_trade.account())
        for asset in data.get('assets', []):
            if asset.get('asset') == 'USDT':
                return float(asset.get('walletBalance', 0))
    except Exception:
        pass
    return 0.0

# ==========================================
# 7. LOOP UTAMA
# ==========================================
setup_leverage()
model_ai_global = latih_ai_diawal()
print(f"\n🚀 Bot FUTURES v4.6 — Lean 3-Indicator Gate 1 (5m)")
print(f"   EMA      : {EMA_PENDEK}/{EMA_PANJANG} | MACD: {MACD_FAST},{MACD_SLOW},{MACD_SIGNAL} | Stoch: {STOCH_K},{STOCH_D},{STOCH_S}")
print(f"   RSI cek  : {RSI_KONSISTENSI_CANDLE} candle | Loop: {LOOP_DELAY}s | SL ATR: {SL_AWAL_ATR_MULTIPLIER}x | Trail ATR: {TRAILING_ATR_MULTIPLIER}x")
print(f"   Gate 3 baru: Fear&Greed + Funding Rate + Long/Short Ratio")
print(f"   (Menggantikan RSS+FinBERT yang lag berjam-jam)")
print(f"   Pasar    : {SIMBOL_DISPLAY} Perpetual Futures (Testnet)")
print(f"   Leverage : {LEVERAGE}x  |  Modal/Trade: ${MARGIN_USDT}")
print(f"   Timeframe: {TIMEFRAME}\n")

# ── State candle tracker ─────────────────────────────────────────────────────
# Dipakai loop cerdas: analisis berat hanya jalan saat candle baru terbentuk
_last_candle_time = 0
_df_cache         = None      # cache df terakhir untuk mode pelacakan
_balance_cache    = 0.0
_balance_ts       = 0

def ambil_harga_cepat():
    """Fetch harga terkini — ringan, hanya 1 request kecil."""
    try:
        resp = client_data.ticker_price(symbol=SIMBOL)
        return float(parse_resp(resp).get('price', 0))
    except Exception:
        return 0.0

def ambil_saldo_cached():
    """Fetch saldo USDT dengan cache 30 detik — tidak perlu fetch setiap 2 detik."""
    global _balance_cache, _balance_ts
    if time.time() - _balance_ts > 30:
        _balance_cache = ambil_saldo_usdt()
        _balance_ts    = time.time()
    return _balance_cache

def candle_baru_terbentuk():
    """
    Cek apakah candle baru sudah terbentuk berdasarkan server time Binance.
    Cara: floor server time ke interval candle → bandingkan dengan candle terakhir.
    Contoh 1m: floor ke menit → tiap menit angka berubah = candle baru.
    """
    global _last_candle_time
    try:
        server_time  = parse_resp(client_data.time()).get('serverTime', 0)
        # Interval dalam milidetik
        interval_map = {
            '1m': 60_000, '3m': 180_000, '5m': 300_000,
            '15m': 900_000, '30m': 1_800_000, '1h': 3_600_000
        }
        interval_ms  = interval_map.get(TIMEFRAME, 60_000)
        candle_time  = (server_time // interval_ms) * interval_ms

        if candle_time > _last_candle_time:
            _last_candle_time = candle_time
            return True
        return False
    except Exception:
        return False

def loop_pelacakan(harga_skrg):
    """
    Loop CEPAT — dijalankan setiap LOOP_DELAY detik saat ada posisi terbuka.
    Hanya fetch harga (1 request ringan) + update trailing SL.
    Tidak fetch klines, tidak hitung indikator.
    """
    global sedang_ada_posisi, id_order_sl_server, arah_posisi

    # Cek apakah SL sudah kena
    if id_order_sl_server:
        try:
            resp         = client_trade.query_order(symbol=SIMBOL, orderId=id_order_sl_server)
            order_info   = parse_resp(resp)
            if order_info.get('status', '') in ('FILLED', 'CANCELED', 'EXPIRED'):
                harga_exit = float(order_info.get('avgPrice') or harga_sl_server_saat_ini)
                pnl_akhir  = hitung_pnl(harga_exit, harga_entry_sekarang, lot_sekarang, arah_posisi)
                pnl_persen = (pnl_akhir / MARGIN_USDT) * 100
                simpan_ke_db({
                    'Waktu'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Simbol'     : SIMBOL_DISPLAY, 'Side': arah_posisi,
                    'Entry'      : harga_entry_sekarang, 'Exit': harga_exit,
                    'Lot'        : lot_sekarang, 'Leverage': LEVERAGE,
                    'Margin_USDT': MARGIN_USDT,
                    'PnL_USD'    : round(pnl_akhir, 2),
                    'PnL_Persen' : round(pnl_persen, 2),
                    'Status'     : 'CLOSED_BY_TRAILING_SL'
                })
                print(f"\n🔔 POSISI CLOSED! PnL: ${pnl_akhir:.2f} ({pnl_persen:+.1f}%)")
                sedang_ada_posisi = False
                id_order_sl_server = None
                arah_posisi = None
                return
        except Exception:
            pass

    # Update trailing SL
    perlu_update = False
    calon_sl     = 0
    if arah_posisi == 'LONG' and harga_skrg > harga_ekstrem_posisi:
        globals()['harga_ekstrem_posisi'] = harga_skrg
        calon_sl = round(harga_skrg - TRAILING_ATR_MULTIPLIER * atr_posisi, 2)
        if calon_sl > harga_sl_server_saat_ini: perlu_update = True
    elif arah_posisi == 'SHORT' and harga_skrg < harga_ekstrem_posisi:
        globals()['harga_ekstrem_posisi'] = harga_skrg
        calon_sl = round(harga_skrg + TRAILING_ATR_MULTIPLIER * atr_posisi, 2)
        if calon_sl < harga_sl_server_saat_ini: perlu_update = True

    if perlu_update:
        print(f"🔄 Geser SL → ${calon_sl}")
        try:
            if id_order_sl_server:
                client_trade.cancel_order(symbol=SIMBOL, orderId=id_order_sl_server)
            id_baru = pasang_sl_server(calon_sl, arah_posisi)
            if id_baru:
                globals()['id_order_sl_server']       = id_baru
                globals()['harga_sl_server_saat_ini'] = calon_sl
        except Exception as e:
            print(f"⚠️ Gagal update SL: {e}")

    # Tampilkan status ringkas
    profit = hitung_pnl(harga_skrg, harga_entry_sekarang, lot_sekarang, arah_posisi)
    pct    = (profit / MARGIN_USDT) * 100
    tanda  = "+" if profit >= 0 else ""
    emoji  = "📈 LONG" if arah_posisi == 'LONG' else "📉 SHORT"
    print(f"🛡️ [{emoji}] ${harga_skrg:.2f} | PnL: {tanda}${profit:.2f} ({tanda}{pct:.1f}%) | SL: ${harga_sl_server_saat_ini:.2f}")

def loop_radar(df, harga_skrg):
    """
    Loop LAMBAT — dijalankan hanya saat candle baru terbentuk.
    Menjalankan semua Gate 1, 2, 3 dan eksekusi order kalau sinyal valid.
    """
    global sedang_ada_posisi, arah_posisi, harga_entry_sekarang
    global lot_sekarang, id_order_sl_server, harga_sl_server_saat_ini
    global harga_ekstrem_posisi, atr_posisi

    idx_bar      = len(df) - 2
    bar_terakhir = df.iloc[idx_bar]
    skor_long, skor_short, rincian = hitung_skor_teknikal(bar_terakhir, df, idx_bar)

    data_ai      = df.iloc[idx_bar:idx_bar+1][fitur_ai]
    probabilitas = model_ai_global.predict_proba(data_ai)[0]
    prob_naik    = probabilitas[1] * 100
    prob_turun   = probabilitas[0] * 100

    total_trade, win_trade, win_rate = hitung_win_rate()
    current_usdt = ambil_saldo_cached()

    print(f"\n{'='*65}")
    print(f"📡 RADAR 3-GATE v4.5 ({TIMEFRAME}) | Candle baru | Live: ${harga_skrg:.2f}")
    print(f"💰 Balance: ${current_usdt:.2f}  |  Leverage: {LEVERAGE}x")
    print(f"🏆 Win Rate: {win_rate:.1f}% ({win_trade}/{total_trade})")
    print(f"{'='*65}")
    print(f"1️⃣  GATE 1: TEKNIKAL")
    for k, v in rincian.items():
        if v:  # skip separator kosong
            print(f"   {k.ljust(28)} : {v}")
        else:
            print(f"   {k}")
    print(f"\n   SKOR LONG: {skor_long}  |  SKOR SHORT: {skor_short}  (≥{AMBANG_TEKNIKAL})")

    print(f"\n2️⃣  GATE 2: AI ML")
    print(f"   Prob NAIK: {prob_naik:.1f}%  |  Prob TURUN: {prob_turun:.1f}%  (≥{AMBANG_AI}%)")
    print(f"{'-'*65}")

    sinyal_long  = skor_long  >= AMBANG_TEKNIKAL and prob_naik  >= AMBANG_AI
    sinyal_short = skor_short >= AMBANG_TEKNIKAL and prob_turun >= AMBANG_AI

    if sinyal_long or sinyal_short:
        arah_sinyal = 'LONG' if sinyal_long else 'SHORT'
        if sinyal_long and sinyal_short:
            arah_sinyal = 'LONG' if skor_long >= skor_short else 'SHORT'

        print(f"3️⃣  GATE 3: SENTIMEN REALTIME")
        print(f"   📡 Sinyal {arah_sinyal} terdeteksi → cek sentimen...")
        cocok, skor_sentimen, rincian_sentimen = analisis_sentimen_realtime(arah_sinyal)
        for teks in rincian_sentimen:
            print(teks)
        print(f"{'-'*65}")

        if not cocok:
            print(f"⛔ BATAL {arah_sinyal}: Sentimen berlawanan (skor {skor_sentimen:+d})")
        else:
            label = "⚠️  NETRAL" if skor_sentimen == 0 else "🔥 ULTIMATE CONFLUENCE!"
            print(f"{label} → Eksekusi {arah_sinyal}...")
            lot = hitung_lot(harga_skrg)
            try:
                sisi_entry    = 'BUY'  if arah_sinyal == 'LONG' else 'SELL'
                position_side = 'LONG' if arah_sinyal == 'LONG' else 'SHORT'
                resp_entry    = client_trade.new_order(
                    symbol=SIMBOL, side=sisi_entry,
                    type='MARKET', quantity=lot,
                    positionSide=position_side
                )
                data_entry           = parse_resp(resp_entry)
                harga_entry_sekarang = float(data_entry.get('avgPrice') or harga_skrg) or harga_skrg
                lot_sekarang         = lot
                atr_nilai            = float(bar_terakhir['ATR'])
                atr_posisi           = atr_nilai
                harga_ekstrem_posisi = harga_entry_sekarang
                harga_sl_awal        = round(
                    harga_entry_sekarang - SL_AWAL_ATR_MULTIPLIER * atr_nilai, 2
                ) if arah_sinyal == 'LONG' else round(
                    harga_entry_sekarang + SL_AWAL_ATR_MULTIPLIER * atr_nilai, 2
                )
                id_sl = pasang_sl_server(harga_sl_awal, arah_sinyal)
                if id_sl:
                    id_order_sl_server       = id_sl
                    harga_sl_server_saat_ini = harga_sl_awal
                    arah_posisi              = arah_sinyal
                    sedang_ada_posisi        = True
                    print(f"✅ {arah_sinyal} Entry: ${harga_entry_sekarang:.2f} | Lot: {lot} | SL: ${harga_sl_awal:.2f}")
                else:
                    print("⚠️ Gagal pasang SL! Tutup posisi otomatis.")
                    client_trade.new_order(
                        symbol=SIMBOL,
                        side='SELL' if arah_sinyal == 'LONG' else 'BUY',
                        type='MARKET', quantity=lot,
                        positionSide=position_side, reduceOnly='true'
                    )
            except ClientError as e:
                print(f"❌ Gagal Eksekusi (ClientError): {e}")
            except Exception as e:
                print(f"❌ Gagal Eksekusi: {e}")
    else:
        if skor_long < AMBANG_TEKNIKAL and skor_short < AMBANG_TEKNIKAL:
            print("💤 Gate 1 Gagal — tunggu formasi teknikal...")
        else:
            print("💤 Gate 2 Gagal — AI belum yakin...")
    print("="*65)

# ==========================================
# 8. MAIN LOOP — SMART DUAL LOOP
# ==========================================
# Arsitektur:
#   Loop CEPAT (setiap LOOP_DELAY detik):
#     → Hanya fetch harga terkini (1 request ringan)
#     → Kalau ada posisi: update trailing SL, cek SL kena
#     → Kalau tidak ada posisi: cek apakah candle baru terbentuk
#
#   Loop LAMBAT (hanya saat candle baru):
#     → Fetch klines 500 candle (1 request berat)
#     → Hitung semua indikator (CPU intensive)
#     → Jalankan Gate 1, 2, 3
#     → Eksekusi order kalau sinyal valid
#
# Efisiensi: di 1m, klines berat hanya di-fetch 1x per menit
# vs sebelumnya 30x per menit (setiap 2 detik)

while True:
    try:
        # ── Fetch harga terkini — ringan, selalu jalan ──────────────────────
        harga_skrg = ambil_harga_cepat()
        if harga_skrg == 0:
            time.sleep(LOOP_DELAY)
            continue

        # ── MODE PELACAKAN: ada posisi terbuka ──────────────────────────────
        if sedang_ada_posisi:
            loop_pelacakan(harga_skrg)
            time.sleep(LOOP_DELAY)
            continue

        # ── MODE RADAR: cari sinyal, hanya saat candle baru ─────────────────
        if candle_baru_terbentuk():
            df = ambil_data_live()
            if df is not None:
                _df_cache  = df
                harga_skrg = df.iloc[-1]['close']  # pakai close candle, bukan ticker
                loop_radar(df, harga_skrg)
        else:
            # Candle belum baru — tampilkan status tunggu ringkas
            waktu_skrg  = datetime.now().strftime('%H:%M:%S')
            print(f"⏳ [{waktu_skrg}] Menunggu candle baru... | Live: ${harga_skrg:.2f}", end='\r')

        time.sleep(LOOP_DELAY)

    except Exception as e:
        print(f"\n⚠️ Error Utama: {e}")
        time.sleep(5)