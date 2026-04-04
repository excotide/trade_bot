# 🤖 AlgoTrading Bot — ML + Technical Analysis

> **Binance Futures trading bot** dengan pendekatan kuantitatif: kombinasi indikator teknikal institusional, ensemble machine learning, manajemen risiko berbasis ATR, dan pencatatan riwayat transaksi otomatis ke Supabase lengkap dengan kalkulasi winrate.

---

## 📋 Daftar Isi

- [Gambaran Umum](#-gambaran-umum)
- [Arsitektur Sistem](#️-arsitektur-sistem)
- [Cara Bot Bekerja](#-cara-bot-bekerja)
- [Fitur Detail](#-fitur-detail)
- [Struktur Komponen](#-struktur-komponen)
- [Instalasi & Setup](#-instalasi--setup)
- [Konfigurasi](#️-konfigurasi)
- [Database Schema](#️-database-schema-supabase)
- [Statistik Winrate](#-statistik-winrate)
- [Risiko & Disclaimer](#️-risiko--disclaimer)

---

## 🔍 Gambaran Umum

Bot ini dirancang berdasarkan prinsip-prinsip dari literatur **quantitative trading** dan **algorithmic finance**:

| Komponen | Detail |
|---|---|
| **Exchange** | Binance USDT-Margined Futures (`UMFutures`) |
| **SDK** | `binance-futures-connector` (official Binance Python SDK) |
| **Timeframe Sinyal** | 15 menit (entry) + 1 jam (konfirmasi tren) |
| **Timeframe ML** | 1 jam (optimal untuk pola prediktif, minim noise) |
| **Model ML** | Random Forest + Gradient Boosting (ensemble) |
| **Database** | Supabase (PostgreSQL) — history transaksi + winrate |
| **Manajemen Risiko** | ATR-based dynamic stop-loss & take-profit |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRADING BOT                              │
│                                                                 │
│  ┌──────────────┐    ┌───────────────────────────────────────┐  │
│  │   BINANCE    │    │           DATA PIPELINE               │  │
│  │   FUTURES    │───▶│  fetch_ohlcv() ──▶ FeatureEngineer   │  │
│  │  (UMFutures) │    │  funding_rate()    ├─ VWAP            │  │
│  │              │    │  open_interest()   ├─ RSI / MACD / ATR│  │
│  └──────────────┘    │  balance()         ├─ Price Action    │  │
│         ▲            │                    └─ Derivative Data  │  │
│         │            └──────────────┬────────────────────────┘  │
│         │                           │                            │
│         │            ┌──────────────▼────────────────────────┐  │
│         │            │          ML MODEL (Ensemble)           │  │
│         │            │  RandomForest + GradientBoosting       │  │
│         │            │  → predict() → {signal, confidence}    │  │
│         │            └──────────────┬────────────────────────┘  │
│         │                           │                            │
│         │            ┌──────────────▼────────────────────────┐  │
│         │            │         SIGNAL ENGINE                  │  │
│         │            │  Rule-based + ML Gate                  │  │
│         │            │  VWAP bias + RSI/MACD + Trend 1H       │  │
│         │            │  → action: LONG / SHORT / HOLD         │  │
│         │            └──────────────┬────────────────────────┘  │
│         │                           │                            │
│         │            ┌──────────────▼────────────────────────┐  │
│         │            │        ORDER MANAGER                   │  │
│         │            │  ATR-based sizing                      │  │
│         └────────────│  Market order + SL + TP                │  │
│                      └──────────────┬────────────────────────┘  │
│                                     │                            │
│                      ┌──────────────▼────────────────────────┐  │
│                      │       SUPABASE DATABASE                │  │
│                      │  insert_trade() → status: OPEN         │  │
│                      │  close_trade()  → status: WIN/LOSS     │  │
│                      │  get_stats()    → winrate + PnL        │  │
│                      └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Cara Bot Bekerja

Bot berjalan dalam **siklus otomatis setiap 15 menit**. Berikut alur lengkap satu siklus:

### Fase 1 — Inisialisasi (hanya saat pertama kali dijalankan)

```
python trading_bot.py
        │
        ▼
[1] Koneksi ke Binance Futures (ping test)
        │
        ▼
[2] Koneksi ke Supabase
        │
        ▼
[3] Cek apakah model ML sudah tersimpan (model_trading.pkl)
        │
    ┌───┴────────────────────────────────────────────┐
    │ BELUM ADA                    SUDAH ADA          │
    ▼                              ▼                  │
[4a] Fetch data historis       [4b] Load model        │
     3 tahun × 1H (±26.280          dari disk         │
     baris candle)                  langsung siap      │
        │                                             │
        ▼                                             │
[5] Feature Engineering                               │
    (VWAP, RSI, MACD, ATR,                           │
     Price Action, Derivatives)                       │
        │                                             │
        ▼                                             │
[6] Training ML (RF + GB)                             │
    dengan TimeSeriesSplit (5-fold)                   │
        │                                             │
        ▼                                             │
[7] Simpan model ke disk ─────────────────────────────┘
        │
        ▼
[8] Tampilkan statistik awal dari Supabase
```

### Fase 2 — Siklus Trading (tiap 15 menit)

```
⏰ Scheduler trigger (setiap 15 menit)
        │
        ▼
[1] POSITION MONITOR
    └── Cek posisi OPEN di Supabase
    └── Bandingkan dengan posisi aktif di Binance
    └── Kalau sudah tidak aktif → ambil realized PnL
    └── Update Supabase: exit_price, pnl, status (WIN/LOSS)
        │
        ▼
[2] FETCH DATA REAL-TIME
    ├── OHLCV 15m (300 candle terakhir) ── untuk sinyal entry
    ├── OHLCV 1H  (100 candle terakhir) ── untuk konfirmasi tren
    ├── Funding Rate ──────────────────── derivatif, bias pasar
    └── Open Interest ─────────────────── derivatif, positioning
        │
        ▼
[3] FEATURE ENGINEERING
    ├── VWAP + jarak relatif ke VWAP
    ├── RSI (14), MACD (12/26/9), ATR (14)
    ├── Log return 1/5/20 periode
    ├── Morfologi candle (body, upper/lower wick)
    ├── Volatilitas rolling 20 candle
    ├── Price position dalam range 20 candle
    ├── Volume Z-score
    └── Funding rate + Open interest (sebagai scalar)
        │
        ▼
[4] ML PREDICTION
    ├── RandomForest.predict_proba(X_last_candle)
    ├── GradientBoosting.predict_proba(X_last_candle)
    ├── Ensemble: rata-rata kedua probabilitas
    └── Output: {signal: 1/-1/0, confidence: 0.0–1.0}
        │
        ▼
[5] SIGNAL ENGINE (Rule-Based Gate)
    ├── Cek [1]: VWAP bias (harga vs VWAP ± 0.1% band)
    ├── Cek [2]: RSI ekstrem ATAU MACD crossover
    ├── Cek [3]: Tren 1H searah (harga vs VWAP di 1H)
    └── Cek [4]: ML confidence ≥ 60%
            │
        Semua 4 ✅     Ada yang ❌
            │               │
            ▼               ▼
         LONG/SHORT        HOLD
            │
            ▼
[6] ORDER EXECUTION
    ├── Cek posisi terbuka (max 3)
    ├── Hitung balance USDT
    ├── Kalkulasi size (ATR-based, risiko 1% per trade)
    ├── Hitung SL = entry ± (1.5 × ATR)
    ├── Hitung TP = entry ± (2.5 × ATR)
    ├── Place MARKET order → Binance
    ├── Place STOP_MARKET (SL) → Binance
    ├── Place TAKE_PROFIT_MARKET (TP) → Binance
    └── Insert record ke Supabase (status: OPEN)
        │
        ▼
[7] Setiap 10 siklus → tampilkan statistik winrate
```

---

## 🧩 Fitur Detail

### 1. 📡 Binance Futures Connector (Official SDK)

Bot menggunakan `binance-futures-connector` — library resmi dari Binance, bukan wrapper pihak ketiga.

| Method | Fungsi |
|---|---|
| `client.ping()` | Tes koneksi saat startup |
| `client.klines()` | Ambil data OHLCV (real-time & bulk historis) |
| `client.funding_rate()` | Ambil funding rate terkini |
| `client.open_interest()` | Ambil total open interest |
| `client.balance()` | Cek available USDT |
| `client.get_position_risk()` | Cek posisi yang aktif |
| `client.change_leverage()` | Set leverage sebelum order |
| `client.new_order()` | Kirim order (MARKET / STOP_MARKET / TAKE_PROFIT_MARKET) |
| `client.get_income_history()` | Ambil realized PnL dari history |

**Dukungan Testnet:** Cukup ganti `BINANCE_BASE_URL` ke URL testnet — tidak perlu ubah kode lain.

---

### 2. 📊 Multi-Timeframe Analysis

Bot bekerja di **3 timeframe sekaligus** dengan peran berbeda:

```
15m ─── Sinyal Entry
         Deteksi konfluensi VWAP + RSI/MACD untuk menentukan
         kapan tepatnya bot masuk posisi.

1H  ─── Konfirmasi Bias Tren
         Pastikan arah trade di 15m searah dengan tren yang
         lebih besar di 1H. Ini filter noise yang kuat.

1H  ─── Training ML
         Model ML dilatih di 1H karena di bawah 1H,
         algoritma cenderung belajar noise acak.
```

---

### 3. 🧮 Feature Engineering (17 Fitur)

Total **17 fitur** yang dimasukkan ke model ML, dibagi 4 kelompok:

#### Kelompok A — VWAP (1 fitur)
| Fitur | Deskripsi |
|---|---|
| `vwap_dist_pct` | Jarak relatif harga ke VWAP dalam persen. Positif = bullish zone. |

#### Kelompok B — Indikator Teknikal (5 fitur)
| Fitur | Deskripsi |
|---|---|
| `rsi` | RSI 14 periode — deteksi kelelahan momentum |
| `macd` | Nilai MACD line (12/26) |
| `macd_sig` | Signal line MACD (9) |
| `macd_hist` | Histogram MACD — kecepatan divergensi tren |
| `atr_pct` | ATR 14 sebagai persentase harga — ukuran volatilitas |

#### Kelompok C — Price Action (6 fitur, >60% feature importance)
| Fitur | Deskripsi |
|---|---|
| `log_ret_1` | Return logaritmik 1 candle — momentum jangka pendek |
| `log_ret_5` | Return logaritmik 5 candle — momentum medium |
| `log_ret_20` | Return logaritmik 20 candle — momentum panjang |
| `body_size` | Ukuran body candle relatif — kekuatan candle |
| `upper_wick` | Panjang upper wick — penolakan harga di atas |
| `lower_wick` | Panjang lower wick — penolakan harga di bawah |
| `volatility_20` | Standar deviasi rolling 20 candle |
| `price_position` | Posisi harga dalam range 20 candle (0 = low, 1 = high) |
| `volume_zscore` | Volume relatif terhadap rata-rata — deteksi anomali volume |

#### Kelompok D — Derivatif Futures (2 fitur)
| Fitur | Deskripsi |
|---|---|
| `funding_rate` | Funding rate terkini — bias positioning pasar |
| `open_interest` | Total kontrak aktif — ukuran eksposur pasar |

---

### 4. 🧠 Machine Learning Ensemble

Model menggunakan **dua algoritma yang di-ensemble** (rata-rata probabilitas):

#### Random Forest
```
n_estimators     = 300 pohon
max_depth        = 8 level
min_samples_leaf = 50 sampel minimum per leaf (anti-overfit)
class_weight     = balanced (handle imbalanced label)
```

#### Gradient Boosting
```
n_estimators  = 200 iterasi
learning_rate = 0.05 (konservatif, generalisasi lebih baik)
max_depth     = 4 level
subsample     = 0.8 (regularisasi — hanya 80% data per iterasi)
```

#### Validasi Anti-Cheating
- Pakai **`TimeSeriesSplit`** (bukan `KFold` biasa)
- 5-fold cross-validation berbasis waktu
- Mencegah **look-ahead bias** — data masa depan tidak pernah masuk ke set training
- Evaluasi pakai **F1-macro** untuk handle class imbalance (LONG/HOLD/SHORT tidak sama banyak)

#### Output Label Training
```
+1  = LONG  → harga naik >0.3% dalam 3 candle ke depan
-1  = SHORT → harga turun >0.3% dalam 3 candle ke depan
 0  = HOLD  → pergerakan terlalu kecil, tidak layak di-trade
```

---

### 5. 🎯 Signal Engine — 4-Layer Filter

Sinyal entry hanya dieksekusi kalau **semua 4 kondisi terpenuhi sekaligus**. Ini yang membuat bot selektif dan menghasilkan sinyal kualitas tinggi.

```
Layer 1 — VWAP Bias
  LONG  : close > VWAP × 1.001  (harga 0.1% di atas VWAP)
  SHORT : close < VWAP × 0.999  (harga 0.1% di bawah VWAP)

Layer 2 — RSI atau MACD (salah satu cukup)
  LONG  : RSI < 35 (oversold)  ATAU  MACD crossover ke atas
  SHORT : RSI > 65 (overbought) ATAU  MACD crossover ke bawah

Layer 3 — Konfirmasi Tren 1H
  LONG  : close_1h > VWAP_1h  (tren 1H masih bullish)
  SHORT : close_1h < VWAP_1h  (tren 1H masih bearish)

Layer 4 — ML Confidence Gate
  LONG  : ML prediksi signal=+1 dengan confidence ≥ 60%
  SHORT : ML prediksi signal=-1 dengan confidence ≥ 60%
```

Kalau ada 1 layer saja yang tidak terpenuhi → **HOLD**, tidak ada order.

---

### 6. 💰 Manajemen Risiko (ATR-Based)

#### Position Sizing
```
risiko_dollar = balance × 1%
sl_distance   = 1.5 × ATR
size (asset)  = risiko_dollar ÷ sl_distance ÷ harga_entry
```
Contoh: balance $1000, ATR = $500, harga BTC = $65.000
```
risiko_dollar = $1000 × 0.01 = $10
sl_distance   = 1.5 × $500  = $750
size          = $10 ÷ $750 ÷ $65.000 = 0.000205 BTC
```
Artinya kalau kena SL, loss maksimal hanya **$10 (1% modal)**.

#### Stop-Loss & Take-Profit Adaptif
```
SL = entry ± (1.5 × ATR)   → menyesuaikan volatilitas saat itu
TP = entry ± (2.5 × ATR)   → Risk:Reward ratio = 1:1.67
```
Dengan R:R 1:1.67, bot bisa **profit meski winrate hanya ~38%**.

#### Batas Posisi
- Maksimal **3 posisi terbuka** bersamaan
- Leverage dibatasi **maksimal 5×**
- Semua SL/TP dipasang sebagai **`closePosition: true`** — auto-close tanpa sisa kontrak

---

### 7. 📦 Data Fetching Bulk (untuk ML Training)

Bot secara otomatis mengambil **data historis 3 tahun** untuk training:
- Binance izinkan max 1.500 candle per request
- Bot melakukan **loop iteratif** dari tanggal awal sampai sekarang
- Setiap request diberi delay `0.3 detik` untuk menghindari rate limit
- Minimum **10.000 baris** diperlukan — sesuai literatur kuantitatif

Untuk timeframe 1H, 3 tahun = ±26.280 candle (jauh di atas minimum).

---

### 8. 🗄️ Supabase — History Transaksi

Setiap order yang dieksekusi **langsung dicatat** ke Supabase secara real-time.

#### Alur Pencatatan
```
Order berhasil dibuat di Binance
        │
        ▼
insert_trade() → Supabase
  status    = "OPEN"
  entry_price, sl_price, tp_price
  size, ml_confidence, atr
  order_id (untuk tracking)
        │
        ▼ (saat posisi close — kena SL atau TP)
        │
PositionMonitor.sync() (tiap siklus)
        │
        ▼
Binance income_history → ambil realized PnL
        │
        ▼
close_trade() → Supabase
  exit_price = estimasi dari PnL
  pnl        = realized PnL dalam USDT
  pnl_pct    = PnL dalam persentase
  status     = "WIN" atau "LOSS"
```

#### Field yang Tersimpan
| Field | Tipe | Keterangan |
|---|---|---|
| `id` | UUID | Primary key auto-generate |
| `created_at` | TIMESTAMPTZ | Waktu trade dibuat |
| `symbol` | TEXT | Pair trading (contoh: BTCUSDT) |
| `side` | TEXT | LONG atau SHORT |
| `entry_price` | FLOAT | Harga masuk |
| `exit_price` | FLOAT | Harga keluar (NULL saat OPEN) |
| `sl_price` | FLOAT | Level stop-loss |
| `tp_price` | FLOAT | Level take-profit |
| `size` | FLOAT | Ukuran posisi dalam asset |
| `pnl` | FLOAT | Realized PnL dalam USDT |
| `pnl_pct` | FLOAT | PnL dalam desimal |
| `status` | TEXT | OPEN / WIN / LOSS |
| `ml_confidence` | FLOAT | Probabilitas ML saat entry |
| `atr` | FLOAT | Nilai ATR saat entry |
| `order_id` | TEXT | Binance order ID |
| `signal_ts` | TIMESTAMPTZ | Timestamp candle sinyal |

---

### 9. 📈 Winrate & Statistik Performa

Ditampilkan otomatis setiap **10 siklus trading** (±2.5 jam):

```
════════════════════════════════════════════════════════
  📊  STATISTIK PERFORMA BOT
────────────────────────────────────────────────────────
  Total Trade    :    247 trade
  WIN / LOSS     : 157 WIN  /   90 LOSS
  Winrate        :   63.56 %
────────────────────────────────────────────────────────
  Winrate LONG   :   67.80 %
  Winrate SHORT  :   58.20 %
────────────────────────────────────────────────────────
  Total PnL      :  +1.842,3400 USDT
  Avg PnL/Trade  :      +7.4585 USDT
  Profit Factor  :       2.143
  Best Trade     :    +124.5600 USDT
  Worst Trade    :     -38.2100 USDT
  Streak Kini    :  4× WIN
════════════════════════════════════════════════════════
```

**Profit Factor** = total gross profit ÷ total gross loss. Nilai di atas **1.5** dianggap sistem yang baik.

---

## 🧱 Struktur Komponen

```
trading_bot.py
│
├── Config                  # Semua parameter di satu tempat
├── create_binance_client() # Inisialisasi UMFutures + ping
│
├── SupabaseManager         # Operasi database
│   ├── insert_trade()
│   ├── close_trade()
│   ├── get_stats()         ◄─ winrate + profit factor + streak
│   └── get_open_trades()
│
├── DataFetcher             # Ambil data dari Binance
│   ├── fetch_ohlcv()       ◄─ real-time (tiap siklus)
│   ├── fetch_ohlcv_bulk()  ◄─ historis bulk (untuk training ML)
│   ├── fetch_funding_rate()
│   ├── fetch_open_interest()
│   ├── fetch_balance()
│   └── fetch_open_positions()
│
├── FeatureEngineer         # Pipeline 17 fitur
│   ├── add_vwap()
│   ├── add_rsi_macd_atr()
│   ├── add_price_action()
│   ├── add_derivative_metrics()
│   ├── add_labels()        ◄─ hanya saat training
│   └── build_features()    ◄─ pipeline utama
│
├── MLModel                 # Ensemble RF + GB
│   ├── train()             ◄─ TimeSeriesSplit 5-fold
│   ├── predict()           ◄─ ensemble inference
│   ├── save() / load()     ◄─ persist ke disk (.pkl)
│   └── _build_rf/gb()
│
├── SignalEngine            # 4-layer decision gate
│   └── compute()
│
├── OrderManager            # Eksekusi order + catat Supabase
│   ├── _calc_size()        ◄─ ATR-based position sizing
│   └── place_order()       ◄─ market + SL + TP + insert DB
│
├── PositionMonitor         # Sinkronisasi close ke Supabase
│   ├── sync()
│   └── _get_realized_pnl()
│
└── TradingBot              # Orkestrator utama
    ├── initialize()        ◄─ load/train model + tampilkan stats
    ├── run_once()          ◄─ satu siklus lengkap
    └── start()             ◄─ scheduler tiap 15 menit
```

---

## ⚙️ Instalasi & Setup

### 1. Clone / Download

```bash
git clone <repo-url>
cd trading-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

```
binance-futures-connector>=4.0.0
supabase>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
ta>=0.11.0
scikit-learn>=1.4.0
joblib>=1.3.0
schedule>=1.2.0
```

### 3. Setup Supabase

Buka **Supabase > SQL Editor**, jalankan:

```sql
CREATE TABLE trades (
    id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT now(),
    symbol        TEXT        NOT NULL,
    side          TEXT        NOT NULL,
    entry_price   FLOAT       NOT NULL,
    exit_price    FLOAT,
    sl_price      FLOAT,
    tp_price      FLOAT,
    size          FLOAT,
    pnl           FLOAT,
    pnl_pct       FLOAT,
    status        TEXT        NOT NULL,
    ml_confidence FLOAT,
    atr           FLOAT,
    order_id      TEXT,
    signal_ts     TIMESTAMPTZ
);
```

### 4. Isi Konfigurasi

Edit bagian `Config` di `trading_bot.py`:

```python
class Config:
    BINANCE_API_KEY    = "api_key_binance_kamu"
    BINANCE_API_SECRET = "api_secret_binance_kamu"
    BINANCE_BASE_URL   = "https://testnet.binancefuture.com"  # ← testnet dulu!

    SUPABASE_URL       = "https://xxxx.supabase.co"
    SUPABASE_KEY       = "anon_key_supabase_kamu"
```

### 5. Jalankan

```bash
python trading_bot.py
```

Output awal yang diharapkan:

```
2025-01-15 08:00:01 | INFO | ✅ Binance Futures (TESTNET ⚠️) — terhubung
2025-01-15 08:00:01 | INFO | ✅ Supabase — terhubung
2025-01-15 08:00:01 | INFO | 🧠 Model belum ada. Mulai training...
2025-01-15 08:00:01 | INFO | 📥 Bulk fetch: BTCUSDT 1h — 3 tahun terakhir...
2025-01-15 08:03:45 | INFO | ✅ Bulk fetch selesai — 26.280 baris
2025-01-15 08:03:45 | INFO | 🔄 Training Random Forest...
2025-01-15 08:05:10 | INFO |    RF  F1-macro: 0.612 ± 0.031
2025-01-15 08:05:10 | INFO | 🔄 Training Gradient Boosting...
2025-01-15 08:06:30 | INFO |    GB  F1-macro: 0.598 ± 0.028
2025-01-15 08:06:30 | INFO | 💾 Model disimpan → model_trading.pkl
2025-01-15 08:06:30 | INFO | 📊 Belum ada trade yang selesai.
2025-01-15 08:06:30 | INFO | ⏰ Scheduler aktif — eksekusi tiap 15 menit
```

---

## 🗂️ Konfigurasi

Semua parameter ada di class `Config`. Tidak perlu ubah kode lain.

| Parameter | Default | Keterangan |
|---|---|---|
| `BINANCE_BASE_URL` | testnet URL | Ganti ke `https://fapi.binance.com` untuk live |
| `SYMBOL` | `BTCUSDT` | Pair yang di-trade |
| `TF_SIGNAL` | `15m` | Timeframe entry |
| `TF_TREND` | `1h` | Timeframe konfirmasi tren |
| `TF_ML` | `1h` | Timeframe training ML |
| `RISK_PER_TRADE` | `0.01` | 1% modal per trade |
| `ATR_SL_MULT` | `1.5` | Stop-loss = 1.5× ATR |
| `ATR_TP_MULT` | `2.5` | Take-profit = 2.5× ATR |
| `MAX_LEVERAGE` | `5` | Leverage maksimal |
| `MAX_OPEN_TRADES` | `3` | Maksimal posisi terbuka bersamaan |
| `MIN_TRAIN_ROWS` | `10_000` | Minimum data training ML |
| `TRAIN_YEARS` | `3` | Tahun data historis untuk training |
| `RSI_OVERSOLD` | `35` | Threshold RSI oversold |
| `RSI_OVERBOUGHT` | `65` | Threshold RSI overbought |
| `ML_CONFIDENCE` | `0.60` | Minimum confidence ML untuk entry |
| `VWAP_BAND_PCT` | `0.001` | Buffer zona netral VWAP (0.1%) |

---

## 🗃️ Database Schema Supabase

```sql
trades
├── id            UUID        — Primary key (auto UUID)
├── created_at    TIMESTAMPTZ — Waktu record dibuat
├── symbol        TEXT        — "BTCUSDT"
├── side          TEXT        — "LONG" | "SHORT"
├── entry_price   FLOAT       — Harga masuk posisi
├── exit_price    FLOAT       — Harga keluar (NULL = masih OPEN)
├── sl_price      FLOAT       — Level stop-loss
├── tp_price      FLOAT       — Level take-profit
├── size          FLOAT       — Ukuran posisi (dalam BTC)
├── pnl           FLOAT       — Realized PnL dalam USDT (NULL = OPEN)
├── pnl_pct       FLOAT       — PnL dalam desimal (0.05 = 5%)
├── status        TEXT        — "OPEN" | "WIN" | "LOSS"
├── ml_confidence FLOAT       — Probabilitas ML saat entry (0.0–1.0)
├── atr           FLOAT       — Nilai ATR saat entry
├── order_id      TEXT        — Binance order ID untuk tracking
└── signal_ts     TIMESTAMPTZ — Timestamp candle yang memicu sinyal
```

---

## 📊 Statistik Winrate

Metrik yang dihitung dan ditampilkan secara otomatis:

| Metrik | Penjelasan |
|---|---|
| **Winrate Keseluruhan** | % trade yang menghasilkan profit dari total trade |
| **Winrate LONG** | Winrate khusus posisi beli |
| **Winrate SHORT** | Winrate khusus posisi jual |
| **Total PnL** | Akumulasi profit/loss seluruh riwayat |
| **Avg PnL/Trade** | Rata-rata profit/loss per trade |
| **Profit Factor** | Gross profit ÷ gross loss (>1.5 = bagus, >2.0 = sangat bagus) |
| **Best Trade** | Trade dengan profit terbesar |
| **Worst Trade** | Trade dengan loss terbesar |
| **Current Streak** | Berapa kali WIN/LOSS berturut-turut terakhir |

---

## ⚠️ Risiko & Disclaimer

> **Trading futures melibatkan leverage dan risiko kehilangan modal yang sangat tinggi. Bot ini adalah alat bantu, bukan jaminan profit.**

Sebelum live trading:

1. **Wajib testnet minimal 2–4 minggu** — pantau log dan statistik, pastikan bot berjalan stabil
2. **Jangan naikkan leverage** di atas default — leverage tinggi memperbesar risiko likuidasi
3. **Monitor log harian** — file `trading_bot.log` menyimpan semua aktivitas
4. **Model perlu di-retrain berkala** — kondisi pasar berubah, model yang lama bisa degradasi
5. **Mulai dengan modal kecil** saat pertama live — validasi dulu di akun nyata dengan jumlah minimal

---

*Bot ini dibuat untuk tujuan edukasi dan penelitian algoritmik. Gunakan dengan bijak.*