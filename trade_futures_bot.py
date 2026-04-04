"""
╔══════════════════════════════════════════════════════════════════════╗
║              ALGO TRADING BOT — ML + TECHNICAL ANALYSIS             ║
║  Timeframe  : 15m / 1H / 4H                                         ║
║  Indikator  : VWAP · RSI · MACD · ATR                               ║
║  Model ML   : Random Forest + Gradient Boosting                      ║
║  Exchange   : Binance Futures (binance-futures-connector)            ║
║  Database   : Supabase (history transaksi + winrate otomatis)        ║
╚══════════════════════════════════════════════════════════════════════╝

Cara pakai:
    1. pip install binance-futures-connector supabase pandas-ta scikit-learn joblib schedule
    2. Isi semua kredensial di bagian Config
    3. Buat tabel di Supabase (lihat SQL di bawah ini)
    4. Jalankan: python trading_bot.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SQL SCHEMA — jalankan di Supabase > SQL Editor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CREATE TABLE trades (
        id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at    TIMESTAMPTZ DEFAULT now(),
        symbol        TEXT        NOT NULL,
        side          TEXT        NOT NULL,   -- 'LONG' | 'SHORT'
        entry_price   FLOAT       NOT NULL,
        exit_price    FLOAT,                  -- NULL selama masih OPEN
        sl_price      FLOAT,
        tp_price      FLOAT,
        size          FLOAT,
        pnl           FLOAT,                  -- NULL selama masih OPEN
        pnl_pct       FLOAT,                  -- PnL dalam desimal (0.05 = 5%)
        status        TEXT        NOT NULL,   -- 'OPEN' | 'WIN' | 'LOSS'
        ml_confidence FLOAT,
        atr           FLOAT,
        order_id      TEXT,
        signal_ts     TIMESTAMPTZ
    );
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
import os
import time
import logging
import warnings
import joblib
import schedule
import signal
import atexit
import sys
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# ── Binance official SDK ──────────────────────────────────
from binance.um_futures import UMFutures
from binance.error import ClientError

# ── Supabase ──────────────────────────────────────────────
from supabase import create_client, Client as SupabaseClient

# ── Technical Analysis ────────────────────────────────────
from ta.momentum  import RSIIndicator
from ta.trend     import MACD
from ta.volatility import AverageTrueRange

# ── Machine Learning ──────────────────────────────────────
from sklearn.ensemble     import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline     import Pipeline
from sklearn.metrics      import classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         1. KONFIGURASI                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

class Config:

    load_dotenv()   # Baca file .env otomatis dari direktori yang sama

    # ── Google Gemini API ─────────────────────────────────────────────
    GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")       # dari aistudio.google.com/apikey

    # ── Binance Futures ───────────────────────────────────────────────
    BINANCE_API_KEY    = os.getenv("BINANCE_DEMO_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")
    #   Testnet : "https://testnet.binancefuture.com"
    #   Live    : "https://fapi.binance.com"
    BINANCE_BASE_URL   = "https://testnet.binancefuture.com"

    # ── Supabase ──────────────────────────────────────────────────────
    SUPABASE_URL       = os.getenv("SUPABASE_URL")
    SUPABASE_KEY       = os.getenv("SUPABASE_KEY")
    SUPABASE_TABLE     = "trades"

    # ── Symbol & Timeframe ────────────────────────────────────────────
    SYMBOL             = "BTCUSDT"      # Format Binance (tanpa slash)
    TF_SIGNAL          = "15m"          # Sinyal entry utama
    TF_TREND           = "1h"           # Konfirmasi bias tren
    TF_ML              = "1h"           # Timeframe training ML (1H-4H optimal)

    # ── Manajemen Risiko ──────────────────────────────────────────────
    RISK_PER_TRADE     = 0.01           # Max 1% modal per trade
    ATR_SL_MULT        = 1.5           # SL = 1.5× ATR
    ATR_TP_MULT        = 2.5           # TP = 2.5× ATR  (R:R = 1:1.67)
    MAX_LEVERAGE       = 5
    MAX_OPEN_TRADES    = 3

    # ── ML Model ──────────────────────────────────────────────────────
    MODEL_PATH         = Path("model_trading.pkl")
    MIN_TRAIN_ROWS     = 10_000         # Minimum wajib sesuai literatur kuantitatif
    TRAIN_YEARS        = 3              # Ambil 3 tahun data historis

    # ── Threshold Sinyal ──────────────────────────────────────────────
    RSI_OVERSOLD       = 35
    RSI_OVERBOUGHT     = 65
    ML_CONFIDENCE      = 0.60           # ML harus ≥60% yakin sebelum entry
    VWAP_BAND_PCT      = 0.001          # 0.1% buffer zona netral VWAP


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              2. BINANCE FUTURES CLIENT                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def create_binance_client() -> UMFutures:
    """
    Inisialisasi koneksi ke Binance UM-Futures (USDT-Margined).
    Testnet dan live dibedain dari base_url di Config.
    """
    client = UMFutures(
        key      = Config.BINANCE_API_KEY,
        secret   = Config.BINANCE_API_SECRET,
        base_url = Config.BINANCE_BASE_URL,
    )
    client.ping()   # Cek koneksi
    mode = "TESTNET ⚠️" if "testnet" in Config.BINANCE_BASE_URL else "LIVE 🔴"
    log.info(f"✅ Binance Futures ({mode}) — terhubung")
    return client


# ╔══════════════════════════════════════════════════════════════════════╗
# ║           3. SUPABASE MANAGER — History & Winrate                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

class SupabaseManager:
    """
    Semua operasi database transaksi:
      - insert_trade  : catat order baru (status=OPEN)
      - close_trade   : update saat posisi selesai (WIN/LOSS + PnL)
      - get_stats     : hitung dan tampilkan winrate + metrik performa
      - get_open_trades : ambil posisi yang masih terbuka
    """

    def __init__(self):
        self.db: SupabaseClient = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_KEY,
        )
        log.info("✅ Supabase — terhubung")

    # ── INSERT trade baru ─────────────────────────────────────────────
    def insert_trade(self, payload: dict) -> str | None:
        """
        Simpan record trade baru ke tabel `trades`.
        Return UUID row yang baru (dipakai untuk update saat close).
        """
        try:
            res      = self.db.table(Config.SUPABASE_TABLE).insert(payload).execute()
            trade_id = res.data[0]["id"]
            log.info(f"📝 Trade baru dicatat | ID: {trade_id[:8]}...")
            return trade_id
        except Exception as e:
            log.error(f"❌ Supabase insert gagal: {e}")
            return None

    # ── UPDATE saat trade close ───────────────────────────────────────
    def close_trade(self, trade_id: str, exit_price: float,
                    pnl: float, pnl_pct: float, status: str) -> bool:
        """
        Tutup trade: isi exit_price, pnl, pnl_pct, dan status (WIN/LOSS).
        Dipanggil oleh PositionMonitor saat posisi sudah tidak aktif di Binance.
        """
        try:
            self.db.table(Config.SUPABASE_TABLE).update({
                "exit_price" : round(exit_price, 4),
                "pnl"        : round(pnl, 4),
                "pnl_pct"    : round(pnl_pct, 6),
                "status"     : status,
            }).eq("id", trade_id).execute()

            icon = "🟢 WIN" if status == "WIN" else "🔴 LOSS"
            log.info(
                f"{icon} | ID={trade_id[:8]}... | "
                f"exit={exit_price:.2f} | PnL={pnl:+.4f} USDT ({pnl_pct:+.2%})"
            )
            return True
        except Exception as e:
            log.error(f"❌ Supabase close_trade gagal: {e}")
            return False

    # ── QUERY winrate & statistik performa ───────────────────────────
    def get_stats(self) -> dict:
        """
        Hitung statistik lengkap dari semua trade yang sudah selesai (bukan OPEN).

        Metrics yang dihitung:
          - Winrate keseluruhan (%)
          - Winrate LONG vs SHORT terpisah
          - Total PnL kumulatif
          - Rata-rata PnL per trade
          - Trade terbaik dan terburuk
          - Profit factor (gross profit / gross loss)
        """
        try:
            res  = (
                self.db.table(Config.SUPABASE_TABLE)
                .select("status, pnl, pnl_pct, side")
                .neq("status", "OPEN")
                .execute()
            )
            rows = res.data

            if not rows:
                log.info("📊 Belum ada trade yang selesai.")
                return self._empty_stats()

            df      = pd.DataFrame(rows)
            total   = len(df)
            wins    = len(df[df["status"] == "WIN"])
            losses  = len(df[df["status"] == "LOSS"])
            winrate = (wins / total * 100) if total > 0 else 0.0

            pnl_series  = df["pnl"].astype(float)
            total_pnl   = pnl_series.sum()
            avg_pnl     = pnl_series.mean()
            best_trade  = pnl_series.max()
            worst_trade = pnl_series.min()

            # Profit factor = gross profit / |gross loss|
            gross_profit = pnl_series[pnl_series > 0].sum()
            gross_loss   = abs(pnl_series[pnl_series < 0].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            # Winrate per sisi
            long_df   = df[df["side"] == "LONG"]
            short_df  = df[df["side"] == "SHORT"]
            long_wr   = _wr(long_df)
            short_wr  = _wr(short_df)

            # Streak win/loss saat ini
            streak    = self._calc_streak(df)

            stats = {
                "total_trades"  : total,
                "wins"          : wins,
                "losses"        : losses,
                "winrate_pct"   : round(winrate, 2),
                "total_pnl"     : round(total_pnl, 4),
                "avg_pnl"       : round(avg_pnl, 4),
                "best_trade"    : round(best_trade, 4),
                "worst_trade"   : round(worst_trade, 4),
                "profit_factor" : round(profit_factor, 3),
                "long_winrate"  : round(long_wr, 2),
                "short_winrate" : round(short_wr, 2),
                "current_streak": streak,
            }

            self._print_stats(stats)
            return stats

        except Exception as e:
            log.error(f"❌ Gagal hitung stats: {e}")
            return self._empty_stats()

    # ── Helper: current streak ────────────────────────────────────────
    @staticmethod
    def _calc_streak(df: pd.DataFrame) -> str:
        """Hitung streak WIN/LOSS terkini dari riwayat terbaru."""
        statuses = df["status"].tolist()[::-1]  # Terbaru di depan
        if not statuses:
            return "–"
        current  = statuses[0]
        count    = 1
        for s in statuses[1:]:
            if s == current:
                count += 1
            else:
                break
        return f"{count}× {current}"

    @staticmethod
    def _print_stats(s: dict):
        log.info(
            f"\n{'═'*56}\n"
            f"  📊  STATISTIK PERFORMA BOT\n"
            f"{'─'*56}\n"
            f"  Total Trade    : {s['total_trades']:>6} trade\n"
            f"  WIN / LOSS     : {s['wins']:>3} WIN  /  {s['losses']:>3} LOSS\n"
            f"  Winrate        : {s['winrate_pct']:>7.2f} %\n"
            f"{'─'*56}\n"
            f"  Winrate LONG   : {s['long_winrate']:>7.2f} %\n"
            f"  Winrate SHORT  : {s['short_winrate']:>7.2f} %\n"
            f"{'─'*56}\n"
            f"  Total PnL      : {s['total_pnl']:>+10.4f} USDT\n"
            f"  Avg PnL/Trade  : {s['avg_pnl']:>+10.4f} USDT\n"
            f"  Profit Factor  : {s['profit_factor']:>10.3f}\n"
            f"  Best Trade     : {s['best_trade']:>+10.4f} USDT\n"
            f"  Worst Trade    : {s['worst_trade']:>+10.4f} USDT\n"
            f"  Streak Kini    : {s['current_streak']}\n"
            f"{'═'*56}"
        )

    # ── Ambil posisi yang masih OPEN ──────────────────────────────────
    def get_open_trades(self) -> list[dict]:
        try:
            res = (
                self.db.table(Config.SUPABASE_TABLE)
                .select("*")
                .eq("status", "OPEN")
                .execute()
            )
            return res.data or []
        except Exception as e:
            log.error(f"❌ Gagal ambil open trades: {e}")
            return []

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "winrate_pct": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0, "profit_factor": 0.0,
            "long_winrate": 0.0, "short_winrate": 0.0, "current_streak": "–",
        }


def _wr(df: pd.DataFrame) -> float:
    """Helper kecil hitung winrate dari sub-dataframe."""
    if len(df) == 0:
        return 0.0
    return len(df[df["status"] == "WIN"]) / len(df) * 100


# ╔══════════════════════════════════════════════════════════════════════╗
# ║             4. DATA FETCHER (Binance Native SDK)                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class DataFetcher:
    """
    Semua pengambilan data pakai UMFutures dari binance-futures-connector.
    Output selalu berupa DataFrame ber-index datetime UTC.
    """

    _INTERVAL_MS = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }

    def __init__(self, client: UMFutures):
        self.client = client

    # ── OHLCV real-time (untuk sinyal) ───────────────────────────────
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Ambil N candle terakhir.
        Dipakai setiap siklus 15 menit untuk sinyal entry.
        """
        raw = self.client.klines(symbol=symbol, interval=interval, limit=limit)
        return self._parse_klines(raw)

    # ── OHLCV bulk historis (untuk training ML) ───────────────────────
    def fetch_ohlcv_bulk(self, symbol: str, interval: str, years: int = 3) -> pd.DataFrame:
        """
        Ambil data historis besar dengan loop iteratif.
        Binance izinkan max 1.500 candle per request, jadi kita loop.
        Target minimum: 10.000 baris sesuai Config.MIN_TRAIN_ROWS.
        """
        now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - years * 365 * 24 * 3_600_000   # Approx N tahun ke belakang
        all_rows = []

        log.info(f"📥 Bulk fetch: {symbol} {interval} — {years} tahun terakhir...")

        current = start_ms
        while current < now_ms:
            try:
                raw = self.client.klines(
                    symbol    = symbol,
                    interval  = interval,
                    startTime = current,
                    limit     = 1500,
                )
            except ClientError as e:
                log.error(f"❌ Binance API error: {e}")
                break

            if not raw:
                break

            all_rows.extend(raw)
            current = int(raw[-1][0]) + 1
            time.sleep(0.3)     # Jaga rate limit Binance

        df = self._parse_klines(all_rows).drop_duplicates()
        log.info(f"✅ Bulk fetch selesai — {len(df):,} baris")

        if len(df) < Config.MIN_TRAIN_ROWS:
            log.warning(
                f"⚠️  {len(df):,} baris < minimum {Config.MIN_TRAIN_ROWS:,}. "
                f"Coba naikkan TRAIN_YEARS atau ganti ke timeframe lebih kecil."
            )
        return df

    # ── Funding Rate ──────────────────────────────────────────────────
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Ambil funding rate terkini.
        Nilai positif → pasar terlalu long (potensi reversal ke bawah).
        """
        try:
            data = self.client.funding_rate(symbol=symbol, limit=1)
            return float(data[0]["fundingRate"]) if data else 0.0
        except ClientError:
            return 0.0

    # ── Open Interest ──────────────────────────────────────────────────
    def fetch_open_interest(self, symbol: str) -> float:
        """
        Total kontrak futures terbuka.
        OI naik + harga turun → short accumulation → potensi short squeeze.
        """
        try:
            data = self.client.open_interest(symbol=symbol)
            return float(data.get("openInterest", 0.0))
        except ClientError:
            return 0.0

    # ── Balance ───────────────────────────────────────────────────────
    def fetch_balance(self) -> float:
        """Ambil available balance USDT dari akun futures."""
        try:
            for asset in self.client.balance():
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
            return 0.0
        except ClientError as e:
            log.error(f"❌ Gagal ambil balance: {e}")
            return 0.0

    # ── Posisi terbuka ────────────────────────────────────────────────
    def fetch_open_positions(self, symbol: str) -> list[dict]:
        """
        Ambil posisi futures yang masih aktif (positionAmt != 0).
        """
        try:
            all_pos = self.client.get_position_risk(symbol=symbol)
            return [p for p in all_pos if float(p.get("positionAmt", 0)) != 0]
        except ClientError:
            return []

    # ── Parser internal ───────────────────────────────────────────────
    @staticmethod
    def _parse_klines(raw: list) -> pd.DataFrame:
        """
        Konversi raw klines dari Binance ke DataFrame standard.
        Kolom: open, high, low, close, volume — index: timestamp UTC.
        """
        df = pd.DataFrame(raw, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_ts", "quote_vol", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    5. FEATURE ENGINEERING                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

class FeatureEngineer:
    """
    Pipeline pembuatan fitur untuk ML dan sinyal rule-based.

    Urutan wajib diikuti:
      VWAP → RSI/MACD/ATR → Price Action → Derivative Metrics → Label

    JANGAN balik urutan atau tambah fitur setelah labeling
    → bisa menyebabkan data leakage!
    """

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP — indikator favorit institusi & trader profesional.
        harga > VWAP = pasar dikontrol buyer (bias long).
        harga < VWAP = pasar dikontrol seller (bias short).
        """
        df = df.copy()
        cum_vp         = (df["close"] * df["volume"]).cumsum()
        cum_v          = df["volume"].cumsum()
        df["vwap"]     = cum_vp / cum_v
        # Jarak relatif ke VWAP (feature penting buat ML)
        df["vwap_dist_pct"] = (df["close"] - df["vwap"]) / df["vwap"]
        return df

    @staticmethod
    def add_rsi_macd_atr(df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI   → deteksi kelelahan momentum & zona overbought/oversold
        MACD  → konfirmasi arah perubahan tren (sinyal crossover kuat)
        ATR   → ukuran volatilitas pasar → dipakai hitung SL/TP adaptif
        """
        df = df.copy()

        rsi_ind         = RSIIndicator(df["close"], window=14)
        df["rsi"]       = rsi_ind.rsi()

        macd_ind        = MACD(df["close"])
        df["macd"]      = macd_ind.macd()
        df["macd_sig"]  = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        atr_ind         = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"]       = atr_ind.average_true_range()
        df["atr_pct"]   = df["atr"] / df["close"]

        return df

    @staticmethod
    def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
        """
        Vektor price action mentah.
        Berdasarkan riset, kelompok fitur ini paling tinggi feature importance
        di model ML (>60% kontribusi prediktif).
        """
        df = df.copy()

        # Return logaritmik multi-period
        df["log_ret_1"]  = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_5"]  = np.log(df["close"] / df["close"].shift(5))
        df["log_ret_20"] = np.log(df["close"] / df["close"].shift(20))

        # Morfologi candle
        df["body_size"]  = abs(df["close"] - df["open"]) / (df["open"] + 1e-8)
        df["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / (df["open"] + 1e-8)
        df["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"])  / (df["open"] + 1e-8)

        # Volatilitas rolling 20 candle
        df["volatility_20"] = df["log_ret_1"].rolling(20).std()

        # Posisi harga dalam range 20 candle (0 = di low, 1 = di high)
        rh = df["high"].rolling(20).max()
        rl = df["low"].rolling(20).min()
        df["price_position"] = (df["close"] - rl) / (rh - rl + 1e-8)

        # Volume relatif (Z-score)
        vol_mean = df["volume"].rolling(20).mean()
        vol_std  = df["volume"].rolling(20).std() + 1e-8
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

        return df

    @staticmethod
    def add_derivative_metrics(df: pd.DataFrame,
                                funding_rate: float = 0.0,
                                open_interest: float = 0.0) -> pd.DataFrame:
        """
        Masukkan data derivatif sebagai fitur konstan per batch.
        Funding rate dan OI penting untuk deteksi risiko likuidasi massal
        yang sering terjadi di pasar futures crypto.
        """
        df = df.copy()
        df["funding_rate"]  = funding_rate
        df["open_interest"] = open_interest
        return df

    @staticmethod
    def add_labels(df: pd.DataFrame, horizon: int = 3,
                   threshold: float = 0.003) -> pd.DataFrame:
        """
        Label target untuk training ML.
        Cek return N candle ke depan:
          +1  = LONG  jika return > +threshold
          -1  = SHORT jika return < -threshold
           0  = HOLD  jika di antara threshold (noise)
        horizon=3 dan threshold=0.3% adalah parameter default.
        """
        df      = df.copy()
        fut_ret = df["close"].shift(-horizon) / df["close"] - 1
        df["label"] = 0
        df.loc[fut_ret >  threshold, "label"] =  1
        df.loc[fut_ret < -threshold, "label"] = -1
        return df.dropna()

    @classmethod
    def build_features(cls, df: pd.DataFrame,
                       funding_rate: float = 0.0,
                       open_interest: float = 0.0) -> pd.DataFrame:
        """
        Jalankan pipeline lengkap.
        Panggil ini sebelum training maupun inferensi real-time.
        """
        df = cls.add_vwap(df)
        df = cls.add_rsi_macd_atr(df)
        df = cls.add_price_action(df)
        df = cls.add_derivative_metrics(df, funding_rate, open_interest)
        return df

    @staticmethod
    def feature_columns() -> list[str]:
        """Daftar nama kolom yang dipakai model ML (harus konsisten train/infer)."""
        return [
            "vwap_dist_pct",
            "rsi", "macd", "macd_sig", "macd_hist", "atr_pct",
            "log_ret_1", "log_ret_5", "log_ret_20",
            "body_size", "upper_wick", "lower_wick",
            "volatility_20", "price_position", "volume_zscore",
            "funding_rate", "open_interest",
        ]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    6. MACHINE LEARNING MODEL                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

class MLModel:
    """
    Ensemble classifier: RF + GB.
    Kedua model di-average probabilitasnya → lebih robust dari satu model.

    Keputusan training di 1H (bukan 15m):
      Di bawah 1H, ML cenderung belajar noise acak, bukan pola prediktif.
    """

    def __init__(self):
        self.rf  = self._build_rf()
        self.gb  = self._build_gb()
        self.cols = FeatureEngineer.feature_columns()
        self.trained = False

    @staticmethod
    def _build_rf() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators     = 300,
                max_depth        = 8,
                min_samples_leaf = 50,    # Tiap leaf min 50 sampel → cegah overfit
                class_weight     = "balanced",
                random_state     = 42,
                n_jobs           = -1,
            )),
        ])

    @staticmethod
    def _build_gb() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators  = 200,
                learning_rate = 0.05,
                max_depth     = 4,
                subsample     = 0.8,   # 80% sampel per tree → regularisasi
                random_state  = 42,
            )),
        ])

    def train(self, df: pd.DataFrame):
        """
        Training dengan TimeSeriesSplit (5-fold).

        PENTING: Jangan pernah ganti ke KFold biasa untuk data time series!
        KFold random akan menyebabkan look-ahead bias karena
        data masa depan bisa masuk ke set training.
        """
        df = FeatureEngineer.add_labels(df)
        X  = df[self.cols].fillna(0)
        y  = df["label"]

        if len(X) < Config.MIN_TRAIN_ROWS:
            raise ValueError(
                f"❌ Data hanya {len(X):,} baris. "
                f"Minimum {Config.MIN_TRAIN_ROWS:,} untuk model yang reliable!"
            )

        log.info(f"🧠 Training pada {len(X):,} sampel...")
        tscv = TimeSeriesSplit(n_splits=5)

        cv_rf = cross_val_score(self.rf, X, y, cv=tscv, scoring="f1_macro", n_jobs=-1)
        self.rf.fit(X, y)
        log.info(f"  RF  F1-macro: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")

        cv_gb = cross_val_score(self.gb, X, y, cv=tscv, scoring="f1_macro")
        self.gb.fit(X, y)
        log.info(f"  GB  F1-macro: {cv_gb.mean():.3f} ± {cv_gb.std():.3f}")

        log.info("\n" + classification_report(
            y, self.rf.predict(X),
            target_names=["SHORT", "HOLD", "LONG"]
        ))

        self.trained = True
        self.save()

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Prediksi sinyal dari candle terbaru.
        Ensemble: rata-rata probabilitas RF dan GB.

        Return:
            {
              "signal"    : int   →  1=LONG, -1=SHORT, 0=HOLD
              "confidence": float →  probabilitas kelas terpilih (0.0–1.0)
            }
        """
        if not self.trained:
            return {"signal": 0, "confidence": 0.0}

        X = df[self.cols].fillna(0).tail(1)

        p_rf  = self.rf.predict_proba(X)[0]
        p_gb  = self.gb.predict_proba(X)[0]
        proba = (p_rf + p_gb) / 2.0

        classes     = self.rf.classes_
        best_idx    = int(np.argmax(proba))
        best_class  = int(classes[best_idx])
        confidence  = float(proba[best_idx])

        return {"signal": best_class, "confidence": confidence}

    def save(self):
        joblib.dump({"rf": self.rf, "gb": self.gb}, Config.MODEL_PATH)
        log.info(f"💾 Model disimpan → {Config.MODEL_PATH}")

    def load(self) -> bool:
        if not Config.MODEL_PATH.exists():
            return False
        data     = joblib.load(Config.MODEL_PATH)
        self.rf  = data["rf"]
        self.gb  = data["gb"]
        self.trained = True
        log.info(f"📂 Model di-load dari {Config.MODEL_PATH}")
        return True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              7. AI DECISION ENGINE (Gemini API — Free Tier)         ║
# ╚══════════════════════════════════════════════════════════════════════╝

class AIDecisionEngine:
    """
    Keputusan trading dari Google Gemini Flash (GRATIS, 1.500 req/hari).

    Semua data teknikal (VWAP, RSI, MACD, ATR, price action, funding rate,
    open interest, dan prediksi ML) dikirim ke Gemini sebagai konteks.
    Gemini membaca seluruh gambaran pasar dan output-nya berupa JSON:
      {
        "action"     : "LONG" | "SHORT" | "HOLD",
        "confidence" : float (0.0 – 1.0),
        "reasoning"  : "penjelasan singkat keputusan"
      }

    Free tier Gemini Flash:
      - 1.500 request/hari  (bot pakai ~96/hari → sangat aman)
      - 15 request/menit
      - Daftar API key gratis: aistudio.google.com/apikey
    """

    # Gemini REST endpoint — generateContent untuk gemini-1.5-flash
    GEMINI_API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/gemini-2.5-flash:generateContent"
    )
    MODEL = "gemini-2.5-flash"

    # System instruction — dikirim sebagai systemInstruction di Gemini API
    SYSTEM_PROMPT = """Kamu adalah analis trading algoritmik di Binance Futures.
Balas HANYA dengan satu baris JSON, tidak ada teks lain sama sekali.
Format WAJIB (satu baris, tanpa newline):
{"action":"HOLD","confidence":0.75,"reasoning":"max 10 kata alasan"}
Aturan: action hanya LONG/SHORT/HOLD, confidence 0.0-1.0, reasoning maksimal 10 kata."""

    def __init__(self, api_key: str):
        self.api_key          = api_key
        self._last_reasoning  = ""   # Simpan reasoning terakhir untuk logging

    @staticmethod
    def _build_snapshot(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                        ml_result: dict, open_positions: list,
                        balance: float, funding_rate: float,
                        open_interest: float) -> dict:
        """
        Kumpulkan semua data relevan menjadi satu snapshot pasar
        yang akan dikirim ke Claude sebagai konteks.
        """
        last_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        last_1h  = df_1h.iloc[-1]

        # Deteksi MACD crossover
        macd_cross = "none"
        if prev_15m["macd"] <= prev_15m["macd_sig"] and last_15m["macd"] > last_15m["macd_sig"]:
            macd_cross = "bullish_crossover"
        elif prev_15m["macd"] >= prev_15m["macd_sig"] and last_15m["macd"] < last_15m["macd_sig"]:
            macd_cross = "bearish_crossover"

        # Hitung candle terakhir naik/turun berapa persen
        candle_change_pct = round((last_15m["close"] - last_15m["open"]) / last_15m["open"] * 100, 3)

        # Return 5 candle terakhir sebagai konteks momentum
        recent_closes = df_15m["close"].tail(5).round(2).tolist()
        recent_volumes = df_15m["volume_zscore"].tail(5).round(3).tolist()

        return {
            "timestamp_utc"   : str(df_15m.index[-1]),
            "symbol"          : Config.SYMBOL,
            "timeframe_signal": Config.TF_SIGNAL,
            "timeframe_trend" : Config.TF_TREND,

            # ── Harga & VWAP ─────────────────────────────────────────
            "price": {
                "current"         : round(float(last_15m["close"]), 2),
                "vwap_15m"        : round(float(last_15m["vwap"]), 2),
                "vwap_dist_pct"   : round(float(last_15m["vwap_dist_pct"]) * 100, 3),
                "vwap_1h"         : round(float(last_1h["vwap"]), 2),
                "above_vwap_15m"  : bool(last_15m["close"] > last_15m["vwap"]),
                "above_vwap_1h"   : bool(last_1h["close"] > last_1h["vwap"]),
                "candle_change_pct": candle_change_pct,
                "recent_closes_5" : recent_closes,
            },

            # ── Momentum & Tren ──────────────────────────────────────
            "indicators": {
                "rsi_14"          : round(float(last_15m["rsi"]), 2),
                "rsi_zone"        : (
                    "oversold"    if last_15m["rsi"] < Config.RSI_OVERSOLD  else
                    "overbought"  if last_15m["rsi"] > Config.RSI_OVERBOUGHT else
                    "neutral"
                ),
                "macd"            : round(float(last_15m["macd"]), 6),
                "macd_signal"     : round(float(last_15m["macd_sig"]), 6),
                "macd_histogram"  : round(float(last_15m["macd_hist"]), 6),
                "macd_crossover"  : macd_cross,
                "atr_14"          : round(float(last_15m["atr"]), 2),
                "atr_pct"         : round(float(last_15m["atr_pct"]) * 100, 3),
            },

            # ── Price Action ─────────────────────────────────────────
            "price_action": {
                "log_return_1c"   : round(float(last_15m["log_ret_1"]), 6),
                "log_return_5c"   : round(float(last_15m["log_ret_5"]), 6),
                "log_return_20c"  : round(float(last_15m["log_ret_20"]), 6),
                "body_size_pct"   : round(float(last_15m["body_size"]) * 100, 3),
                "upper_wick_pct"  : round(float(last_15m["upper_wick"]) * 100, 3),
                "lower_wick_pct"  : round(float(last_15m["lower_wick"]) * 100, 3),
                "volatility_20c"  : round(float(last_15m["volatility_20"]), 6),
                "price_position"  : round(float(last_15m["price_position"]), 3),
                "volume_zscore"   : round(float(last_15m["volume_zscore"]), 3),
                "recent_vol_zscore_5": recent_volumes,
            },

            # ── Derivatif Futures ────────────────────────────────────
            "derivatives": {
                "funding_rate"    : round(funding_rate * 100, 6),
                "funding_bias"    : (
                    "bullish_overheated" if funding_rate > 0.001 else
                    "bearish_overheated" if funding_rate < -0.001 else
                    "neutral"
                ),
                "open_interest"   : round(open_interest, 2),
            },

            # ── Prediksi ML (Random Forest + Gradient Boosting) ──────
            "ml_ensemble": {
                "signal"          : ml_result["signal"],
                "signal_label"    : (
                    "LONG" if ml_result["signal"] ==  1 else
                    "SHORT" if ml_result["signal"] == -1 else "HOLD"
                ),
                "confidence"      : round(ml_result["confidence"], 4),
                "above_threshold" : ml_result["confidence"] >= Config.ML_CONFIDENCE,
            },

            # ── Kondisi Akun ─────────────────────────────────────────
            "account": {
                "balance_usdt"    : round(balance, 2),
                "open_positions"  : len(open_positions),
                "max_positions"   : Config.MAX_OPEN_TRADES,
                "risk_per_trade"  : f"{Config.RISK_PER_TRADE*100:.0f}%",
                "sl_atr_mult"     : Config.ATR_SL_MULT,
                "tp_atr_mult"     : Config.ATR_TP_MULT,
            },
        }

    def _call_gemini(self, snapshot: dict) -> dict:
        """
        Kirim snapshot pasar ke Gemini API dan parse responsenya.
        Retry otomatis 2x jika gagal (network hiccup, rate limit, dll).

        Gemini API pakai query param ?key=API_KEY (bukan header x-api-key).
        Return: {"action": str, "confidence": float, "reasoning": str}
        """
        import json
        import urllib.request
        import urllib.error
        import urllib.parse

        user_message = (
            "Analisis data pasar berikut dan buat keputusan trading:\n\n"
            f"{json.dumps(snapshot, indent=2, ensure_ascii=False)}\n\n"
            "Berikan keputusan trading dalam format JSON yang diminta."
        )

        # Gemini API: systemInstruction terpisah dari contents
        payload = json.dumps({
            "systemInstruction": {
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]}
            ],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature"    : 0.1,
            },
        }).encode("utf-8")

        # API key dimasukkan sebagai query parameter (bukan header)
        url     = f"{self.GEMINI_API_URL}?key={urllib.parse.quote(self.api_key)}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, 4):   # Max 3 percobaan
            try:
                req  = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                log.debug(f"Gemini raw: {str(body)[:400]}")

                # Ekstrak teks — robust terhadap variasi struktur Gemini 2.5
                raw_text = ""
                try:
                    parts = body["candidates"][0]["content"]["parts"]
                    # Ambil semua part bertipe "text", skip part bertipe "thought"
                    texts = [p["text"] for p in parts if p.get("text") and not p.get("thought")]
                    raw_text = " ".join(texts).strip()
                except (KeyError, IndexError):
                    pass

                if not raw_text:
                    log.warning(f"⚠️  Gemini response kosong, body={str(body)[:200]}")
                    time.sleep(3 * attempt)
                    continue

                # Strip markdown fence kalau ada (```json ... ```)
                if "```" in raw_text:
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()

                # Strip karakter non-JSON di awal/akhir
                start = raw_text.find("{")
                end   = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]

                decision = json.loads(raw_text)

                # Validasi struktur JSON
                action = decision.get("action", "HOLD").upper()
                if action not in ("LONG", "SHORT", "HOLD"):
                    action = "HOLD"

                return {
                    "action"    : action,
                    "confidence": float(decision.get("confidence", 0.0)),
                    "reasoning" : str(decision.get("reasoning", "–")),
                }

            except urllib.error.HTTPError as e:
                # Baca body error dari Anthropic supaya pesan spesifik kelihatan di log
                try:
                    err_body = json.loads(e.read().decode("utf-8"))
                    err_msg  = err_body.get("error", {}).get("message", str(e))
                except Exception:
                    err_msg  = str(e)
                log.warning(
                    f"⚠️  Gemini API HTTP {e.code} (attempt {attempt}/3): {err_msg}"
                )
                if e.code == 429:
                    time.sleep(10 * attempt)
                elif e.code in (400, 401, 403):
                    # 400/401/403 tidak akan berhasil di-retry → langsung keluar
                    log.error(f"❌ Gemini API error fatal ({e.code}) — tidak di-retry.")
                    break
                else:
                    time.sleep(3 * attempt)

            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"⚠️  Gagal parse response Gemini (attempt {attempt}/3): {e}")
                time.sleep(3)

            except Exception as e:
                log.warning(f"⚠️  Error Gemini API (attempt {attempt}/3): {e}")
                time.sleep(3 * attempt)

        # Semua retry gagal → fallback ke HOLD yang aman
        log.error("❌ Gemini API gagal setelah 3 percobaan. Fallback ke HOLD.")
        return {"action": "HOLD", "confidence": 0.0, "reasoning": "API tidak tersedia"}

    def compute(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                ml_result: dict, open_positions: list,
                balance: float, funding_rate: float,
                open_interest: float) -> dict:
        """
        Entry point utama — bangun snapshot, kirim ke Gemini, return keputusan.
        """
        snapshot = self._build_snapshot(
            df_15m, df_1h, ml_result, open_positions,
            balance, funding_rate, open_interest
        )

        decision = self._call_gemini(snapshot)
        self._last_reasoning = decision["reasoning"]

        log.info(
            f"🤖 AI Decision={decision['action']:5s} | "
            f"confidence={decision['confidence']:.0%} | "
            f"ML={ml_result['signal']}({ml_result['confidence']:.0%}) | "
            f"RSI={df_15m.iloc[-1]['rsi']:.1f} | "
            f"ATR={df_15m.iloc[-1]['atr']:.2f}"
        )
        log.info(f"💬 Reasoning: {decision['reasoning']}")

        return {
            "action"    : decision["action"],
            "close"     : float(df_15m.iloc[-1]["close"]),
            "atr"       : float(df_15m.iloc[-1]["atr"]),
            "ml_conf"   : float(ml_result["confidence"]),
            "ai_conf"   : decision["confidence"],
            "signal_ts" : str(df_15m.index[-1]),
        }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║            8. ORDER MANAGER (Binance + Supabase)                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class OrderManager:
    """
    Eksekusi order dan pencatatan transaksi ke Supabase.

    Flow:
      place_order()
        ├── Cek batas posisi (MAX_OPEN_TRADES)
        ├── Hitung size ATR-based
        ├── Market order → Binance
        ├── Stop-loss order → Binance
        ├── Take-profit order → Binance
        └── insert_trade() → Supabase (status=OPEN)
    """

    def __init__(self, fetcher: DataFetcher, db: SupabaseManager):
        self.fetcher = fetcher
        self.client  = fetcher.client
        self.db      = db

    def _calc_size(self, price: float, atr: float, balance: float) -> float:
        """
        ATR-based position sizing:
          risiko_dollar = balance × risk_pct
          sl_distance   = ATR_SL_MULT × ATR
          size          = risiko_dollar / sl_distance / price

        Ini memastikan loss maksimal per trade = 1% modal,
        berapapun volatilitas pasar saat itu.
        """
        risk     = balance * Config.RISK_PER_TRADE
        sl_dist  = Config.ATR_SL_MULT * atr
        return round(risk / sl_dist / price, 4)

    def place_order(self, signal: dict) -> dict | None:
        if signal["action"] == "HOLD":
            return None

        open_pos = self.fetcher.fetch_open_positions(Config.SYMBOL)
        if len(open_pos) >= Config.MAX_OPEN_TRADES:
            log.warning(f"⚠️  Sudah {len(open_pos)} posisi terbuka — skip entry baru")
            return None

        balance = self.fetcher.fetch_balance()
        if balance <= 0:
            log.error("❌ Balance nol atau gagal diambil")
            return None

        side       = "BUY"  if signal["action"] == "LONG"  else "SELL"
        close_side = "SELL" if side == "BUY" else "BUY"
        price      = signal["close"]
        atr        = signal["atr"]
        size       = self._calc_size(price, atr, balance)

        if size <= 0:
            log.warning("⚠️  Kalkulasi size = 0, skip")
            return None

        sl = round(price - Config.ATR_SL_MULT * atr if side == "BUY" else price + Config.ATR_SL_MULT * atr, 2)
        tp = round(price + Config.ATR_TP_MULT * atr if side == "BUY" else price - Config.ATR_TP_MULT * atr, 2)

        try:
            # Set leverage
            self.client.change_leverage(symbol=Config.SYMBOL, leverage=Config.MAX_LEVERAGE)

            # Market order utama
            order    = self.client.new_order(
                symbol   = Config.SYMBOL,
                side     = side,
                type     = "MARKET",
                quantity = size,
            )
            order_id = str(order["orderId"])

            # Stop-Loss (stop market, closePosition)
            self.client.new_order(
                symbol        = Config.SYMBOL,
                side          = close_side,
                type          = "STOP_MARKET",
                stopPrice     = sl,
                closePosition = "true",
            )

            # Take-Profit (take profit market, closePosition)
            self.client.new_order(
                symbol        = Config.SYMBOL,
                side          = close_side,
                type          = "TAKE_PROFIT_MARKET",
                stopPrice     = tp,
                closePosition = "true",
            )

            log.info(
                f"✅ {signal['action']:5s} | "
                f"size={size} | entry≈{price:.2f} | "
                f"SL={sl} | TP={tp} | ID={order_id}"
            )

            # Catat ke Supabase
            self.db.insert_trade({
                "symbol"        : Config.SYMBOL,
                "side"          : signal["action"],
                "entry_price"   : round(price, 4),
                "exit_price"    : None,
                "sl_price"      : sl,
                "tp_price"      : tp,
                "size"          : size,
                "pnl"           : None,
                "pnl_pct"       : None,
                "status"        : "OPEN",
                "ml_confidence" : round(signal["ml_conf"], 4),
                "atr"           : round(atr, 4),
                "order_id"      : order_id,
                "signal_ts"     : signal["signal_ts"],
            })

            return order

        except ClientError as e:
            log.error(f"❌ Order gagal — Binance ClientError: {e}")
            return None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║           9. POSITION MONITOR (Auto-sync ke Supabase)               ║
# ╚══════════════════════════════════════════════════════════════════════╝

class PositionMonitor:
    """
    Deteksi posisi yang sudah ditutup (kena SL atau TP) di Binance,
    lalu update status dan PnL di Supabase secara otomatis.

    Cara kerja:
      1. Ambil daftar trade OPEN dari Supabase
      2. Cek posisi aktif di Binance
      3. Kalau order_id tidak ada di posisi aktif → sudah close
      4. Ambil realized PnL dari income history Binance
      5. Update Supabase dengan WIN/LOSS + PnL + exit price
    """

    def __init__(self, fetcher: DataFetcher, db: SupabaseManager):
        self.fetcher = fetcher
        self.db      = db

    def sync(self):
        """Sinkronisasi status posisi Binance → Supabase."""
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return  # Tidak ada yang perlu di-sync

        # Kumpulkan order_id posisi yang masih aktif di Binance
        active_pos = self.fetcher.fetch_open_positions(Config.SYMBOL)
        active_ids = {str(p.get("orderId", "")) for p in active_pos}

        for trade in open_trades:
            order_id    = str(trade.get("order_id", ""))
            trade_id    = trade["id"]
            entry_price = float(trade["entry_price"])
            size        = float(trade.get("size", 1))
            side        = trade["side"]

            # Kalau tidak ada di active positions → posisi sudah close
            if order_id not in active_ids:
                pnl, exit_px = self._get_realized_pnl(entry_price, size, side)
                pnl_pct      = pnl / (entry_price * size) if (entry_price * size) > 0 else 0.0
                status       = "WIN" if pnl > 0 else "LOSS"
                self.db.close_trade(trade_id, exit_px, pnl, pnl_pct, status)

    def _get_realized_pnl(self, entry: float, size: float, side: str) -> tuple[float, float]:
        """
        Ambil realized PnL dari income history Binance.
        Kalau gagal, estimasi dari harga entry (worst case).
        """
        try:
            income = self.fetcher.client.get_income_history(
                symbol     = Config.SYMBOL,
                incomeType = "REALIZED_PNL",
                limit      = 5,
            )
            if income:
                pnl = float(income[0]["income"])
                # Estimasi exit price dari PnL yang dilaporkan
                if size > 0 and entry > 0:
                    exit_px = round(
                        entry + pnl / size if side == "LONG" else entry - pnl / size,
                        4
                    )
                else:
                    exit_px = entry
                return pnl, exit_px
        except ClientError:
            pass
        return 0.0, entry


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   10. TRADING BOT UTAMA                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════╗
# ║              10. GRACEFUL SHUTDOWN — Auto-Close Semua Posisi        ║
# ╚══════════════════════════════════════════════════════════════════════╝

class GracefulShutdown:
    """
    Tangkap sinyal shutdown dari OS dan tutup semua posisi terbuka
    sebelum proses benar-benar berhenti.

    Sinyal yang ditangkap:
      SIGINT  → Ctrl+C di terminal
      SIGTERM → kill <pid> / docker stop / systemd stop
      SIGHUP  → terminal window ditutup paksa

    Alur saat shutdown:
      1. Set flag _shutting_down = True (cegah siklus baru berjalan)
      2. Ambil semua posisi aktif di Binance
      3. Kirim MARKET order sisi berlawanan (close) untuk tiap posisi
      4. Ambil realized PnL dari income history
      5. Update semua record OPEN di Supabase → WIN / LOSS
      6. Tampilkan statistik akhir
      7. Biarkan proses exit normal
    """

    def __init__(self, fetcher: "DataFetcher", db: "SupabaseManager"):
        self.fetcher       = fetcher
        self.db            = db
        self._shutting_down = False

        # Daftarkan handler untuk semua sinyal relevan
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        # SIGHUP hanya ada di Unix/Linux/macOS
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle)

        # Fallback: atexit dipanggil bahkan kalau proses exit lewat sys.exit()
        atexit.register(self._atexit_handler)

        log.info("🛡️  GracefulShutdown terdaftar (SIGINT / SIGTERM / SIGHUP / atexit)")

    # ── Handler sinyal OS ─────────────────────────────────────────────
    def _handle(self, signum: int, frame):
        sig_name = signal.Signals(signum).name
        log.warning(f"\n⚠️  Sinyal {sig_name} diterima — memulai graceful shutdown...")
        self._shutdown()
        sys.exit(0)

    # ── Fallback atexit (dipanggil saat sys.exit atau unhandled exception) ─
    def _atexit_handler(self):
        if not self._shutting_down:
            log.warning("⚠️  Proses keluar tanpa sinyal — menjalankan emergency close...")
            self._shutdown()

    # ── Proses utama shutdown ─────────────────────────────────────────
    def _shutdown(self):
        """
        Routine lengkap penutupan posisi.
        Flag _shutting_down mencegah double-call kalau sinyal masuk dua kali.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        log.info("=" * 60)
        log.info("🔴 GRACEFUL SHUTDOWN — Menutup semua posisi terbuka...")
        log.info("=" * 60)

        try:
            closed = self._close_all_positions()
            self._sync_supabase_on_exit(closed)
            self.db.get_stats()
            log.info("✅ Semua posisi berhasil ditutup. Bot berhenti dengan aman.")
        except Exception as e:
            log.error(f"❌ Error saat shutdown: {e}", exc_info=True)

    # ── Tutup semua posisi di Binance ─────────────────────────────────
    def _close_all_positions(self) -> list[dict]:
        """
        Ambil semua posisi aktif dari Binance, lalu kirim MARKET order
        di sisi berlawanan untuk menutup tiap posisi.

        Return: list detail posisi yang berhasil ditutup.
        """
        try:
            positions = self.fetcher.fetch_open_positions(Config.SYMBOL)
        except Exception as e:
            log.error(f"❌ Gagal ambil posisi dari Binance: {e}")
            return []

        if not positions:
            log.info("ℹ️  Tidak ada posisi terbuka — tidak ada yang perlu ditutup.")
            return []

        log.info(f"📋 Ditemukan {len(positions)} posisi terbuka. Memulai penutupan...")
        closed = []

        for pos in positions:
            symbol    = pos.get("symbol", Config.SYMBOL)
            amt       = float(pos.get("positionAmt", 0))
            entry_px  = float(pos.get("entryPrice", 0))

            if amt == 0:
                continue

            # positionAmt positif = LONG → tutup dengan SELL
            # positionAmt negatif = SHORT → tutup dengan BUY
            close_side = "SELL" if amt > 0 else "BUY"
            close_qty  = abs(amt)
            side_label = "LONG" if amt > 0 else "SHORT"

            try:
                # Batalkan semua order terbuka dulu (SL/TP) supaya tidak konflik
                self._cancel_open_orders(symbol)

                # Kirim market close order
                order = self.fetcher.client.new_order(
                    symbol    = symbol,
                    side      = close_side,
                    type      = "MARKET",
                    quantity  = close_qty,
                    params    = {"reduceOnly": "true"},
                )

                log.info(
                    f"  ✅ Close {side_label:5s} {symbol} | "
                    f"qty={close_qty} | entry≈{entry_px:.2f} | "
                    f"orderID={order.get('orderId', '?')}"
                )

                closed.append({
                    "symbol"    : symbol,
                    "side"      : side_label,
                    "qty"       : close_qty,
                    "entry_px"  : entry_px,
                    "order_id"  : str(order.get("orderId", "")),
                })

                # Beri jeda kecil antar order supaya tidak kena rate limit
                time.sleep(0.3)

            except ClientError as e:
                log.error(f"  ❌ Gagal close {side_label} {symbol}: {e}")

        return closed

    # ── Batalkan order SL/TP yang masih pending ───────────────────────
    def _cancel_open_orders(self, symbol: str):
        """
        Batalkan semua open order (SL, TP, limit) untuk symbol ini.
        Wajib dilakukan sebelum close position supaya tidak double-close.
        """
        try:
            self.fetcher.client.cancel_open_orders(symbol=symbol)
            log.info(f"  🗑️  Semua pending order {symbol} dibatalkan")
        except ClientError as e:
            # Kalau tidak ada order, Binance return error kode -2011 — aman diabaikan
            if "-2011" not in str(e):
                log.warning(f"  ⚠️  Gagal cancel orders {symbol}: {e}")

    # ── Sinkron status penutupan ke Supabase ──────────────────────────
    def _sync_supabase_on_exit(self, closed_positions: list[dict]):
        """
        Setelah semua posisi ditutup di Binance, update record OPEN
        yang ada di Supabase menjadi WIN atau LOSS.

        Ambil realized PnL dari income history Binance untuk akurasi.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        log.info(f"📝 Sinkronisasi {len(open_trades)} record OPEN ke Supabase...")

        # Ambil income history sekali untuk semua — lebih efisien
        pnl_map = self._fetch_recent_pnl_map()

        for trade in open_trades:
            trade_id    = trade["id"]
            order_id    = str(trade.get("order_id", ""))
            entry_price = float(trade["entry_price"])
            size        = float(trade.get("size", 0))
            side        = trade.get("side", "LONG")

            # Cari PnL berdasarkan order_id kalau ada di map
            pnl = pnl_map.get(order_id, None)

            if pnl is None:
                # Fallback: cari dari posisi yang baru ditutup
                matched = next(
                    (p for p in closed_positions if p["order_id"] == order_id), None
                )
                # Kalau tidak ketemu juga, anggap break-even (PnL=0)
                pnl = 0.0
                if matched:
                    log.debug(f"  Posisi {order_id[:8]} ditemukan di closed_positions")

            pnl_pct  = pnl / (entry_price * size) if (entry_price * size) > 0 else 0.0
            exit_px  = self._estimate_exit(entry_price, pnl, size, side)
            status   = "WIN" if pnl > 0 else "LOSS"

            self.db.close_trade(trade_id, exit_px, pnl, pnl_pct, status)

    def _fetch_recent_pnl_map(self) -> dict[str, float]:
        """
        Ambil income history Binance dan buat mapping order_id → PnL.
        Dipakai untuk mencocokkan trade Supabase dengan realized PnL aktual.
        """
        pnl_map = {}
        try:
            income_list = self.fetcher.client.get_income_history(
                symbol     = Config.SYMBOL,
                incomeType = "REALIZED_PNL",
                limit      = 50,   # Ambil 50 entry terbaru
            )
            for entry in income_list:
                oid = str(entry.get("tradeId", entry.get("orderId", "")))
                pnl = float(entry.get("income", 0))
                pnl_map[oid] = pnl
        except ClientError as e:
            log.warning(f"⚠️  Gagal ambil income history: {e}")
        return pnl_map

    @staticmethod
    def _estimate_exit(entry: float, pnl: float, size: float, side: str) -> float:
        """Estimasi exit price dari realized PnL."""
        if size <= 0 or entry <= 0:
            return entry
        return round(
            entry + pnl / size if side == "LONG" else entry - pnl / size,
            4
        )

    @property
    def is_shutting_down(self) -> bool:
        """Cek dari luar apakah proses sedang dalam mode shutdown."""
        return self._shutting_down


class TradingBot:
    """
    Orkestrator utama yang nyatuin semua komponen:

    DataFetcher
      ↓
    FeatureEngineer
      ↓
    MLModel  ────────────────┐
      ↓                      ↓
    SignalEngine ← (rule-based + ML gate)
      ↓
    OrderManager → Binance + Supabase
      ↓
    PositionMonitor → sync close ke Supabase
      ↓
    SupabaseManager.get_stats() → tampilkan winrate
    """

    def __init__(self):
        self.client   = create_binance_client()
        self.db       = SupabaseManager()
        self.fetcher  = DataFetcher(self.client)
        self.ml       = MLModel()
        self.orders   = OrderManager(self.fetcher, self.db)
        self.monitor  = PositionMonitor(self.fetcher, self.db)
        self.ai       = AIDecisionEngine(Config.GEMINI_API_KEY)
        self._cycle   = 0
        # Daftarkan handler shutdown — harus setelah fetcher & db siap
        self.shutdown = GracefulShutdown(self.fetcher, self.db)

    def initialize(self):
        """
        Load model kalau ada, kalau belum → training dari data historis.
        Setelah siap, tampilkan statistik terakhir dari Supabase.
        """
        if not self.ml.load():
            log.info("🧠 Belum ada model tersimpan. Mulai training...")
            df_raw  = self.fetcher.fetch_ohlcv_bulk(
                Config.SYMBOL, Config.TF_ML, Config.TRAIN_YEARS
            )
            fr = self.fetcher.fetch_funding_rate(Config.SYMBOL)
            oi = self.fetcher.fetch_open_interest(Config.SYMBOL)
            df = FeatureEngineer.build_features(df_raw, fr, oi)
            self.ml.train(df)
        else:
            log.info("✅ Model ML siap.")

        # Tampilkan statistik awal dari database
        self.db.get_stats()

    def run_once(self):
        """
        Satu siklus lengkap setiap 15 menit:
          sync → fetch → feature → predict → signal → order

        Siklus dibatalkan kalau bot sedang dalam proses shutdown.
        """
        # Jangan jalankan siklus baru kalau sedang shutdown
        if self.shutdown.is_shutting_down:
            return

        self._cycle += 1
        log.info(f"{'─'*60}")
        log.info(f"🔁 Siklus #{self._cycle} — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

        try:
            # 1. Sinkron posisi yang sudah close di Binance ke Supabase
            self.monitor.sync()

            # 2. Ambil data OHLCV terbaru
            df_15m = self.fetcher.fetch_ohlcv(Config.SYMBOL, Config.TF_SIGNAL, limit=300)
            df_1h  = self.fetcher.fetch_ohlcv(Config.SYMBOL, Config.TF_TREND,  limit=100)
            fr     = self.fetcher.fetch_funding_rate(Config.SYMBOL)
            oi     = self.fetcher.fetch_open_interest(Config.SYMBOL)

            # 3. Feature engineering
            df_15m = FeatureEngineer.build_features(df_15m, fr, oi)
            df_1h  = FeatureEngineer.build_features(df_1h,  fr, oi)

            # 4. Prediksi ML (Random Forest + Gradient Boosting)
            ml_result = self.ml.predict(df_15m)

            # 5. Ambil kondisi akun real-time untuk konteks AI
            balance      = self.fetcher.fetch_balance()
            open_pos     = self.fetcher.fetch_open_positions(Config.SYMBOL)

            # 6. Keputusan AI — Claude membaca semua data dan output LONG/SHORT/HOLD
            signal = self.ai.compute(
                df_15m       = df_15m,
                df_1h        = df_1h,
                ml_result    = ml_result,
                open_positions = open_pos,
                balance      = balance,
                funding_rate = fr,
                open_interest = oi,
            )

            # 7. Eksekusi order (kalau tidak HOLD)
            if signal["action"] != "HOLD":
                self.orders.place_order(signal)
            else:
                log.info("💤 HOLD — AI memutuskan tidak ada entry yang layak")

            # 8. Tampilkan stats setiap 10 siklus
            if self._cycle % 10 == 0:
                self.db.get_stats()

        except Exception as e:
            log.error(f"❌ Error di siklus #{self._cycle}: {e}", exc_info=True)

    def start(self):
        log.info("🚀 TradingBot dimulai!")
        self.initialize()

        # Langsung jalankan satu siklus pertama
        self.run_once()

        # Jadwalkan tiap 15 menit
        schedule.every(15).minutes.do(self.run_once)
        log.info("⏰ Scheduler aktif — eksekusi tiap 15 menit")

        while True:
            schedule.run_pending()
            time.sleep(10)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         ENTRY POINT                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()