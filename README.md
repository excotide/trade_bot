## Clone Repositori:

### Bash
git clone https://github.com/username/trade_bot.git
cd trade_bot

### Buat Virtual Environment:

### Bash
python -m venv venv

#### Aktivasi (Windows)
venv\Scripts\activate

#### Aktivasi (Linux/Mac)
source venv/bin/activate
Install Dependensi:

#### Bash
pip install ccxt pandas pandas_ta scikit-learn transformers torch feedparser python-dotenv
Konfigurasi Environment:
Buat file .env di folder utama dan masukkan API Key Demo Anda:

Cuplikan kode
BINANCE_FUTURES_API_KEY=your_demo_api_key_here
BINANCE_FUTURES_SECRET_KEY=your_demo_secret_key_here
Jalankan Bot:

Bash
python bot.py