from google import genai
import os
from dotenv import load_dotenv
import time

# 1. Memuat variabel dari file .env ke dalam sistem
load_dotenv()

# 2. Mengambil API key dari file .env
# Pastikan nama di dalam kurung sama persis dengan yang ada di file .env
kunci_api = os.getenv("GEMINI_API_KEY")

# 3. Cek apakah API Key berhasil dimuat (opsional, untuk memastikan)
if not kunci_api:
    raise ValueError("❌ API Key tidak ditemukan! Cek file .env Anda.")

# 4. Inisialisasi Client menggunakan variabel kunci_api
client = genai.Client(api_key=kunci_api)

daftar_koin = ["BTC", "ETH", "SOL"]

for koin in daftar_koin:
    prompt = f"Analisa pergerakan harga {koin} saat ini dengan RSI dan ATR..."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Atau ganti ke 3.1-flash-lite jika sudah tersedia
            contents=prompt
        )
        print(f"🤖 AI Decision untuk {koin}: {response.text}")
        
    except Exception as e:
        print(f"❌ Gemini API gagal untuk {koin}: {e}")
        
    # Jeda 4 detik untuk mencegah error 429
    time.sleep(4)