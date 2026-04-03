import feedparser
from transformers import pipeline
import time

print("⏳ Sedang memuat AI FinBERT ke dalam memori komputer... (Hanya butuh waktu saat pertama kali dijalankan)")
# Mengunduh dan memuat model FinBERT dari Hugging Face secara gratis
ai_sentimen = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ AI FinBERT Siap!\n")

def ambil_berita_kripto(jumlah_berita=5):
    """
    Fungsi untuk menyedot berita terbaru dari RSS Feed CoinTelegraph.
    """
    url_rss = "https://cointelegraph.com/rss"
    print(f"📡 Mengambil {jumlah_berita} berita terbaru dari {url_rss}...\n")
    
    try:
        feed = feedparser.parse(url_rss)
        berita_terkini = feed.entries[:jumlah_berita]
        return berita_terkini
    except Exception as e:
        print(f"❌ Gagal mengambil berita: {e}")
        return []

def analisis_sentimen(berita_list):
    """
    Fungsi untuk menyuruh AI membaca setiap judul berita dan memberikan skor.
    """
    print("="*60)
    print("🧠 HASIL ANALISIS SENTIMEN OLEH AI FINBERT")
    print("="*60)
    
    total_skor = 0
    
    for item in berita_list:
        judul = item.title
        # Menyuruh AI menganalisis teks judul
        hasil_ai = ai_sentimen(judul)[0] 
        
        label = hasil_ai['label'].upper() # POSITIVE, NEGATIVE, atau NEUTRAL
        skor_keyakinan = hasil_ai['score'] * 100 # Diubah ke persentase
        
        # Format warna untuk terminal (Opsional, agar lebih mudah dibaca)
        if label == "POSITIVE":
            simbol = "🟢"
            # Jika positif, kita tambah skornya
            total_skor += skor_keyakinan
        elif label == "NEGATIVE":
            simbol = "🔴"
            # Jika negatif, kita kurangi skornya
            total_skor -= skor_keyakinan
        else:
            simbol = "⚪"
            # Jika netral, skor tidak banyak berubah
        
        print(f"📰 Judul : {judul}")
        print(f"🤖 Hasil : {simbol} {label} (Keyakinan: {skor_keyakinan:.1f}%)")
        print("-" * 60)
        
        time.sleep(0.5) # Jeda sedikit agar tulisan tidak muncul terlalu cepat

# ==========================================
# 3. BLOK EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # 1. Ambil 5 berita terbaru
    daftar_berita = ambil_berita_kripto(jumlah_berita=5)
    
    # 2. Jika berhasil mendapat berita, lakukan analisis
    if daftar_berita:
        analisis_sentimen(daftar_berita)
    else:
        print("⚠️ Tidak ada berita yang bisa dianalisis saat ini.")