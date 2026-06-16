# Proyek Analisis Sentimen

Submission ini menggunakan ulasan Google Play berbahasa Indonesia sebagai data hasil scraping mandiri. Dataset dilabeli menjadi tiga kelas sentimen (`negatif`, `netral`, `positif`) dari rating ulasan, lalu dipakai untuk melatih dan membandingkan tiga skema model machine learning berbasis fitur teks.

## Struktur Berkas

- `scrape_reviews.py`: kode scraping ulasan Google Play dan pelabelan awal.
- `train_sentiment.py`: utilitas training, evaluasi, dan inference.
- `notebook_pelatihan_model.ipynb`: notebook pelatihan model dengan output dan bukti inference.
- `google_play_reviews_sentiment.csv`: dataset hasil scraping.
- `requirements.txt`: dependensi proyek.

## Cara Menjalankan Ulang

Install dependensi:

```bash
pip install -r requirements.txt
```

Scrape ulang dataset:

```bash
python scrape_reviews.py --target 12000 --output google_play_reviews_sentiment.csv
```

Jalankan training dari terminal:

```bash
python train_sentiment.py --dataset google_play_reviews_sentiment.csv
```

Notebook sudah dijalankan dan berisi output agar reviewer tidak perlu menjalankan ulang untuk melihat hasil.
