# Prediksi Harga Laptop Bekas
Aplikasi machine learning untuk memprediksi harga laptop bekas berdasarkan spesifikasi hardware.

Deskripsi
Proyek ini menggunakan model regresi linear untuk memprediksi harga laptop bekas berdasarkan berbagai spesifikasi seperti brand, processor, RAM, penyimpanan, dan kartu grafis. Dataset yang digunakan berisi informasi laptop dari berbagai merek dengan harga dalam Rupee India (INR) yang dikonversi ke Rupiah Indonesia (IDR).

# Fitur
- Prediksi harga laptop berdasarkan spesifikasi
- Mendukung berbagai merek laptop (Lenovo, HP, ASUS, Dell, Acer, dll)
- Mendukung berbagai jenis processor (Intel, AMD)
- Antarmuka pengguna yang mudah digunakan dengan Streamlit
- Dataset
- Dataset yang digunakan adalah "Cleaned_Laptop_data.csv" yang berisi informasi tentang laptop dengan kolom-kolom berikut:
  brand: Merek laptop
  processor_brand: Merek processor (Intel, AMD)
  processor_name: Nama processor (Core i3, Core i5, Core i7, Ryzen, dll)
  ram_gb: Kapasitas RAM dalam GB
  ssd: Kapasitas SSD dalam GB
  hdd: Kapasitas HDD dalam GB
  graphic_card_gb: Kapasitas kartu grafis dalam GB
  latest_price: Harga terbaru dalam Rupee India (dikonversi ke IDR)
  old_price: Harga lama dalam Rupee India (dikonversi ke IDR)

# Teknologi yang Digunakan
- Python 3
- Pandas untuk manipulasi data
- Scikit-learn untuk model machine learning
- Streamlit untuk antarmuka pengguna
- Joblib untuk menyimpan dan memuat model

# Cara Menggunakan
- Prasyarat
  pip install pandas scikit-learn streamlit joblib

- Melatih Model
  python train_model.py
Perintah ini akan melatih model menggunakan dataset dan menyimpan model serta mapping kategori ke folder model/.

- Menjalankan Aplikasi
  streamlit run app.py
Perintah ini akan menjalankan aplikasi Streamlit yang dapat diakses melalui browser.

# Pengembangan Lebih Lanjut
Beberapa ide untuk pengembangan lebih lanjut:

- Menambahkan lebih banyak fitur seperti ukuran layar, berat, dll
- Mencoba model machine learning lain seperti Random Forest atau XGBoost
- Menambahkan visualisasi data untuk memahami faktor-faktor yang mempengaruhi harga
- Menambahkan fitur untuk membandingkan harga prediksi dengan harga pasar saat ini
