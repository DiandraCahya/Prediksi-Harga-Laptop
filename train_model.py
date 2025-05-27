# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json

# Load data
df = pd.read_csv("data/Cleaned_Laptop_data.csv")

# Bersihkan data
df.dropna(inplace=True)
df = df[df['latest_price'] < 50000]  # filter outlier (optional)

# Konversi harga dari Rupee India (INR) ke Rupiah Indonesia (IDR)
# 1 Rupee India (INR) ≈ 190 Rupiah Indonesia (IDR) - nilai kurs dapat disesuaikan
kurs_inr_to_idr = 190
df['latest_price'] = df['latest_price'] * kurs_inr_to_idr
df['old_price'] = df['old_price'] * kurs_inr_to_idr

# Bersihkan kolom numerik teks
def to_number(x):
    try:
        return int(str(x).split()[0])
    except:
        return 0

df['ram_gb'] = df['ram_gb'].apply(to_number)
df['ssd'] = df['ssd'].apply(to_number)
df['hdd'] = df['hdd'].apply(to_number)
df['graphic_card_gb'] = df['graphic_card_gb'].apply(to_number)

# Simpan mapping untuk brand
brand_categories = df['brand'].unique()
brand_mapping = {brand: idx for idx, brand in enumerate(brand_categories)}

# Simpan mapping untuk processor brand
processor_brand_categories = df['processor_brand'].unique()
processor_brand_mapping = {brand: idx for idx, brand in enumerate(processor_brand_categories)}

# Simpan mapping untuk processor name
processor_name_categories = df['processor_name'].unique()
processor_name_mapping = {name: idx for idx, name in enumerate(processor_name_categories)}

# Encode fitur kategorikal
df['brand_encoded'] = df['brand'].map(brand_mapping)
df['processor_brand_encoded'] = df['processor_brand'].map(processor_brand_mapping)
df['processor_name_encoded'] = df['processor_name'].map(processor_name_mapping)

# Fitur dan target
X = df[['brand_encoded', 'processor_brand_encoded', 'processor_name_encoded', 'ram_gb', 'ssd', 'hdd', 'graphic_card_gb']]
y = df['latest_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f} Rupiah")
print(f"Model dilatih dengan harga dalam Rupiah Indonesia (IDR)")

# Simpan model
joblib.dump(model, "model/laptop_price_model.pkl")

# Simpan mapping kategori untuk digunakan di app.py
mapping_data = {
    'brand_mapping': brand_mapping,
    'processor_brand_mapping': processor_brand_mapping,
    'processor_name_mapping': processor_name_mapping
}

with open('model/category_mapping.json', 'w') as f:
    json.dump(mapping_data, f)

print("Model dan mapping kategori berhasil disimpan!")