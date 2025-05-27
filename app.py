import streamlit as st
import pandas as pd
import joblib
import json

# Load model
model = joblib.load("model/laptop_price_model.pkl")

# Load mapping kategori
with open('model/category_mapping.json', 'r') as f:
    mapping_data = json.load(f)

brand_mapping = mapping_data['brand_mapping']
processor_brand_mapping = mapping_data['processor_brand_mapping']
processor_name_mapping = mapping_data['processor_name_mapping']

# Invert mapping untuk tampilan di UI
brand_options = list(brand_mapping.keys())
processor_brand_options = list(processor_brand_mapping.keys())
processor_name_options = list(processor_name_mapping.keys())

st.title("Prediksi Harga Laptop Bekas")

# Input user
brand = st.selectbox("Brand", brand_options)
processor_brand = st.selectbox("Processor Brand", processor_brand_options)
processor_name = st.selectbox("Processor Name", processor_name_options)
ram_gb = st.slider("RAM (GB)", 2, 64, 8)
ssd = st.slider("SSD (GB)", 0, 2000, 256)
hdd = st.slider("HDD (GB)", 0, 2000, 0)
graphic_card_gb = st.slider("Graphic Card (GB)", 0, 16, 2)

# Menggunakan mapping untuk encoding
input_df = pd.DataFrame({
    'brand_encoded': [brand_mapping[brand]],
    'processor_brand_encoded': [processor_brand_mapping[processor_brand]],
    'processor_name_encoded': [processor_name_mapping[processor_name]],
    'ram_gb': [ram_gb],
    'ssd': [ssd],
    'hdd': [hdd],
    'graphic_card_gb': [graphic_card_gb]
})

# Prediksi harga
prediksi = model.predict(input_df)[0]
st.subheader(f"Prediksi Harga: Rp {int(prediksi):,}")