import pandas as pd
from apyori import apriori
from fuzzywuzzy import fuzz

# Membaca data penjualan
data = pd.read_csv("data_penjualan.csv")

# Mengubah tanggal menjadi format datetime
data["Tanggal"] = pd.to_datetime(data["Tanggal"])

# Apriori algorithm
apriori_results = list(apriori(data.groupby("Tanggal")["Nama Barang"].apply(list), min_support=0.1, min_confidence=0.5))

# Fuzzy matching untuk mencocokkan barang dengan aturan Apriori
def fuzzy_match(barang, aturan_apriori):
    for item in aturan_apriori:
        if fuzz.ratio(barang, list(item.items)[0]) > 80:
            return item
    return None

# Prediksi penjualan
def prediksi_penjualan(barang, tanggal, aturan_apriori):
    aturan = fuzzy_match(barang, aturan_apriori)
    if aturan is None:
        return None
    
    prediksi = 0
    for item in list(aturan.ordered_statistics):
        if item.items_base and list(item.items_base)[0] != barang:
            data_penjualan_item = data[(data["Nama Barang"] == list(item.items_base)[0]) & (data["Tanggal"] == tanggal)]
            if not data_penjualan_item.empty:
                penjualan_item_pada_tanggal = data_penjualan_item.iloc[0]["Jumlah Terjual"]
                prediksi += penjualan_item_pada_tanggal * item.confidence
    
    return prediksi

# Contoh penggunaan
barang_prediksi = "Keripik Singkong"
tanggal_prediksi = pd.to_datetime("2023-01-05")

aturan_apriori = apriori_results
prediksi = prediksi_penjualan(barang_prediksi, tanggal_prediksi, aturan_apriori)

print(f"Prediksi penjualan {barang_prediksi} pada tanggal {tanggal_prediksi}: {prediksi}")
