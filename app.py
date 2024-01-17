import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memprediksi Predikat Karyawan
def predict_predikat_karyawan(data, model):
    prediction = model.predict(data)
    return prediction[0]

# Menampilkan confusion matrix
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Baik', 'Cukup', 'Kurang', 'Buruk'], yticklabels=['Baik', 'Cukup', 'Kurang', 'Buruk'])
    st.pyplot(fig)

# Membaca model dari file pickle
with open('decision_tree_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Tampilan aplikasi Streamlit
st.title('Aplikasi Penilaian Predikat Karyawan')
st.write("Selamat datang di Aplikasi Penilaian Predikat Karyawan. Aplikasi ini menggunakan model Decision Tree untuk memPenilaian predikat karyawan berdasarkan beberapa variabel.")

# Formulir untuk pengisian data karyawan
st.sidebar.header('Menu')
menu_selection = st.sidebar.radio('Pilih Menu', ['Penilaian Karyawan', 'Akurasi dan Confusion Matrix'])

if menu_selection == 'Penilaian Karyawan':
    # Formulir untuk pengisian data karyawan
    st.header('Isi Formulir Data Karyawan')
    absen = st.number_input('Jumlah Absen', min_value=0, max_value=30, step=1)
    total_paket = st.number_input('Total Paket', min_value=0, max_value=10000, step=50)
    cod_tepat_waktu = st.number_input('Jumlah COD Tepat Waktu', min_value=0, max_value=10000, step=1)
    paket_gagal_diantar = st.number_input('Jumlah Paket Gagal Diantar', min_value=0, max_value=10000, step=10)
    paket_dnr = st.number_input('Jumlah Paket DNR', min_value=0, max_value=10000, step=1)

    # Tombol untuk memprediksi
    if st.button('Penilaian Predikat Karyawan'):
        # Mengonversi data formulir ke dataframe
        input_data = pd.DataFrame({
            'absen': [absen],
            'total_paket': [total_paket],
            'cod_tepat_waktu': [cod_tepat_waktu],
            'paket_gagal_diantar': [paket_gagal_diantar],
            'paket_dnr': [paket_dnr]
        })

        # Menyesuaikan fitur yang digunakan oleh model
        fitur = ['absen', 'total_paket', 'cod_tepat_waktu', 'paket_gagal_diantar', 'paket_dnr']
        input_data = input_data[fitur]

        # Melakukan prediksi
        prediksi = predict_predikat_karyawan(input_data, model)

        # Menampilkan hasil prediksi
        st.success(f'Penilaian Predikat Karyawan: {prediksi}')

elif menu_selection == 'Akurasi dan Confusion Matrix':
    # Membaca data untuk evaluasi model
    data_evaluasi = pd.read_csv("jnt.csv", sep=';')

    # Menyesuaikan fitur yang digunakan oleh model
    fitur = ['absen', 'total_paket', 'cod_tepat_waktu', 'paket_gagal_diantar', 'paket_dnr']
    
    data_evaluasi = data_evaluasi.fillna(0)  # Mengganti NaN dengan nilai 0


    # Memisahkan data menjadi fitur (X) dan label (y)
    X_eval = data_evaluasi[fitur]
    y_true = data_evaluasi['Predikat_Karyawan']

    # Melakukan prediksi
    y_pred = model.predict(X_eval)

    # Menghitung akurasi
    akurasi = accuracy_score(y_true, y_pred)
    st.write(f'Akurasi Model: {akurasi}')

    # Menampilkan confusion matrix
    show_confusion_matrix(y_true, y_pred)
    st.pyplot()
