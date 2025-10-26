import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

def _safe_name(up):
    try:
        return up.name
    except Exception:
        return ""

def _safe_val(x):
    try:
        return float(x)
    except Exception:
        return x  # deixa string/None como está


# Função para filtro passa-baixa


def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Frequência de Nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# Título do app
st.set_page_config(layout="wide")
st.title("Análise de Dados: Interpolação, Detrend e Filtro Passa-Baixa")

# Carregar o arquivo de texto
uploaded_file_acc = st.file_uploader(
    "Escolha um arquivo do acelerômetro do smartphone", type=["txt"])
# Lê o arquivo como DataFrame
if uploaded_file_acc:
    df = pd.read_csv(uploaded_file_acc, sep=";", engine='python')
    tempo = df.iloc[:, 0].values
    acc_x = df.iloc[:, 1].values
    acc_y = df.iloc[:, 2].values
    acc_z = df.iloc[:, 3].values

    new_fs = 100
    interpf = scipy.interpolate.interp1d(tempo, acc_x)
    time_ = np.arange(
        start=tempo[0], stop=tempo[len(tempo)-1], step=10)
    x_ = interpf(time_)
    time_interpolated, acc_x_interpolated = time_/1000, x_
    interpf = scipy.interpolate.interp1d(tempo, acc_y)
    time_ = np.arange(
        start=tempo[0], stop=tempo[len(tempo)-1], step=10)
    y_ = interpf(time_)
    time_interpolated, acc_y_interpolated = time_/1000, y_
    interpf = scipy.interpolate.interp1d(tempo, acc_z)
    time_ = np.arange(
        start=tempo[0], stop=tempo[len(tempo)-1], step=10)
    z_ = interpf(time_)
    time_interpolated, acc_z_interpolated = time_/1000, z_

    acc_x_detrended = detrend(acc_x_interpolated)
    acc_y_detrended = detrend(acc_y_interpolated)
    acc_z_detrended = detrend(acc_z_interpolated)

    # Filtro passa-baixa (10 Hz)
    cutoff = 2.5  # Frequência de corte
    acc_x_filtered = low_pass_filter(
        acc_x_detrended, cutoff, new_fs)
    acc_y_filtered = low_pass_filter(
        acc_y_detrended, cutoff, new_fs)
    acc_z_filtered = low_pass_filter(
        acc_z_detrended, cutoff, new_fs)
    acc_norm_filtered = np.sqrt(
        acc_x_filtered**2+acc_y_filtered**2+acc_z_filtered**2)
    

    valor_acc = st.slider(
        "Ajuste o trigger do acc", min_value=0, max_value=len(time_interpolated), value=0)
    time_interpolated = time_interpolated - \
        time_interpolated[valor_acc]

    if np.mean(acc_x) > np.mean(acc_y):
      ml_acc = np.sqrt(acc_y_filtered**2)
      v_acc = np.sqrt(acc_x_filtered**2)
      ap_acc = np.sqrt(acc_z_filtered**2)
    else:
      v_acc = np.sqrt(acc_y_filtered**2)
      ml_acc = np.sqrt(acc_x_filtered**2)
      ap_acc = np.sqrt(acc_z_filtered**2)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(time_interpolated[0:2000], v_acc[0:2000], 'k-')
    ax.plot([0, 0], [0, 30], 'r--')
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)






