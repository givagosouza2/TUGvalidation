import streamlit as st
import scipy
import pandas as pd
import numpy as np
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

tab1, tab2, tab3 = st.tabs(['Kinematics','Acceleration','Angular velocity'])
with tab1:
    # Carregar o arquivo de texto
    uploaded_file_kinem = st.file_uploader(
        "Escolha um arquivo da cinemática", type=["csv"])
    if uploaded_file_kinem:
        df = pd.read_csv(uploaded_file_kinem, sep=",", engine='python')
        disp_x = (df.iloc[:, 0].values)/1000
        disp_y = (df.iloc[:, 1].values)/1000
        disp_z = (df.iloc[:, 2].values)/1000    
        # cinemática
        original_fs_kinem = 100
        new_fs_kinem = 100
        cutoff_kinem = 2
        time_original_kinem = np.arange(
            0, len(disp_y)) / original_fs_kinem
    
        valor = st.slider("Ajustar o trigger da cinemática", min_value=0, max_value=len(
            time_original_kinem), value=0)
        time_original_kinem = time_original_kinem - \
        time_original_kinem[valor]

        dy_dx = np.diff(disp_z) / np.diff(time_original_kinem)
        baseline = np.mean(dy_dx)
        sd_baseline = np.std(dy_dx)
        # Diferença entre tempos consecutivos

        peaks, properties = find_peaks(-1*disp_y, height=-1)
        onsets = []
        offsets = []
        for peak in peaks:
            # Busca para trás: início da queda
            for i in range(peak, 1, -1):
                if disp_y[i] > disp_y[i-1]:
                    onsets.append(i)
                    break

        # Busca para frente: fim da queda
        for i in range(peak, len(disp_y)-1):
            if disp_y[i] > disp_y[i+1]:
                offsets.append(i)
                break

        standing_time = []
        for values in onsets:
            for idx, i in enumerate(disp_z[values:values+200]):
                if i == max(disp_z[values:values+200]):
                    standing_time.append(
                        time_original_kinem[values+idx])
                    break

        sitting_time = []
        for values in offsets:
            for idx, i in enumerate(disp_z[values-400:values]):
                if i == max(disp_z[values-400:values]):
                    sitting_time.append(
                        time_original_kinem[values-400+idx])
                    break

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(
        time_original_kinem[0:2000], disp_z[0:2000], 'k-')
        ax.plot([0, 0], [0, 2], 'r-')
        
        #num_ciclos = min(len(onsets), len(offsets))
        
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_original_kinem, disp_y, 'k-')
            # Verificação básica para evitar erros
            num_ciclos = min(len(onsets), len(offsets))
            
            for i in range(num_ciclos):
                t_onset = time_original_kinem[onsets[i]]
                t_offset = time_original_kinem[offsets[i]]
                # Linha tracejada: início
                ax.axvline(t_onset, linestyle='--', color='orange',
                           label='Início da queda' if i == 0 else "")
                # Linha tracejada: fim
                ax.axvline(t_offset, linestyle='--', color='green',
                           label='Fim da queda' if i == 0 else "")
                # Faixa entre onset e offset
                ax.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                           label='Fase de queda' if i == 0 else "")

                # Se houver um próximo ciclo, pinta o intervalo entre o offset atual e o próximo onset
                if i + 1 < num_ciclos:
                    t_next_onset = time_original_kinem[onsets[i+1]]
                    ax.axvspan(t_offset, t_next_onset, color='lightblue',
                               alpha=0.3, label='Intervalo' if i == 0 else "")

                # Mínimos detectados
                for i, t in enumerate(time_original_kinem[peaks]):
                    ax.axvline(t, linestyle='--', color='blue',
                               label='Mínimo' if i == 0 else "")
    
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Amplitude")
                
                st.pyplot(fig)


        
                
