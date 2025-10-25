import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks
import matplotlib.pyplot as plt

# ------------------------
# Utilitários
# ------------------------
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ------------------------
# App
# ------------------------
st.set_page_config(layout="wide")
st.title("Análise de Dados: Interpolação, Detrend e Filtro Passa-Baixa")

tab1, tab2, tab3 = st.tabs(['Kinematics', 'Acceleration', 'Angular velocity'])

with tab1:
    uploaded_file_kinem = st.file_uploader(
        "Escolha um arquivo da cinemática (.csv com 3 colunas: X, Y, Z em mm)", type=["csv"]
    )

    if uploaded_file_kinem:
        # Leitura e pré-processamento
        df = pd.read_csv(uploaded_file_kinem, sep=",", engine='python')

        # Garante pelo menos 3 colunas numéricas
        if df.shape[1] < 3:
            st.error("O arquivo precisa ter ao menos 3 colunas numéricas (X, Y, Z).")
            st.stop()

        # Converte para metros como no seu código original (/1000)
        disp_x = df.iloc[:, 0].astype(float).values / 1000.0
        disp_y = df.iloc[:, 1].astype(float).values / 1000.0
        disp_z = df.iloc[:, 2].astype(float).values / 1000.0

        # Parâmetros de amostragem (ajuste se necessário)
        original_fs_kinem = 100.0  # Hz
        time_original_kinem = np.arange(len(disp_y)) / original_fs_kinem

        # Controles de pré-processamento
        with st.expander("Pré-processamento (opcional)"):
            do_detrend = st.checkbox("Aplicar detrend (remove tendência)", value=False)
            do_filter = st.checkbox("Aplicar filtro passa-baixa", value=False)
            cutoff_kinem = st.number_input("Cutoff do filtro (Hz)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

            if do_detrend:
                disp_x = detrend(disp_x)
                disp_y = detrend(disp_y)
                disp_z = detrend(disp_z)

            if do_filter:
                disp_x = low_pass_filter(disp_x, cutoff_kinem, fs=original_fs_kinem, order=4)
                disp_y = low_pass_filter(disp_y, cutoff_kinem, fs=original_fs_kinem, order=4)
                disp_z = low_pass_filter(disp_z, cutoff_kinem, fs=original_fs_kinem, order=4)

        # Trigger (garante índice válido)
        valor = st.slider(
            "Ajustar o trigger da cinemática (alinha t=0)",
            min_value=0,
            max_value=len(time_original_kinem) - 1,
            value=0
        )
        time_original_kinem = time_original_kinem - time_original_kinem[valor]

        # Detecção de picos (mínimos de disp_y)
        with st.expander("Parâmetros de detecção de eventos"):
            prominence = st.number_input("Prominence mínima", min_value=0.0, value=0.0, step=0.01)
            min_distance_samples = st.number_input("Distância mínima entre picos (amostras)", min_value=1, value=10, step=1)

        peaks, properties = find_peaks(-1 * disp_y, prominence=prominence if prominence > 0 else None,
                                       distance=int(min_distance_samples))

        # Busca de onsets e offsets associada a cada pico
        onsets = []
        offsets = []
        for peak in peaks:
            # Busca para trás: início da queda (critério simples, ajuste se desejar)
            found_onset = None
            for j in range(peak, 1, -1):
                if disp_y[j] > disp_y[j - 1]:
                    found_onset = j
                    break
            if found_onset is not None:
                onsets.append(found_onset)

            # Busca para frente: fim da queda
            found_offset = None
            for j in range(peak, len(disp_y) - 1):
                if disp_y[j] > disp_y[j + 1]:
                    found_offset = j
                    break
            if found_offset is not None:
                offsets.append(found_offset)

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

        num_ciclos = min(len(onsets), len(offsets))

        # --------- Plot 1: disp_z (janela inicial) ----------
        fig1, ax1 = plt.subplots(figsize=(10, 2))
        end_idx = min(2000, len(time_original_kinem))
        ax1.plot(time_original_kinem[0:end_idx], disp_z[0:end_idx], 'k-', label="disp_z")
        ax1.axvline(0, color='r', label="t=0")
        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylabel("Amplitude (m)")
        ax1.legend()
        st.pyplot(fig1)

        # --------- Plot 2: disp_y com marcações ----------
        col1, col2 = st.columns(2)
        with col1:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(time_original_kinem, disp_y, 'k-', label="Deslocamento AP")

            for i in range(num_ciclos):
                t_onset = time_original_kinem[onsets[i]]
                t_offset = time_original_kinem[offsets[i]]

                ax2.axvline(t_onset, linestyle='--', color='orange',
                            label='Início' if i == 0 else "")
                ax2.axvline(t_offset, linestyle='--', color='green',
                            label='Fim' if i == 0 else "")
                ax2.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                            label='Teste' if i == 0 else "")

                if i + 1 < num_ciclos:
                    t_next_onset = time_original_kinem[onsets[i + 1]]
                    ax2.axvspan(t_offset, t_next_onset, color='lightblue', alpha=0.3,
                                label='Intervalo entre testes' if i == 0 else "")

            # Mínimos detectados (fora do loop e sem sobrescrever i)
            for k, t in enumerate(time_original_kinem[peaks]):
                ax2.axvline(t, linestyle='--', color='blue',
                            label='3 m' if k == 0 else "")

            ax2.set_xlabel("Tempo (s)")
            ax2.set_ylabel("Amplitude (m)")
            ax2.legend()
            st.pyplot(fig2)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                time_original_kinem, disp_z, 'k-')
            ax.plot([0, 0], [0, 2], 'r-')

            # Verificação básica para evitar erros

            num_ciclos = min(
                len(onsets), len(offsets))

            for i in range(num_ciclos):
                t_onset = time_original_kinem[onsets[i]]
                t_offset = time_original_kinem[offsets[i]]

                #    # Linha tracejada: início
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
