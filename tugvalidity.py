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

# ------------------------
# Sidebar (Cinemática)
# ------------------------
st.sidebar.header("Configurações — Cinemática")

# Pré-processamento (cinemática)
st.sidebar.subheader("Pré-processamento (cinemática)")
do_detrend = st.sidebar.checkbox("Aplicar detrend", value=False)
do_filter = st.sidebar.checkbox("Aplicar filtro passa-baixa", value=True)
cutoff_kinem = st.sidebar.number_input(
    "Cutoff do filtro (Hz)", min_value=0.1, max_value=20.0, value=2.0, step=0.1
)

# Parâmetros de detecção (cinemática)
st.sidebar.subheader("Detecção de eventos (cinemática)")
prominence = st.sidebar.number_input(
    "Prominence mínima", min_value=0.0, value=2.5, step=0.1
)
min_distance_samples = st.sidebar.number_input(
    "Distância mínima entre picos (amostras)", min_value=1, value=200, step=1
)

# Upload do arquivo (cinemática)
uploaded_file_kinem = st.sidebar.file_uploader(
    "Arquivo de cinemática (.csv: X, Y, Z em mm)", type=["csv"]
)

# Botão para liberar abas extras
if "show_dyn_tabs" not in st.session_state:
    st.session_state.show_dyn_tabs = False

st.sidebar.markdown("---")
if st.sidebar.button("Mostrar abas de Acceleration e Angular velocity"):
    st.session_state.show_dyn_tabs = True

# Monta as abas dinamicamente
tab_labels = ["Kinematics"]
if st.session_state.show_dyn_tabs:
    tab_labels += ["Acceleration", "Angular velocity"]

tabs = st.tabs(tab_labels)
tab_map = {label: tabs[i] for i, label in enumerate(tab_labels)}

# ------------------------
# Aba Kinematics
# ------------------------
with tab_map["Kinematics"]:
    if uploaded_file_kinem:
        # Leitura e pré-processamento
        df = pd.read_csv(uploaded_file_kinem, sep=",", engine='python')

        if df.shape[1] < 3:
            st.error("O arquivo precisa ter ao menos 3 colunas numéricas (X, Y, Z).")
            st.stop()

        try:
            disp_x = df.iloc[:, 0].astype(float).values / 1000.0
            disp_y = df.iloc[:, 1].astype(float).values / 1000.0
            disp_z = df.iloc[:, 2].astype(float).values / 1000.0
        except Exception:
            st.error("As três primeiras colunas devem ser numéricas.")
            st.stop()

        # Amostragem
        original_fs_kinem = 100.0  # Hz
        time_original_kinem = np.arange(len(disp_y)) / original_fs_kinem

        # Aplicar pré-processamento
        if do_detrend:
            disp_x = detrend(disp_x)
            disp_y = detrend(disp_y)
            disp_z = detrend(disp_z)

        if do_filter:
            disp_x = low_pass_filter(disp_x, cutoff_kinem, fs=original_fs_kinem)
            disp_y = low_pass_filter(disp_y, cutoff_kinem, fs=original_fs_kinem)
            disp_z = low_pass_filter(disp_z, cutoff_kinem, fs=original_fs_kinem)

        # Trigger na barra lateral
        valor = st.sidebar.slider(
            "Trigger (alinha t=0) — cinemática",
            min_value=0,
            max_value=len(time_original_kinem) - 1,
            value=0
        )
        time_original_kinem = time_original_kinem - time_original_kinem[valor]

        # Detecção de picos (mínimos de disp_y)
        peak_kwargs = {}
        if prominence > 0:
            peak_kwargs["prominence"] = float(prominence)
        if min_distance_samples > 1:
            peak_kwargs["distance"] = int(min_distance_samples)

        peaks, _ = find_peaks(-disp_y, **peak_kwargs)

        # Busca de eventos (onsets/offsets)
        onsets, offsets = [], []
        for peak in peaks:
            # Busca para trás: onset
            for j in range(peak, 1, -1):
                if disp_y[j] > disp_y[j - 1]:
                    onsets.append(j)
                    break
            # Busca para frente: offset
            for j in range(peak, len(disp_y) - 1):
                if disp_y[j] > disp_y[j + 1]:
                    offsets.append(j)
                    break

        num_ciclos = min(len(onsets), len(offsets))

        # standing e sitting (protegendo janelas)
        standing_time, sitting_time = [], []
        for i in range(num_ciclos):
            v = onsets[i]
            a, b = v, min(v + 200, len(disp_z))
            if b > a:
                standing_time.append(time_original_kinem[a + np.argmax(disp_z[a:b])])

            v = offsets[i]
            a, b = max(0, v - 400), v
            if b > a:
                sitting_time.append(time_original_kinem[a + np.argmax(disp_z[a:b])])

        # ---------------- PLOTS ----------------
        fig1, ax1 = plt.subplots(figsize=(10, 2))
        ax1.plot(time_original_kinem[:2000], disp_z[:2000], 'k-', label="disp_z")
        ax1.axvline(0, color='r', label="t=0")
        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylabel("Amplitude (m)")
        ax1.legend(loc="lower left")
        st.pyplot(fig1)

        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(time_original_kinem, disp_y, 'k-', label="Desloc. AP")

            for i in range(num_ciclos):
                t_onset = time_original_kinem[onsets[i]]
                t_offset = time_original_kinem[offsets[i]]

                ax2.axvline(t_onset, linestyle='--', color='orange',
                            label='Início' if i == 0 else "")
                ax2.axvline(t_offset, linestyle='--', color='green',
                            label='Fim' if i == 0 else "")
                ax2.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                            label='Teste' if i == 0 else "")

                if i < len(standing_time):
                    ax2.axvline(standing_time[i], linestyle='--', color='red',
                                label='Pico em pé' if i == 0 else "")
                if i < len(sitting_time):
                    ax2.axvline(sitting_time[i], linestyle='--', color='black',
                                label='Pico para sentar' if i == 0 else "")

            for k, t in enumerate(time_original_kinem[peaks]):
                ax2.axvline(t, linestyle='--', color='blue',
                            label='Mínimos' if k == 0 else "")

            ax2.set_xlabel("Tempo (s)")
            ax2.set_ylabel("Amplitude (m)")
            ax2.legend(loc="lower left")
            st.pyplot(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(time_original_kinem, disp_z, 'k-', label="Desloc. vertical")

            for i in range(num_ciclos):
                t_onset = time_original_kinem[onsets[i]]
                t_offset = time_original_kinem[offsets[i]]

                ax3.axvline(t_onset, linestyle='--', color='orange',
                            label='Início' if i == 0 else "")
                ax3.axvline(t_offset, linestyle='--', color='green',
                            label='Fim' if i == 0 else "")
                ax3.axvspan(t_onset, t_offset, color='gray', alpha=0.3,
                            label='Teste' if i == 0 else "")

                if i < len(standing_time):
                    ax3.axvline(standing_time[i], linestyle='--', color='red',
                                label='Pico em pé' if i == 0 else "")
                if i < len(sitting_time):
                    ax3.axvline(sitting_time[i], linestyle='--', color='black',
                                label='Pico para sentar' if i == 0 else "")

            for k, t in enumerate(time_original_kinem[peaks]):
                ax3.axvline(t, linestyle='--', color='blue',
                            label='Mínimos' if k == 0 else "")

            ax3.set_xlabel("Tempo (s)")
            ax3.set_ylabel("Amplitude (m)")
            ax3.legend(loc="lower left")
            st.pyplot(fig3)

        # Resumo quantitativo
        st.info(
            f"**Resumo**  \n"
            f"• Picos detectados: {len(peaks)}  \n"
            f"• Onsets: {len(onsets)}  \n"
            f"• Offsets: {len(offsets)}  \n"
            f"• Ciclos válidos: {num_ciclos}"
        )
    else:
        st.warning("Faça upload de um arquivo para ver os resultados de cinemática.")

# ------------------------
# Abas extras (aparecem após o clique)
# ------------------------
if st.session_state.show_dyn_tabs:
    with tab_map["Acceleration"]:
        st.write("Conteúdo de Acceleration (a definir).")
    with tab_map["Angular velocity"]:
        st.write("Conteúdo de Angular velocity (a definir).")
