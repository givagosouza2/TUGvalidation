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


# =========================
# App
# =========================
st.set_page_config(layout="wide")
st.title("Análise de dados da medidas acelerométricas do TUG")
st.expander("Esta rotina foi criada para que se importe dados de medidas acelerométricas, aplique o trigger do registro e verifique se as marcações automatizadas correspondentes aos eventos biomecânicos realizados durante o TUG. Caso seja necessário, o usuário deverá fazer ajustes nestas marcações.")
# defaults de estado que usamos em UI/plot (evita NameError)
defaults = {
    "acc_trig": 0.0,  
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Estados para ajustes finos (cinemática)
for key in ("adj_onset", "adj_offset", "adj_stand", "adj_sit", "adj_peaks"):
    if key not in st.session_state:
        st.session_state[key] = {}

# Estados para ajustes finos (aceleração)
for key in ("adj_onset_acc", "adj_offset_acc", "adj_peak_acc"):
    if key not in st.session_state:
        st.session_state[key] = {}
        
tab1, = st.tabs(["Acceleration"])
with tab1:
    # Layout: coluna de controles + coluna de visualização (com subcolunas)
    c_ctrl, c_plot1 = st.columns([0.7, 2])
    with c_ctrl:
        st.subheader("Controles — Aceleração")

        uploaded_file_acc = st.file_uploader(
            "Arquivo (.txt: X, Y, Z em m/s2)", type=["txt"], key="kin_file"
        )

        st.markdown("**Trigger (alinha t=0)**")
        trigger_idx_shift = st.number_input("Índice de referência", 0, 100000, 0, 1, key="kin_trig")

        # Defaults solicitados: detrend desmarcado, filtro marcado
        do_detrend = True
        do_filter = True

        # Parâmetros fixos solicitados
        cutoff_acc = 2.0
        min_distance_samples = 200

        st.markdown("**Ajustes finos**")
        sel_cycle = st.number_input("Ciclo (0-index)", 0, 9999, 0, 1, key="kin_sel_cycle")
        d_on = st.number_input("Δ Tempo de A1 v (s)", -2.0, 2.0, float(st.session_state["adj_onset"].get(sel_cycle, 0.0)), 0.01, key="kin_don")
        d_off = st.number_input("Δ Tempo de A2 v (s)", -2.0, 2.0, float(st.session_state["adj_offset"].get(sel_cycle, 0.0)), 0.01, key="kin_doff")
        
        st.session_state["adj_onset"][sel_cycle] = d_on
        st.session_state["adj_offset"][sel_cycle] = d_off
        
        cr1, cr2 = st.columns(2)
        if cr1.button("Reset ciclo", key="btn_reset_cycle_kin"):
            st.session_state["adj_onset"].pop(sel_cycle, None)
            st.session_state["adj_offset"].pop(sel_cycle, None)
            
        if cr2.button("Reset tudo", key="btn_reset_all_kin"):
            for k in ("adj_onset","adj_offset"):
                st.session_state[k].clear()
        
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
    
        indices_v, propriedades = find_peaks(v_acc, height = 2, distance = 500)
        
        indices_ap, propriedades = find_peaks(ap_acc, height = 2, distance = 500)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(time_interpolated[0:2000], v_acc[0:2000], 'k-')   
        ax.plot([0, 0], [0, 30], 'r--')
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_interpolated, v_acc, 'k-')
            for i in range(1,9,1):
                ax.plot(time_interpolated[indices_v[i]], v_acc[indices_v[i]], 'ro')
            ax.plot([0, 0], [0, 30], 'r--')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração vertical")
            st.pyplot(fig)
        with col2:    
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_interpolated, ap_acc, 'k-')
            for i in range(1,9,1):
                ax.plot(time_interpolated[indices_ap[i]], ap_acc[indices_ap[i]], 'ro')
            ax.plot([0, 0], [0, 30], 'r--')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração AP")
            st.pyplot(fig)
        
        rows_v = []
        i = 1
        for valor in range(1,9,2):
            rows_v.append({"ciclo": i, "Latência A1": time_interpolated[indices_v[valor]], "Amplitude A1 V": v_acc[indices_v[valor]],"Latência A2": time_interpolated[indices_v[valor+1]],"Amplitude A2 V": v_acc[indices_v[valor+1]]})
            i = i + 1
    
        df_tempos_v = pd.DataFrame(rows_v)
        st.subheader("Tempos por ciclo — Aceleração vertical")
        st.dataframe(df_tempos_v, width='stretch')
    
        rows_ap = []
        i = 1
        for valor in range(1,9,2):
            rows_ap.append({"ciclo": i, "Latência A1 AP": time_interpolated[indices_ap[valor]], "Amplitude A1 AP": ap_acc[indices_ap[valor]],"Latência A2 AP": time_interpolated[indices_ap[valor+1]],"Amplitude A2 AP": ap_acc[indices_ap[valor+1]]})
            i = i + 1
    
        df_tempos_ap = pd.DataFrame(rows_ap)
        st.subheader("Tempos por ciclo — Aceleração AP")
        st.dataframe(df_tempos_ap, width='stretch')
    
    
    
    
    
    
