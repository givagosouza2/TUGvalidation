import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =========================
# Utilitários
# =========================
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================
# App
# =========================
st.set_page_config(layout="wide")
st.title("Análise de dados da medidas acelerométricas do TUG")

st.info(
    "Esta rotina importa dados acelerométricos, aplica o trigger e exibe as marcações "
    "automáticas dos eventos. Se necessário, ajuste os tempos por ciclo (Δ A1 / Δ A2)."
)

# defaults de estado que usamos em UI/plot (evita NameError)
if "acc_trig" not in st.session_state:
    st.session_state["acc_trig"] = 0.0  # futuro: trigger em segundos, se usar vetor de tempo absoluto

# Estados para ajustes finos (acelerômetro): por ciclo (0,1,2,...)
for key in ("Onset","G0 peak", "G2 peak", "G1 peak", "G3 peak","Offset"):
    if key not in st.session_state:
        st.session_state[key] = {}

tab1, = st.tabs(["Gyroscope"])
with tab1:
    # Layout: coluna de controles + coluna de visualização (com subcolunas)
    c_ctrl, c_plot1 = st.columns([0.7, 2])

    with c_ctrl:
        st.subheader("Controles — Giroscópio")

        uploaded_file_acc = st.file_uploader(
            "Arquivo (.txt: tempo(ms); gx; gy; gz) — separador ';'",
            type=["txt"],
            key="acc_file",
        )

        st.markdown("**Trigger (alinha t=0 por índice do vetor reamostrado)**")
        trigger_idx_shift = st.number_input(
            "Índice de referência (t=0)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0,
            key="acc_trig_idx",
        )

        # Pré-processamento
        do_detrend = True
        do_filter = True
        cutoff_acc = 1.5  # Hz

        # Parâmetros de detecção
        height_thresh_ap = 3
        height_thresh = 2
        distance_samples = 500  # em amostras do vetor reamostrado

        st.markdown("**Ajustes finos por ciclo (aplicados à V e AP)**")
        sel_cycle = st.number_input("Ciclo (0-index)", 0, 9999, 0, 1, key="acc_sel_cycle")
        d_0 = st.number_input(
            "Onset (s)",
            0.0, 100.0,
            float(st.session_state["Onset"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_donset",
        )
        
        d_1 = st.number_input(
            "peak G0 (s)",
            0.0, 100.0,
            float(st.session_state["G0 peak"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_dA1",
        )
        d_2 = st.number_input(
            "peak G1 (s)",
            0.0, 100.0,
            float(st.session_state["G1 peak"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_dA2",
        )
        d_3 = st.number_input(
            "peak G2 (s)",
            0.0, 100.0,
            float(st.session_state["G2 peak"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_dA1ap",
        )
        #d_4 = st.number_input(
            "peak G3 (s)",
            0.0, 100.0,
            float(st.session_state["G3 peak"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_dA2ap",
        )

        d_5 = st.number_input(
            "Offset (s)",
            0.0, 100.0,
            float(st.session_state["Offset"].get(sel_cycle, 0.0)),
            0.01,
            key="acc_dOffset",
        )
        st.session_state["Onset"][sel_cycle] = d_0
        st.session_state["G0 peak"][sel_cycle] = d_1
        st.session_state["G1 peak"][sel_cycle] = d_2
        st.session_state["G2 peak"][sel_cycle] = d_3
        st.session_state["G3 peak"][sel_cycle] = d_4
        st.session_state["Offset"][sel_cycle] = d_5

        cr1, cr2 = st.columns(2)
        if cr1.button("Reset ciclo", key="btn_reset_cycle_acc"):
            st.session_state["Onset"].pop(sel_cycle, None)
            st.session_state["G0 peak"].pop(sel_cycle, None)
            st.session_state["G1 peak"].pop(sel_cycle, None)
            st.session_state["G2 peak"].pop(sel_cycle, None)
            st.session_state["G3 peak"].pop(sel_cycle, None)
            st.session_state["Offset"].pop(sel_cycle, None)
        if cr2.button("Reset tudo", key="btn_reset_all_acc"):
            st.session_state["Onset"].clear()
            st.session_state["G0 peak"].clear()
            st.session_state["G1 peak"].clear()
            st.session_state["G2 peak"].clear()
            st.session_state["G3 peak"].clear()
            st.session_state["Offset"].clear()

    # ===== Processamento =====
    if uploaded_file_acc is not None:
        # 1) Lê arquivo: tempo(ms); ax; ay; az com ';'
        df = pd.read_csv(uploaded_file_acc, sep=";", engine="python")
        if df.shape[1] < 4:
            st.error("O arquivo deve ter ao menos 4 colunas: tempo(ms); ax; ay; az.")
            st.stop()

        try:
            tempo_ms = df.iloc[:, 0].astype(float).values
            gyro_x_raw = df.iloc[:, 1].astype(float).values
            gyro_y_raw = df.iloc[:, 2].astype(float).values
            gyro_z_raw = df.iloc[:, 3].astype(float).values
        except Exception:
            st.error("As quatro primeiras colunas precisam ser numéricas.")
            st.stop()

        # 2) Interpola/reamostra para 100 Hz
        new_fs = 100.0  # Hz
        # tempo em segundos (converte ms -> s)
        tempo_s = tempo_ms / 1000.0

        # garante tempo crescente e sem duplicatas
        order = np.argsort(tempo_s)
        tempo_s, gyro_x_raw, gyro_y_raw, gyro_z_raw = (
            tempo_s[order], gyro_x_raw[order], gyro_y_raw[order], gyro_z_raw[order]
        )
        uniq = np.diff(tempo_s, prepend=tempo_s[0] - 1.0) > 0
        tempo_s, gyro_x_raw, gyro_y_raw, gyro_z_raw = (
            tempo_s[uniq], gyro_x_raw[uniq], gyro_y_raw[uniq], gyro_z_raw[uniq]
        )
        if len(tempo_s) < 2:
            st.error("Tempo insuficiente após ordenar/remover duplicatas.")
            st.stop()

        t_start, t_end = float(tempo_s[0]), float(tempo_s[-1])
        t_new = np.arange(t_start, t_end, 1.0/new_fs)
        t = t_new - trigger_idx_shift

        fx = interp1d(tempo_s, gyro_x_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        fy = interp1d(tempo_s, gyro_y_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        fz = interp1d(tempo_s, gyro_z_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        gyro_x = fx(t_new)
        gyro_y = fy(t_new)
        gyro_z = fz(t_new)

        # 3) Pré-processamento
        if do_detrend:
            gyro_x = detrend(gyro_x)
            gyro_y = detrend(gyro_y)
            gyro_z = detrend(gyro_z)

        if do_filter:
            gyro_x = low_pass_filter(gyro_x, cutoff_acc, new_fs)
            gyro_y = low_pass_filter(gyro_y, cutoff_acc, new_fs)
            gyro_z = low_pass_filter(gyro_z, cutoff_acc, new_fs)

        
        v_gyro = np.abs(gyro_y)
        ml_gyro = np.abs(gyro_x)
        ap_gyro = np.abs(gyro_z)

        norm = np.sqrt(v_gyro**2+ml_gyro**2+ap_gyro**2)

        # Plotagem
        with c_plot1:
            # Trigger plot (janela inicial)
            st.markdown("**Trigger — Aceleração (t = 0)**")
            fig_trig, ax_trig = plt.subplots(figsize=(10, 2))
            nwin = min(2000, len(t))
            ax_trig.plot(t[:nwin], v_gyro[:nwin], 'k-', label="V (|ax| ou |ay|)")
            ax_trig.axvline(0, color='r', label="t=0")
            ax_trig.set_xlabel("Tempo (s)")
            ax_trig.set_ylabel("Velocidade angular V")
            ax_trig.legend(loc="lower left")
            st.pyplot(fig_trig)

            c1, c2 = st.columns(2)
            rows_v = []
            rows_ap = []            
            # ---- Coluna 1: Vertical ----
            with c1:
                fig_v, ax_v = plt.subplots(figsize=(10, 6))
                ax_v.plot(t, norm, 'k-', label='Vertical')
                ax_v.axvline(0, color='r', ls='--', label="t=0")
                num_ciclos = 4

                for i in range(num_ciclos):
                    da0 = float(st.session_state["Onset"].get(i, 0.0))
                    da1 = float(st.session_state["G0 peak"].get(i, 0.0))
                    da2 = float(st.session_state["G3 peak"].get(i, 0.0))
                    da5 = float(st.session_state["Offset"].get(i, 0.0))
                    da3 = float(st.session_state["G1 peak"].get(i, 0.0))
                    da4 = float(st.session_state["G2 peak"].get(i, 0.0))

                    for index, valor in enumerate(norm):
                        if t[index] > da0:
                            G0onset = norm[index-1]
                            break
    
                    for index, valor in enumerate(norm):
                        if t[index] > da1:
                            G0peak = norm[index-1]
                            break
                            
                    for index, valor in enumerate(norm):
                        if t[index] > da3:
                            G1peak = norm[index-1]
                            break
    
                    for index, valor in enumerate(norm):
                        if t[index] > da4:
                            G2peak = norm[index-1]
                            break
    
                    for index, valor in enumerate(norm):
                        if t[index] > da2:
                            G3peak = norm[index-1]
                            break
    
                    for index, valor in enumerate(norm):
                        if t[index] > da5:
                            G0offset = norm[index-1]
                            break
                            
    
                    # pontos originais
                    ax_v.plot(da0, G0onset, 'ro')
                    ax_v.plot(da1, G0peak, 'ro')
                    ax_v.plot(da3, G1peak, 'ro')
                    ax_v.plot(da4, G2peak, 'ro')
                    #ax_v.plot(da2, G3peak, 'ro')
                    ax_v.plot(da5, G0offset, 'ro')
                    # linhas ajustadas
                    ax_v.axvline(da0, color='blue', ls='--', label='A0 (aj)' if i == 0 else "")
                    ax_v.axvline(da1, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_v.axvline(da2, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")
                    ax_v.axvline(da3, color='red', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_v.axvline(da4, color='black', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_v.axvline(da5, color='blue', ls='--', label='A1 (aj)' if i == 0 else "")
    
                    # tabela V
                    
                    rows_v.append({
                        "ciclo": i,
                        "Onset (s)": da0,
                        "G0_t(s)": da1,
                        "G0_amp(V)": float(G0peak),
                        "G1_t(s)": da3,
                        "G1_amp(AP)": float(G1peak),
                        "G2_t(s)": da4,
                        "G2_amp(AP)": float(G2peak),
                        "G3_t(s)": da2,
                        "G3_amp(V)": float(float(G3peak)),
                        "Offset (s)": da5,
                    })
    
                ax_v.set_xlabel("Tempo (s)")
                ax_v.set_ylabel("Velocidade angular (ML)")
                ax_v.legend(loc="lower left")
                st.pyplot(fig_v)

            # ---- Coluna 2: AP ----
            with c2:
                fig_ap, ax_ap = plt.subplots(figsize=(10, 6))
                ax_ap.plot(t, norm, 'k-', label='AP')
                ax_ap.axvline(0, color='r', ls='--', label="t=0")

                for i in range(num_ciclos):
                    da3 = float(st.session_state["G1 peak"].get(i, 0.0))
                    da4 = float(st.session_state["G2 peak"].get(i, 0.0))

                    for index, valor in enumerate(ap_gyro):
                        if t[index] > da3:
                            G1peak = ap_gyro[index-1]
                            break

                    for index, valor in enumerate(ap_gyro):
                        if t[index] > da4:
                            G2peak = ap_gyro[index-1]
                            break

                    # pontos originais
                    ax_ap.plot(da3, G1peak, 'ro')
                    ax_ap.plot(da4, G2peak, 'ro')
                    # linhas ajustadas
                    ax_ap.axvline(da3, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_ap.axvline(da4, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")

                    # tabela AP
                   
                    rows_ap.append({
                        "ciclo": i,
                        "G1_t(s)": da3,
                        "G1_amp(AP)": float(G1peak),
                        "G2_t(s)": da4,
                        "G2_amp(AP)": float(G2peak),
                    })

                ax_ap.set_xlabel("Tempo (s)")
                ax_ap.set_ylabel("Velocidade angular (V)")
                ax_ap.legend(loc="lower left")
                st.pyplot(fig_ap)

            # ===== Tabelas =====
            st.subheader("Tempos por ciclo — Giroscópio (ML)")
            df_tempos_v = pd.DataFrame(rows_v)
            st.dataframe(df_tempos_v, width='stretch')

            st.subheader("Tempos por ciclo — Giroscópio (V)")
            df_tempos_ap = pd.DataFrame(rows_ap)
            st.dataframe(df_tempos_ap, width='stretch')
            df_join = df_tempos_v.merge(df_tempos_ap, on="ciclo", suffixes=("_V", "_AP"))

            st.download_button(
                "Baixar CSV (Gyro)",
                df_join.to_csv(index=False).encode("utf-8"),
                file_name="tempo_ciclos_gyro.csv",
                mime="text/csv",
                key="btn_export_merged"
            )
