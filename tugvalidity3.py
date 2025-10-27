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
if "gyro_trig" not in st.session_state:
    st.session_state["gyro_trig"] = 0.0  # futuro: trigger em segundos, se usar vetor de tempo absoluto

# Estados para ajustes finos (acelerômetro): por ciclo (0,1,2,...)
for key in ("adj_onset", "adj_offset"):
    if key not in st.session_state:
        st.session_state[key] = {}

tab1, = st.tabs(["Angular velocity"])
with tab1:
    # Layout: coluna de controles + coluna de visualização (com subcolunas)
    c_ctrl, c_plot1 = st.columns([0.7, 2])

    with c_ctrl:
        st.subheader("Controles — Velocidade angular")

        uploaded_file_gyro = st.file_uploader(
            "Arquivo (.txt: tempo(ms); ax; ay; az) — separador ';'",
            type=["txt"],
            key="acc_file",
        )

        st.markdown("**Trigger (alinha t=0 por índice do vetor reamostrado)**")
        trigger_idx_shift = st.number_input(
            "Índice de referência (t=0)",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1,
            key="gyro_trig_idx",
        )

        # Pré-processamento
        do_detrend = True
        do_filter = True
        cutoff_gyro = 1.5  # Hz

        # Parâmetros de detecção
        height_thresh = 2.0
        distance_samples = 500  # em amostras do vetor reamostrado

        st.markdown("**Ajustes finos por ciclo (aplicados à V e AP)**")
        sel_cycle = st.number_input("Ciclo (0-index)", 0, 9999, 0, 1, key="gyro_sel_cycle")
        d_on = st.number_input(
            "Δ Tempo de A1 (s)",
            -2.0, 2.0,
            float(st.session_state["adj_onset"].get(sel_cycle, 0.0)),
            0.01,
            key="gyro_dA1",
        )
        d_off = st.number_input(
            "Δ Tempo de A2 (s)",
            -2.0, 2.0,
            float(st.session_state["adj_offset"].get(sel_cycle, 0.0)),
            0.01,
            key="gyro_dA2",
        )
        st.session_state["adj_onset"][sel_cycle] = d_on
        st.session_state["adj_offset"][sel_cycle] = d_off

        cr1, cr2 = st.columns(2)
        if cr1.button("Reset ciclo", key="btn_reset_cycle_gyro"):
            st.session_state["adj_onset"].pop(sel_cycle, None)
            st.session_state["adj_offset"].pop(sel_cycle, None)
        if cr2.button("Reset tudo", key="btn_reset_all_gyro"):
            st.session_state["adj_onset"].clear()
            st.session_state["adj_offset"].clear()

    # ===== Processamento =====
    if uploaded_file_gyro is not None:
        # 1) Lê arquivo: tempo(ms); ax; ay; az com ';'
        df = pd.read_csv(uploaded_file_gyro, sep=";", engine="python")
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
            gyro_x = low_pass_filter(gyro_x, cutoff_gyro, new_fs)
            gyro_y = low_pass_filter(gyro_y, cutoff_gyro, new_fs)
            gyro_z = low_pass_filter(gyro_z, cutoff_gyro, new_fs)

        v_gyro = np.abs(gyro_y)
        ml_gyro = np.abs(gyro_x)
        ap_gyro = np.abs(gyro_z)
        norm = np.sqrt(v_gyro**2+ml_gyro**2+ap_gyro**2)  

        # 5) Trigger por índice (alinha t=0)
        idx0 = int(clamp(trigger_idx_shift, 0, len(t_new) - 1))
        t = t_new - t_new[idx0]

        # 6) Picos: V e AP (máximos)
        indices_v, _ = find_peaks(v_gyro, height=height_thresh, distance=distance_samples)
        indices_ml, _ = find_peaks(ml_gyro, height=height_thresh, distance=distance_samples)

        # Ignora o primeiro como artefato (mantém seu padrão anterior)
        indices_v = indices_v[1:] if len(indices_v) > 1 else indices_v
        indices_ml = indices_ml[1:] if len(indices_ml) > 1 else indices_ml

        # 7) Agrupa em ciclos (pares sucessivos A1/A2)
        def build_cycles(indices):
            # pares (0,1), (2,3), (4,5), ...
            n_pairs = len(indices) // 2
            return [(indices[2*i], indices[2*i+1]) for i in range(n_pairs)]

        cycles_v = build_cycles(indices_v)
        cycles_ml = build_cycles(indices_ml)

        # usa o mínimo de ciclos disponíveis entre V e AP para tabelas comparáveis
        num_ciclos = min(len(cycles_v), len(cycles_ml))

        # 8) Aplica ajustes por ciclo (Δ A1, Δ A2) e constrói tabelas
        rows_v, rows_ap = [], []
        t_min, t_max = float(t[0]), float(t[-1])

        # Plotagem
        with c_plot1:
            # Trigger plot (janela inicial)
            st.markdown("**Trigger — Velocidade angular (t = 0)**")
            fig_trig, ax_trig = plt.subplots(figsize=(10, 2))
            nwin = min(2000, len(t))
            ax_trig.plot(t[:nwin], norm[:nwin], 'k-', label="V (|ax| ou |ay|)")
            ax_trig.axvline(0, color='r', label="t=0")
            ax_trig.set_xlabel("Tempo (s)")
            ax_trig.set_ylabel("Velocidade angular (rad/s)")
            ax_trig.legend(loc="lower left")
            st.pyplot(fig_trig)

            c1, c2 = st.columns(2)

            # ---- Coluna 1: Vertical ----
            with c1:
                fig_v, ax_v = plt.subplots(figsize=(10, 6))
                ax_v.plot(t, v_gyro, 'k-', label='Vertical')
                ax_v.axvline(0, color='r', ls='--', label="t=0")

                for i in range(num_ciclos):
                    a1_idx, a2_idx = cycles_v[i]
                    a1_t = t[a1_idx]
                    a2_t = t[a2_idx]
                    # ajustes por ciclo
                    da1 = float(st.session_state["adj_onset"].get(i, 0.0))
                    da2 = float(st.session_state["adj_offset"].get(i, 0.0))
                    a1_t_adj = clamp(a1_t + da1, t_min, t_max)
                    a2_t_adj = clamp(a2_t + da2, t_min, t_max)

                    # pontos originais
                    ax_v.plot(t[a1_idx], v_gyro[a1_idx], 'ro')
                    ax_v.plot(t[a2_idx], v_gyro[a2_idx], 'ro')
                    # linhas ajustadas
                    ax_v.axvline(a1_t_adj, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_v.axvline(a2_t_adj, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")

                    # tabela V
                    rows_v.append({
                        "ciclo": i,
                        "A1_t(s)": a1_t_adj,
                        "A1_amp(V)": float(v_gyro[a1_idx]),
                        "A2_t(s)": a2_t_adj,
                        "A2_amp(V)": float(v_gyro[a2_idx]),
                    })

                ax_v.set_xlabel("Tempo (s)")
                ax_v.set_ylabel("Velocidade angular (Vertical)")
                ax_v.legend(loc="lower left")
                st.pyplot(fig_v)

            # ---- Coluna 2: AP ----
            with c2:
                fig_ap, ax_ap = plt.subplots(figsize=(10, 6))
                ax_ap.plot(t, ml_gyro, 'k-', label='AP')
                ax_ap.axvline(0, color='r', ls='--', label="t=0")

                for i in range(num_ciclos):
                    a1_idx, a2_idx = cycles_ap[i]
                    a1_t = t[a1_idx]
                    a2_t = t[a2_idx]
                    # usa os mesmos ajustes por ciclo (Δ A1/A2)
                    da1 = float(st.session_state["adj_onset"].get(i, 0.0))
                    da2 = float(st.session_state["adj_offset"].get(i, 0.0))
                    a1_t_adj = clamp(a1_t + da1, t_min, t_max)
                    a2_t_adj = clamp(a2_t + da2, t_min, t_max)

                    # pontos originais
                    ax_ap.plot(t[a1_idx], ml_gyro[a1_idx], 'ro')
                    ax_ap.plot(t[a2_idx], ml_gyro[a2_idx], 'ro')
                    # linhas ajustadas
                    ax_ap.axvline(a1_t_adj, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_ap.axvline(a2_t_adj, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")

                    # tabela AP
                    rows_ap.append({
                        "ciclo": i,
                        "A1_t(s)": a1_t_adj,
                        "A1_amp(AP)": float(ml_gyro[a1_idx]),
                        "A2_t(s)": a2_t_adj,
                        "A2_amp(AP)": float(ml_gyro[a2_idx]),
                    })

                ax_ap.set_xlabel("Tempo (s)")
                ax_ap.set_ylabel("Velocidade angular (ML)")
                ax_ap.legend(loc="lower left")
                st.pyplot(fig_ap)

            # ===== Tabelas =====
            st.subheader("Tempos por ciclo — Aceleração Vertical (V)")
            df_tempos_v = pd.DataFrame(rows_v)
            st.dataframe(df_tempos_v, width='stretch')

            st.subheader("Tempos por ciclo — Aceleração Antero-Posterior (AP)")
            df_tempos_ap = pd.DataFrame(rows_ap)
            st.dataframe(df_tempos_ap, width='stretch')
            df_join = df_tempos_v.merge(df_tempos_ap, on="ciclo", suffixes=("_V", "_AP"))

            st.download_button(
                "Baixar CSV (Acc)",
                df_join.to_csv(index=False).encode("utf-8"),
                file_name="tempo_ciclos_gyro.csv",
                mime="text/csv",
                key="btn_export_merged"
            )
