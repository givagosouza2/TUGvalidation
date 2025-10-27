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
st.title("Análise de dados de velocidade angular do TUG")

st.info(
    "Esta rotina importa dados de giroscópio, alinha o trigger e marca automaticamente "
    "eventos por picos. Se necessário, ajuste os tempos por ciclo (Δ A1 / Δ A2)."
)

# Estados para ajustes finos (por ciclo 0,1,2,...)
for key in ("adj_onset", "adj_offset"):
    if key not in st.session_state:
        st.session_state[key] = {}
# Estado do índice do trigger (t=0) na malha reamostrada
if "gyro_trig_idx" not in st.session_state:
    st.session_state["gyro_trig_idx"] = 0

tab1, = st.tabs(["Angular velocity"])
with tab1:
    c_ctrl, c_plot1 = st.columns([0.7, 2])

    with c_ctrl:
        st.subheader("Controles — Velocidade angular")

        uploaded_file_gyro = st.file_uploader(
            "Arquivo (.txt: tempo(ms); gx; gy; gz) — separador ';'",
            type=["txt"],
            key="gyro_file",
        )

        st.markdown("**Trigger (alinha t=0 por índice do vetor reamostrado)**")
        st.number_input(
            "Índice de referência (t=0)",
            min_value=0,
            max_value=1_000_000,
            value=int(st.session_state.get("gyro_trig_idx", 0)),
            step=1,
            key="gyro_trig_idx",
        )

        # Pré-processamento (fixos; expose se quiser na UI)
        do_detrend = True
        do_filter = True
        cutoff_gyro = 1.5  # Hz (Nyquist=50 Hz com fs=100 → OK)

        # Parâmetros de detecção
        height_thresh = 1
        distance_samples = 75  # amostras na malha reamostrada (100 Hz → 0.5 s)

        st.markdown("**Ajustes finos por ciclo (aplicados a V e ML)**")
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

        rc1, rc2 = st.columns(2)
        if rc1.button("Reset ciclo", key="btn_reset_cycle_gyro"):
            st.session_state["adj_onset"].pop(sel_cycle, None)
            st.session_state["adj_offset"].pop(sel_cycle, None)
        if rc2.button("Reset tudo", key="btn_reset_all_gyro"):
            st.session_state["adj_onset"].clear()
            st.session_state["adj_offset"].clear()

    # =========================
    # Processamento
    # =========================
    if uploaded_file_gyro is not None:
        # 1) Leitura: tempo(ms); gx; gy; gz (separador ';')
        df = pd.read_csv(uploaded_file_gyro, sep=";", engine="python")
        if df.shape[1] < 4:
            st.error("O arquivo deve ter ao menos 4 colunas: tempo(ms); gx; gy; gz.")
            st.stop()

        try:
            tempo_ms = df.iloc[:, 0].astype(float).values
            gyro_x_raw = df.iloc[:, 1].astype(float).values  # gx
            gyro_y_raw = df.iloc[:, 2].astype(float).values  # gy
            gyro_z_raw = df.iloc[:, 3].astype(float).values  # gz
        except Exception:
            st.error("As quatro primeiras colunas precisam ser numéricas.")
            st.stop()

        # 2) Reamostragem para 100 Hz
        new_fs = 100.0  # Hz
        tempo_s = tempo_ms / 1000.0

        # ordem crescente e sem duplicatas
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
        if t_end <= t_start:
            st.error("A coluna de tempo deve ser estritamente crescente.")
            st.stop()
        t_new = np.arange(t_start, t_end, 1.0/new_fs)

        fx = interp1d(tempo_s, gyro_x_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        fy = interp1d(tempo_s, gyro_y_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        fz = interp1d(tempo_s, gyro_z_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
        gyro_x = fx(t_new)
        gyro_y = fy(t_new)
        gyro_z = fz(t_new)

        # 3) Pré-processamento
        if do_detrend:
            gyro_x = detrend(gyro_x); gyro_y = detrend(gyro_y); gyro_z = detrend(gyro_z)
        if do_filter:
            if cutoff_gyro >= new_fs/2:
                st.error("Cutoff do filtro deve ser menor que a frequência de Nyquist (fs/2).")
                st.stop()
            gyro_x = low_pass_filter(gyro_x, cutoff_gyro, new_fs)
            gyro_y = low_pass_filter(gyro_y, cutoff_gyro, new_fs)
            gyro_z = low_pass_filter(gyro_z, cutoff_gyro, new_fs)

        # 4) Componentes e norma
        v_gyro  = np.abs(gyro_y)  # Vertical (ex.: eixo Y)
        ml_gyro = np.abs(gyro_x)  # Médio-lateral (ex.: eixo X)
        ap_gyro = np.abs(gyro_z)  # Antero-posterior (ex.: eixo Z) [não usado nos gráficos]
        norm = np.sqrt(v_gyro**2 + ml_gyro**2 + ap_gyro**2)

        # 5) Trigger por índice
        idx0 = int(clamp(st.session_state["gyro_trig_idx"], 0, len(t_new) - 1))
        t = t_new - t_new[idx0]
        t_min, t_max = float(t[0]), float(t[-1])

        serie = pd.Series(v_gyro)
        # Parâmetros
        window = 400
        threshold = 0.5  # Limite para marcar no gráfico

        # Média móvel
        media_movel = pd.Series(v_gyro).rolling(window=window, min_periods=1).mean()
        intervalos = []
        chave = 0
        seq = 0
        for index,valor in enumerate(media_movel):
            if valor > threshold and chave == 0 and seq == 0:
                
                intervalos.append(index)
                chave = 1
            elif valor < threshold and chave == 1 and seq <= 2:
                
                chave = 1
                seq = seq + 1
            elif valor < threshold and chave == 1 and seq > 2:
                
                intervalos.append(index)
                chave = 0
                seq = 0    
                
        intervalos = [x - 250 for x in intervalos]

        indices_v = []

        for i in range(0, len(intervalos), 2):
            if i+1 < len(intervalos):  # evita erro de índice ímpar
                ini = intervalos[i]
                fim = intervalos[i+1]
        
                pos_local, _ = find_peaks(
                    v_gyro[ini:fim],
                    height = 2,
                    
                )
        
                # converte os índices locais para índices absolutos
                pos_global = ini + pos_local
        
                indices_v.extend(pos_global)
            
        indices_ml, _ = find_peaks(ml_gyro, distance=150)

        # avisos úteis
        if len(indices_v) < 2 or len(indices_ml) < 2:
            st.warning("Poucos picos detectados em V ou ML. Ajuste 'height'/'distance' ou verifique o sinal.")

        # Ignora o primeiro como possível artefato (opcional)
        if len(indices_v) > 1:  indices_v  = indices_v[1:]
        if len(indices_ml) > 1: indices_ml = indices_ml[1:]

        # 7) Agrupa em ciclos (pares sucessivos A1/A2)
        def build_cycles(indices):
            n_pairs = len(indices) // 2
            return [(indices[2*i], indices[2*i+1]) for i in range(n_pairs)]

        cycles_v  = build_cycles(indices_v)
        cycles_ml = build_cycles(indices_ml)

        num_ciclos = min(len(cycles_v), len(cycles_ml))
        rows_v, rows_ml = [], []

        

        

        # =========================
        # Plotagem
        # =========================
        with c_plot1:
            # Trigger
            st.markdown("**Trigger — Velocidade angular (t = 0)**")
            fig_trig, ax_trig = plt.subplots(figsize=(10, 2))
            nwin = min(2000, len(t))
            ax_trig.plot(t[:nwin], norm[:nwin], 'k-', label="‖ω‖")            
            ax_trig.axvline(0, color='r', label="t=0")
            ax_trig.set_xlabel("Tempo (s)")
            ax_trig.set_ylabel("Velocidade angular (rad/s)")
            ax_trig.legend(loc="lower left")
            st.pyplot(fig_trig)

            c1, c2 = st.columns(2)

            # ---- Coluna 1: Vertical ----
            with c1:
                fig_v, ax_v = plt.subplots(figsize=(10, 6))
                ax_v.plot(t, v_gyro, 'k-', label='Vertical (|ω_y|)')
                #ax_v.plot(t, media_movel, 'r-', label='Vertical (|ω_y|)')
                ax_v.plot([t[intervalos],t[intervalos]],[0.3,5], '-y')
                ax_v.axvline(0, color='r', ls='--', label="t=0")

                for i in range(num_ciclos):
                    a1_idx, a2_idx = cycles_v[i]
                    a1_t = t[a1_idx]; a2_t = t[a2_idx]
                    da1 = float(st.session_state["adj_onset"].get(i, 0.0))
                    da2 = float(st.session_state["adj_offset"].get(i, 0.0))
                    a1_t_adj = clamp(a1_t + da1, t_min, t_max)
                    a2_t_adj = clamp(a2_t + da2, t_min, t_max)

                    ax_v.plot(t[a1_idx], v_gyro[a1_idx], 'ro')
                    ax_v.plot(t[a2_idx], v_gyro[a2_idx], 'ro')
                    ax_v.axvline(a1_t_adj, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_v.axvline(a2_t_adj, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")

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

            # ---- Coluna 2: ML ----
            with c2:
                fig_ml, ax_ml = plt.subplots(figsize=(10, 6))
                ax_ml.plot(t, ml_gyro, 'k-', label='ML (|ω_x|)')
                ax_ml.axvline(0, color='r', ls='--', label="t=0")

                for i in range(num_ciclos):
                    a1_idx, a2_idx = cycles_ml[i]
                    a1_t = t[a1_idx]; a2_t = t[a2_idx]
                    da1 = float(st.session_state["adj_onset"].get(i, 0.0))
                    da2 = float(st.session_state["adj_offset"].get(i, 0.0))
                    a1_t_adj = clamp(a1_t + da1, t_min, t_max)
                    a2_t_adj = clamp(a2_t + da2, t_min, t_max)

                    ax_ml.plot(t[a1_idx], ml_gyro[a1_idx], 'ro')
                    ax_ml.plot(t[a2_idx], ml_gyro[a2_idx], 'ro')
                    ax_ml.axvline(a1_t_adj, color='orange', ls='--', label='A1 (aj)' if i == 0 else "")
                    ax_ml.axvline(a2_t_adj, color='green',  ls='--', label='A2 (aj)' if i == 0 else "")

                    rows_ml.append({
                        "ciclo": i,
                        "A1_t(s)": a1_t_adj,
                        "A1_amp(ML)": float(ml_gyro[a1_idx]),
                        "A2_t(s)": a2_t_adj,
                        "A2_amp(ML)": float(ml_gyro[a2_idx]),
                    })

                ax_ml.set_xlabel("Tempo (s)")
                ax_ml.set_ylabel("Velocidade angular (ML)")
                ax_ml.legend(loc="lower left")
                st.pyplot(fig_ml)

            # =========================
            # Tabelas e Exportação
            # =========================
            st.subheader("Tempos por ciclo — Velocidade angular (Vertical)")
            df_tempos_v = pd.DataFrame(rows_v)
            st.dataframe(df_tempos_v, width='stretch')

            st.subheader("Tempos por ciclo — Velocidade angular (ML)")
            df_tempos_ml = pd.DataFrame(rows_ml)
            st.dataframe(df_tempos_ml, width='stretch')

            df_join = pd.DataFrame()
            if not df_tempos_v.empty and not df_tempos_ml.empty:
                df_join = df_tempos_v.merge(df_tempos_ml, on="ciclo", suffixes=("_V", "_ML"))

            st.subheader("Exportar resultados")
            col_v, col_ml, col_all = st.columns(3)
            with col_v:
                if not df_tempos_v.empty:
                    st.download_button(
                        "📥 CSV — Vertical",
                        df_tempos_v.to_csv(index=False).encode("utf-8"),
                        file_name="tempos_gyro_vertical.csv",
                        mime="text/csv",
                        key="btn_export_v"
                    )
            with col_ml:
                if not df_tempos_ml.empty:
                    st.download_button(
                        "📥 CSV — ML",
                        df_tempos_ml.to_csv(index=False).encode("utf-8"),
                        file_name="tempos_gyro_ml.csv",
                        mime="text/csv",
                        key="btn_export_ml"
                    )
            with col_all:
                if not df_join.empty:
                    st.download_button(
                        "📦 CSV — Vertical + ML",
                        df_join.to_csv(index=False).encode("utf-8"),
                        file_name="tempos_gyro_merged.csv",
                        mime="text/csv",
                        key="btn_export_merged"
                    )
                else:
                    st.caption("Sem ciclos suficientes para exportar o CSV combinado.")
