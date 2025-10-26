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

def build_time_vector(n, fs, t0=0.0):
    return t0 + np.arange(n) / float(fs)

def first_min_within(peaks_times, t_on, t_off):
    cand = [t for t in peaks_times if t_on <= t <= t_off]
    return cand[0] if cand else np.nan

# =========================
# App
# =========================
st.set_page_config(layout="wide")
st.title("Análise de Dados: Interpolação, Detrend e Filtro Passa-Baixa")

# defaults de estado que usamos em UI/plot (evita NameError)
defaults = {
    "acc_trig": 0.0,  # trigger da aba Acceleration (reservado p/ futuras abas)
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Estados para ajustes finos (cinemática)
for key in ("adj_onset", "adj_offset", "adj_stand", "adj_sit", "adj_peaks"):
    if key not in st.session_state:
        st.session_state[key] = {}

# Estados para ajustes finos (aceleração) – reservados para futuras abas
for key in ("adj_onset_acc", "adj_offset_acc", "adj_peak_acc"):
    if key not in st.session_state:
        st.session_state[key] = {}

# =========================
# TAB: KINEMATICS
# =========================
tab1, = st.tabs(["Kinematics"])  # CORREÇÃO: st.tabs recebe uma lista e retorna uma lista de containers
with tab1:
    # Layout: coluna de controles + coluna de visualização (com subcolunas)
    c_ctrl, c_plot1 = st.columns([0.7, 2])

    with c_ctrl:
        st.subheader("Controles — Cinemática")

        uploaded_file_kinem = st.file_uploader(
            "Arquivo (.csv: X, Y, Z em mm)", type=["csv"], key="kin_file"
        )

        st.markdown("**Trigger (alinha t=0)**")
        trigger_idx_shift = st.number_input("Índice de referência", 0, 100000, 0, 1, key="kin_trig")

        # Defaults solicitados: detrend desmarcado, filtro marcado
        do_detrend = False
        do_filter = True

        # Parâmetros fixos solicitados
        cutoff_kinem = 2.0
        prominence = 2.5
        min_distance_samples = 200

        st.markdown("**Ajustes finos**")
        sel_cycle = st.number_input("Ciclo (0-index)", 0, 9999, 0, 1, key="kin_sel_cycle")
        d_on = st.number_input("Δ Onset (s)", -2.0, 2.0, float(st.session_state["adj_onset"].get(sel_cycle, 0.0)), 0.01, key="kin_don")
        d_off = st.number_input("Δ Offset (s)", -2.0, 2.0, float(st.session_state["adj_offset"].get(sel_cycle, 0.0)), 0.01, key="kin_doff")
        d_st = st.number_input("Δ Pico em pé (s)", -2.0, 2.0, float(st.session_state["adj_stand"].get(sel_cycle, 0.0)), 0.01, key="kin_dst")
        d_si = st.number_input("Δ Pico para sentar (s)", -2.0, 2.0, float(st.session_state["adj_sit"].get(sel_cycle, 0.0)), 0.01, key="kin_dsi")

        st.session_state["adj_onset"][sel_cycle] = d_on
        st.session_state["adj_offset"][sel_cycle] = d_off
        st.session_state["adj_stand"][sel_cycle] = d_st
        st.session_state["adj_sit"][sel_cycle]   = d_si

        sel_peak = st.number_input("Pico (mínimo) 0-index", 0, 9999, 0, 1, key="kin_sel_peak")
        d_pk = st.number_input("Δ Mínimo (s)", -2.0, 2.0, float(st.session_state["adj_peaks"].get(sel_peak, 0.0)), 0.01, key="kin_dpk")
        st.session_state["adj_peaks"][sel_peak] = d_pk

        cr1, cr2 = st.columns(2)
        if cr1.button("Reset ciclo", key="btn_reset_cycle_kin"):
            st.session_state["adj_onset"].pop(sel_cycle, None)
            st.session_state["adj_offset"].pop(sel_cycle, None)
            st.session_state["adj_stand"].pop(sel_cycle, None)
            st.session_state["adj_sit"].pop(sel_cycle, None)
        if cr2.button("Reset tudo", key="btn_reset_all_kin"):
            for k in ("adj_onset","adj_offset","adj_stand","adj_sit","adj_peaks"):
                st.session_state[k].clear()

    # Processamento e visualização
    if uploaded_file_kinem is not None:
        df = pd.read_csv(uploaded_file_kinem, sep=",", engine="python")
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

        fs = 100.0
        t = np.arange(len(disp_y)) / fs
        idx0 = int(clamp(trigger_idx_shift, 0, len(t)-1)) if len(t) else 0
        t = t - t[idx0]
        t_min, t_max = (t[0], t[-1]) if len(t) else (0.0, 0.0)

        if do_detrend:
            disp_x = detrend(disp_x); disp_y = detrend(disp_y); disp_z = detrend(disp_z)
        if do_filter:
            disp_x = low_pass_filter(disp_x, cutoff_kinem, fs)
            disp_y = low_pass_filter(disp_y, cutoff_kinem, fs)
            disp_z = low_pass_filter(disp_z, cutoff_kinem, fs)

        pk_kwargs = {}
        if prominence > 0: pk_kwargs["prominence"] = float(prominence)
        if min_distance_samples > 1: pk_kwargs["distance"] = int(min_distance_samples)
        peaks, _ = find_peaks(-disp_y, **pk_kwargs)

        onsets, offsets = [], []
        for p in peaks:
            for j in range(p, 1, -1):
                if disp_y[j] > disp_y[j-1]:
                    onsets.append(j); break
            for j in range(p, len(disp_y)-1):
                if disp_y[j] > disp_y[j+1]:
                    offsets.append(j); break

        num_ciclos = min(len(onsets), len(offsets))

        # standing / sitting
        stand_times, sit_times = [], []
        for i in range(num_ciclos):
            v = onsets[i]
            a, b = v, min(v+200, len(disp_z))
            if b > a:
                stand_times.append(t[a + int(np.argmax(disp_z[a:b]))])
            v = offsets[i]
            a, b = max(0, v-400), v
            if b > a:
                sit_times.append(t[a + int(np.argmax(disp_z[a:b]))])

        # tempos ajustados
        onset_times = [t[i] for i in onsets[:num_ciclos]]
        offset_times = [t[i] for i in offsets[:num_ciclos]]
        peak_times  = [t[i] for i in peaks]
        onset_adj = [clamp(v + st.session_state["adj_onset"].get(i,0.0), t_min, t_max) for i,v in enumerate(onset_times)]
        offset_adj = [clamp(v + st.session_state["adj_offset"].get(i,0.0), t_min, t_max) for i,v in enumerate(offset_times)]
        stand_adj  = [clamp(v + st.session_state["adj_stand"].get(i,0.0), t_min, t_max) for i,v in enumerate(stand_times)]
        sit_adj    = [clamp(v + st.session_state["adj_sit"].get(i,0.0),   t_min, t_max) for i,v in enumerate(sit_times)]
        peak_adj   = [clamp(v + st.session_state["adj_peaks"].get(i,0.0), t_min, t_max) for i,v in enumerate(peak_times)]

        # PLOT 1 (t=0 + dois gráficos lado a lado)
        with c_plot1:
            # Trigger
            st.markdown("**Trigger — Cinemática (t = 0)**")
            fig_trig_kin, ax_trig_kin = plt.subplots(figsize=(10, 3))
            nwin = min(2000, len(t))
            ax_trig_kin.plot(t[:nwin], disp_z[:nwin], 'k-', label="disp_z")
            ax_trig_kin.axvline(0, color='r', label="t=0")
            ax_trig_kin.set_xlabel("Tempo (s)")
            ax_trig_kin.set_ylabel("Amplitude (m)")
            ax_trig_kin.legend(loc="lower left")
            st.pyplot(fig_trig_kin)

            c_plot11, c_plot12 = st.columns(2)

            with c_plot11:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(t, disp_y, 'k-', label="Desloc. AP")
                for i in range(num_ciclos):
                    on, of = onset_adj[i], offset_adj[i]
                    ax2.axvline(on, ls='--', color='orange', label='Início' if i==0 else "")
                    ax2.axvline(of, ls='--', color='green',  label='Fim' if i==0 else "")
                    ax2.axvspan(on, of, color='gray', alpha=0.3, label='Teste' if i==0 else "")
                    if i < len(stand_adj): ax2.axvline(stand_adj[i], ls='--', color='red',   label='Pico em pé' if i==0 else "")
                    if i < len(sit_adj):   ax2.axvline(sit_adj[i],   ls='--', color='black', label='Pico para sentar' if i==0 else "")
                for k, tp in enumerate(peak_adj):
                    ax2.axvline(tp, ls='--', color='blue', label='Mínimos' if k==0 else "")
                ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Amplitude (m)")
                ax2.legend(loc="lower left")
                st.pyplot(fig2)

            with c_plot12:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(t, disp_z, 'k-', label="Desloc. vertical")
                for i in range(num_ciclos):
                    on, of = onset_adj[i], offset_adj[i]
                    ax3.axvline(on, ls='--', color='orange', label='Início' if i==0 else "")
                    ax3.axvline(of, ls='--', color='green',  label='Fim' if i==0 else "")
                    ax3.axvspan(on, of, color='gray', alpha=0.3, label='Teste' if i==0 else "")
                    if i < len(stand_adj): ax3.axvline(stand_adj[i], ls='--', color='red',   label='Pico em pé' if i==0 else "")
                    if i < len(sit_adj):   ax3.axvline(sit_adj[i],   ls='--', color='black', label='Pico para sentar' if i==0 else "")
                for k, tp in enumerate(peak_adj):
                    ax3.axvline(tp, ls='--', color='blue', label='Mínimos' if k==0 else "")
                ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Amplitude (m)")
                ax3.legend(loc="lower left")
                st.pyplot(fig3)

            # Tabela de tempos por ciclo + download
            rows = []
            for i in range(num_ciclos):
                t_on, t_off = onset_adj[i], offset_adj[i]
                t_st = stand_adj[i] if i < len(stand_adj) else np.nan
                t_si = sit_adj[i]   if i < len(sit_adj)   else np.nan
                t_min = first_min_within(peak_adj, t_on, t_off)
                rows.append({"ciclo": i, "onset_s": t_on, "offset_s": t_off,
                             "pico_em_pe_s": t_st, "pico_para_sentar_s": t_si, "minimo_s": t_min})
            df_tempos = pd.DataFrame(rows)
            st.subheader("Tempos por ciclo — Cinemática")
            st.dataframe(df_tempos, width='stretch')
            st.download_button(
                "Baixar CSV (Cinemática)",
                df_tempos.to_csv(index=False).encode("utf-8"),
                file_name="tempos_ciclos_cinematica.csv",
                mime="text/csv"
            )
    else:
        st.info("Carregue um arquivo de cinemática para visualizar.")
