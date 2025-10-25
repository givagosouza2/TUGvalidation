import streamlit as st
import pandas as pd
import numpy as np
import scipy
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

# Estado das abas dinâmicas
if "show_dyn_tabs" not in st.session_state:
    st.session_state.show_dyn_tabs = False

# Botão global para liberar abas extras
st.markdown("—")
col_btn = st.columns([1,6])[0]
if not st.session_state.show_dyn_tabs:
    if col_btn.button("Liberar abas ➜ Aceleração e Velocidade Angular"):
        st.session_state.show_dyn_tabs = True

# Abas (dinâmicas)
tab_labels = ["Kinematics"]
if st.session_state.show_dyn_tabs:
    tab_labels += ["Acceleration", "Angular velocity"]

tabs = st.tabs(tab_labels)
tab_map = {label: tabs[i] for i, label in enumerate(tab_labels)}

# Estados para ajustes finos (cinemática)
for key in ("adj_onset", "adj_offset", "adj_stand", "adj_sit", "adj_peaks"):
    if key not in st.session_state:
        st.session_state[key] = {}

# Estados para ajustes finos (aceleração)
for key in ("adj_onset_acc", "adj_offset_acc", "adj_peak_acc"):
    if key not in st.session_state:
        st.session_state[key] = {}

# =========================
# TAB: KINEMATICS
# =========================
with tab_map["Kinematics"]:
    # Layout: col de controles + duas cols de plots/tabela
    c_ctrl, c_plot1 = st.columns([0.7, 2])

    with c_ctrl:
        st.subheader("Controles — Cinemática")

        uploaded_file_kinem = st.file_uploader(
            "Arquivo (.csv: X, Y, Z em mm)", type=["csv"], key="kin_file"
        )

        st.markdown("**Trigger (alinha t=0)**")
        trigger_idx_shift = st.number_input("Índice de referência", 0, 100000, 0, 1, key="kin_trig")

        st.markdown("**Pré-processamento**")
        do_detrend = st.checkbox("Aplicar detrend", value=False, key="kin_detrend")
        do_filter  = st.checkbox("Aplicar filtro passa-baixa", value=True, key="kin_filt")
        cutoff_kinem = st.number_input("Cutoff (Hz)", 0.1, 20.0, 2.0, 0.1, key="kin_cutoff")

        st.markdown("**Detecção de eventos**")
        prominence = st.number_input("Prominence mínima", 0.0, 1000.0, 2.5, 0.1, key="kin_prom")
        min_distance_samples = st.number_input("Distância mínima (amostras)", 1, 10000, 200, 1, key="kin_dist")

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
        if cr1.button("Reset ciclo"):
            st.session_state["adj_onset"].pop(sel_cycle, None)
            st.session_state["adj_offset"].pop(sel_cycle, None)
            st.session_state["adj_stand"].pop(sel_cycle, None)
            st.session_state["adj_sit"].pop(sel_cycle, None)
        if cr2.button("Reset tudo"):
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
                if disp_y[j] > disp_y[j-1]: onsets.append(j); break
            for j in range(p, len(disp_y)-1):
                if disp_y[j] > disp_y[j+1]: offsets.append(j); break

        num_ciclos = min(len(onsets), len(offsets))

        # standing / sitting
        stand_times, sit_times = [], []
        for i in range(num_ciclos):
            v = onsets[i]
            a, b = v, min(v+200, len(disp_z))
            if b > a: stand_times.append(t[a + int(np.argmax(disp_z[a:b]))])
            v = offsets[i]
            a, b = max(0, v-400), v
            if b > a: sit_times.append(t[a + int(np.argmax(disp_z[a:b]))])

        # tempos ajustados
        onset_times = [t[i] for i in onsets[:num_ciclos]]
        offset_times = [t[i] for i in offsets[:num_ciclos]]
        peak_times  = [t[i] for i in peaks]
        onset_adj = [clamp(v + st.session_state["adj_onset"].get(i,0.0), t_min, t_max) for i,v in enumerate(onset_times)]
        offset_adj = [clamp(v + st.session_state["adj_offset"].get(i,0.0), t_min, t_max) for i,v in enumerate(offset_times)]
        stand_adj  = [clamp(v + st.session_state["adj_stand"].get(i,0.0), t_min, t_max) for i,v in enumerate(stand_times)]
        sit_adj    = [clamp(v + st.session_state["adj_sit"].get(i,0.0),   t_min, t_max) for i,v in enumerate(sit_times)]
        peak_adj   = [clamp(v + st.session_state["adj_peaks"].get(i,0.0), t_min, t_max) for i,v in enumerate(peak_times)]
        
        # PLOT 1 (coluna do meio): disp_y + marcações
        with c_plot1:
            # --- GRÁFICO DE TRIGGER (KINEMÁTICA) ---
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
            st.dataframe(df_tempos, use_container_width=True)
            st.download_button(
                "Baixar CSV (Cinemática)",
                df_tempos.to_csv(index=False).encode("utf-8"),
                file_name="tempos_ciclos_cinematica.csv",
                mime="text/csv"
            )
    else:
        st.info("Carregue um arquivo de cinemática para visualizar.")

# =========================
# TAB: ACCELERATION (após liberar)
# =========================
if st.session_state.show_dyn_tabs:
    with tab_map["Acceleration"]:
        c_ctrl, c_plot1 = st.columns([0.7, 2])

        with c_ctrl:
            st.subheader("Controles — Aceleração")
            uploaded_file_acc = st.file_uploader(
                "Arquivo (.txt: [tempo], ax, ay, az)", type=["txt"], key="acc_file"
            )
            st.markdown("**Pré-processamento**")
            do_detrend_acc = st.checkbox("Aplicar detrend", value=False, key="acc_detrend")
            do_filter_acc  = st.checkbox("Aplicar filtro passa-baixa", value=True, key="acc_filt")
            cutoff_acc = st.number_input("Cutoff (Hz)", 0.1, 50.0, 6.0, 0.1, key="acc_cutoff")

            st.markdown("**Tempo / Amostragem**")
            fs_acc = st.number_input("Frequência de amostragem (Hz)", 1.0, 2000.0, 100.0, 1.0, key="acc_fs")
            trigger_acc = st.number_input("Trigger (s)", -5.0, 5.0, 0.0, 0.01, key="acc_trig")

            st.markdown("**Detecção de eventos**")
            axis_acc = st.selectbox("Eixo para eventos", ["ax", "ay", "az"], index=2, key="acc_axis")
            prominence_acc = st.number_input("Prominence mínima", 0.0, 1000.0, 2.5, 0.1, key="acc_prom")
            min_distance_samples_acc = st.number_input("Distância mínima (amostras)", 1, 10000, 200, 1, key="acc_dist")

            st.markdown("**Ajustes finos**")
            sel_cycle_acc = st.number_input("Ciclo (0-index)", 0, 9999, 0, 1, key="acc_sel_cycle")
            d_on_acc  = st.number_input("Δ Onset (s)",  -2.0, 2.0, float(st.session_state["adj_onset_acc"].get(sel_cycle_acc, 0.0)), 0.01, key="acc_don")
            d_off_acc = st.number_input("Δ Offset (s)", -2.0, 2.0, float(st.session_state["adj_offset_acc"].get(sel_cycle_acc, 0.0)), 0.01, key="acc_doff")
            st.session_state["adj_onset_acc"][sel_cycle_acc] = d_on_acc
            st.session_state["adj_offset_acc"][sel_cycle_acc] = d_off_acc

            sel_peak_acc = st.number_input("Pico (mínimo) 0-index", 0, 9999, 0, 1, key="acc_sel_peak")
            d_pk_acc = st.number_input("Δ Mínimo (s)", -2.0, 2.0, float(st.session_state["adj_peak_acc"].get(sel_peak_acc, 0.0)), 0.01, key="acc_dpk")
            st.session_state["adj_peak_acc"][sel_peak_acc] = d_pk_acc

            rc1, rc2 = st.columns(2)
            if rc1.button("Reset ciclo (acc)"):
                st.session_state["adj_onset_acc"].pop(sel_cycle_acc, None)
                st.session_state["adj_offset_acc"].pop(sel_cycle_acc, None)
            if rc2.button("Reset tudo (acc)"):
                st.session_state["adj_onset_acc"].clear()
                st.session_state["adj_offset_acc"].clear()
                st.session_state["adj_peak_acc"].clear()
                
            if uploaded_file_acc is not None:
                # 1) Leitura robusta (auto-separador)
                df_acc = pd.read_csv(uploaded_file_acc, sep=None, engine="python")
            
                # 2) Identifica formato: [tempo, ax, ay, az] OU [ax, ay, az]
                if df_acc.shape[1] >= 4:
                    try:
                        t_acc = df_acc.iloc[:, 0].astype(float).values
                        ax = df_acc.iloc[:, 1].astype(float).values
                        ay = df_acc.iloc[:, 2].astype(float).values
                        az = df_acc.iloc[:, 3].astype(float).values
                    except Exception:
                        st.error("As quatro primeiras colunas devem ser numéricas (tempo, ax, ay, az).")
                        st.stop()
                elif df_acc.shape[1] >= 3:
                    try:
                        ax = df_acc.iloc[:, 0].astype(float).values
                        ay = df_acc.iloc[:, 1].astype(float).values
                        az = df_acc.iloc[:, 2].astype(float).values
                    except Exception:
                        st.error("As três primeiras colunas devem ser numéricas (ax, ay, az).")
                        st.stop()
                    # Sem coluna de tempo: constrói pelo fs informado
                    t_acc = build_time_vector(len(ax), fs_acc)
                else:
                    st.error("Arquivo deve ter 3 ou 4 colunas ([tempo], ax, ay, az).")
                    st.stop()
            
                # 3) Alinha trigger (em segundos)
                t_acc = t_acc - float(trigger_acc)
            
                # 4) Reamostra para uma taxa uniforme (new_fs)
                new_fs = 100.0  # Hz (ajuste se quiser controlar pela UI)
                t_start, t_end = float(t_acc[0]), float(t_acc[-1])
                if t_end <= t_start:
                    st.error("A coluna de tempo precisa ser estritamente crescente.")
                    st.stop()
            
                t_new = np.arange(t_start, t_end, 1.0/new_fs)
            
                # interp1d exige tempo crescente e sem NaN
                f_ax = interp1d(t_acc, ax, kind="linear", bounds_error=False, fill_value="extrapolate")
                f_ay = interp1d(t_acc, ay, kind="linear", bounds_error=False, fill_value="extrapolate")
                f_az = interp1d(t_acc, az, kind="linear", bounds_error=False, fill_value="extrapolate")
            
                ax_i = f_ax(t_new)
                ay_i = f_ay(t_new)
                az_i = f_az(t_new)
            
                # 5) Pré-processamento (detrend/filtro) — agora usando new_fs!
                if do_detrend_acc:
                    ax_i = detrend(ax_i); ay_i = detrend(ay_i); az_i = detrend(az_i)
                if do_filter_acc:
                    ax_i = low_pass_filter(ax_i, cutoff_acc, new_fs)
                    ay_i = low_pass_filter(ay_i, cutoff_acc, new_fs)
                    az_i = low_pass_filter(az_i, cutoff_acc, new_fs)
            
                # 6) Gráfico de trigger (use a série do eixo escolhido)
                axis_map = {"ax": ax_i, "ay": ay_i, "az": az_i}
                sig = axis_map[axis_acc]
            
                with c_plot1:
                    st.markdown("**Trigger — Aceleração (t = 0)**")
                    fig_trig_acc, ax_trig_acc = plt.subplots(figsize=(10, 2))
                    nwin_acc = min(2000, len(t_new))
                    ax_trig_acc.plot(t_new[:nwin_acc], sig[:nwin_acc], 'k-', label=axis_acc)
                    ax_trig_acc.axvline(0, color='r', label="t=0")
                    ax_trig_acc.set_xlabel("Tempo (s)")
                    ax_trig_acc.set_ylabel("Aceleração")
                    ax_trig_acc.legend(loc="lower left")
                    st.pyplot(fig_trig_acc)
            
                    # Visuais auxiliares: vert (ay) e AP (az) ao quadrado (se quiser manter essa métrica)
                    c_plot11, c_plot12 = st.columns(2)
                    with c_plot11:
                        fig_vert_acc, ax_vert_acc = plt.subplots(figsize=(10, 6))
                        ax_vert_acc.plot(t_new, ay_i**2, 'k-', label='acc V (ay²)')
                        ax_vert_acc.axvline(0, color='r', label="t=0")
                        ax_vert_acc.set_xlabel("Tempo (s)")
                        ax_vert_acc.set_ylabel("Aceleração²")
                        ax_vert_acc.legend(loc="lower left")
                        st.pyplot(fig_vert_acc)
                    with c_plot12:
                        fig_ap_acc, ax_ap_acc = plt.subplots(figsize=(10, 6))
                        ax_ap_acc.plot(t_new, az_i**2, 'k-', label='acc AP (az²)')
                        ax_ap_acc.axvline(0, color='r', label="t=0")
                        ax_ap_acc.set_xlabel("Tempo (s)")
                        ax_ap_acc.set_ylabel("Aceleração²")
                        ax_ap_acc.legend(loc="lower left")
                        st.pyplot(fig_ap_acc)    

    with tab_map["Angular velocity"]:
        st.write("Conteúdo de Angular velocity (a definir).")
