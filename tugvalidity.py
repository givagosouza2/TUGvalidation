import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks
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

# =========================
# Sidebar — CINEMÁTICA (topo da barra, botão por último)
# =========================
st.sidebar.header("Configurações — Cinemática")

# Arquivo de cinemática
uploaded_file_kinem = st.sidebar.file_uploader(
    "Arquivo de cinemática (.csv: X, Y, Z em mm)", type=["csv"]
)

# Pré-processamento (defaults pedidos)
st.sidebar.subheader("Pré-processamento")
do_detrend = st.sidebar.checkbox("Aplicar detrend", value=False)
do_filter = st.sidebar.checkbox("Aplicar filtro passa-baixa", value=True)
cutoff_kinem = st.sidebar.number_input(
    "Cutoff do filtro (Hz)", min_value=0.1, max_value=20.0, value=2.0, step=0.1
)

# Detecção de eventos (defaults pedidos)
st.sidebar.subheader("Detecção dos eventos")
prominence = st.sidebar.number_input(
    "Prominence mínima", min_value=0.0, value=2.5, step=0.1
)
min_distance_samples = st.sidebar.number_input(
    "Distância mínima entre picos (amostras)", min_value=1, value=200, step=1
)

# Trigger (alinha t=0) — cinemática
trigger_idx_shift = st.sidebar.slider(
    "Trigger (alinha t=0) — Cinemática (por índice)",
    min_value=0, max_value=10000, value=0, step=1,
    help="Desloca o t=0 para o índice selecionado (requer arquivo carregado)"
)

# Estado das abas dinâmicas
if "show_dyn_tabs" not in st.session_state:
    st.session_state.show_dyn_tabs = False

# Botão FINAL da sidebar (fica por último enquanto abas extras não liberadas)
st.sidebar.markdown("---")
if not st.session_state.show_dyn_tabs:
    if st.sidebar.button("Liberar abas ➜ Aceleração e Velocidade Angular"):
        st.session_state.show_dyn_tabs = True

# Após liberar, podemos mostrar as seções de Aceleração abaixo (sem violar sua ordem original)
if st.session_state.show_dyn_tabs:
    st.sidebar.markdown("---")
    st.sidebar.header("Configurações — Aceleração")

    # Arquivo de aceleração
    uploaded_file_acc = st.sidebar.file_uploader(
        "Arquivo de aceleração (.csv: [tempo?], ax, ay, az)", type=["csv"], key="acc_file_uploader"
    )

    # Pré-processamento aceleração (herdo defaults de cinemática para consistência)
    st.sidebar.subheader("Pré-processamento (Aceleração)")
    do_detrend_acc = st.sidebar.checkbox("Aplicar detrend (acc)", value=False)
    do_filter_acc = st.sidebar.checkbox("Aplicar filtro passa-baixa (acc)", value=True)
    cutoff_acc = st.sidebar.number_input(
        "Cutoff do filtro (Hz) — acc", min_value=0.1, max_value=50.0, value=6.0, step=0.1
    )

    # Tempo — caso o arquivo não tenha coluna de tempo
    st.sidebar.subheader("Tempo / Amostragem (Aceleração)")
    fs_acc = st.sidebar.number_input(
        "Frequência de amostragem (Hz) — acc", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
    )
    trigger_acc = st.sidebar.slider(
        "Trigger (alinha t=0) — Aceleração (s)", min_value=-5.0, max_value=5.0, value=0.0, step=0.01
    )

    # Eixo para análise de eventos
    st.sidebar.subheader("Detecção de eventos (Aceleração)")
    axis_acc = st.sidebar.selectbox("Eixo para análise", options=["ax", "ay", "az"], index=2)
    prominence_acc = st.sidebar.number_input("Prominence mínima — acc", min_value=0.0, value=2.5, step=0.1)
    min_distance_samples_acc = st.sidebar.number_input(
        "Distância mínima entre picos (amostras) — acc", min_value=1, value=200, step=1
    )

# =========================
# Abas (dinâmicas)
# =========================
tab_labels = ["Kinematics"]
if st.session_state.show_dyn_tabs:
    tab_labels += ["Acceleration", "Angular velocity"]

tabs = st.tabs(tab_labels)
tab_map = {label: tabs[i] for i, label in enumerate(tab_labels)}

# =========================
# Estados para ajustes finos
# =========================
for key in ("adj_onset", "adj_offset", "adj_stand", "adj_sit", "adj_peaks"):
    if key not in st.session_state:
        st.session_state[key] = {}

for key in ("adj_onset_acc", "adj_offset_acc", "adj_peak_acc"):
    if key not in st.session_state:
        st.session_state[key] = {}

# =========================
# KINEMATICS
# =========================
with tab_map["Kinematics"]:
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

        # fs e tempo
        fs = 100.0
        t = np.arange(len(disp_y)) / fs

        # trigger por índice (pedido anterior)
        idx0 = int(clamp(trigger_idx_shift, 0, len(t)-1)) if len(t) > 0 else 0
        t = t - t[idx0]

        # pré-processamento
        if do_detrend:
            disp_x = detrend(disp_x); disp_y = detrend(disp_y); disp_z = detrend(disp_z)
        if do_filter:
            disp_x = low_pass_filter(disp_x, cutoff_kinem, fs)
            disp_y = low_pass_filter(disp_y, cutoff_kinem, fs)
            disp_z = low_pass_filter(disp_z, cutoff_kinem, fs)

        # detecção de mínimos em disp_y
        pk_kwargs = {}
        if prominence > 0: pk_kwargs["prominence"] = float(prominence)
        if min_distance_samples > 1: pk_kwargs["distance"] = int(min_distance_samples)
        peaks, _ = find_peaks(-disp_y, **pk_kwargs)

        # onsets/offsets
        onsets, offsets = [], []
        for p in peaks:
            for j in range(p, 1, -1):
                if disp_y[j] > disp_y[j-1]:
                    onsets.append(j); break
            for j in range(p, len(disp_y)-1):
                if disp_y[j] > disp_y[j+1]:
                    offsets.append(j); break

        num_ciclos = min(len(onsets), len(offsets))
        t_min, t_max = (t[0], t[-1]) if len(t) else (0.0, 0.0)

        # standing / sitting com janelas protegidas
        stand_times, sit_times = [], []
        for i in range(num_ciclos):
            v = onsets[i]
            a, b = v, min(v+200, len(disp_z))
            if b > a: stand_times.append(t[a + int(np.argmax(disp_z[a:b]))])
            v = offsets[i]
            a, b = max(0, v-400), v
            if b > a: sit_times.append(t[a + int(np.argmax(disp_z[a:b]))])

        # ===== Ajustes finos (na sidebar, reaproveitando seção já existente) =====
        st.sidebar.subheader("Ajustes finos — Cinemática")
        sel_cycle = st.sidebar.number_input("Ciclo (0-index)", min_value=0, max_value=max(num_ciclos-1,0), value=0, step=1)
        # recupera valores atuais
        def _get(store, idx): return float(store.get(idx, 0.0))
        d_on = st.sidebar.number_input("Δ Onset (s)", -2.0, 2.0, _get(st.session_state.adj_onset, sel_cycle), 0.01)
        d_off = st.sidebar.number_input("Δ Offset (s)", -2.0, 2.0, _get(st.session_state.adj_offset, sel_cycle), 0.01)
        d_st = st.sidebar.number_input("Δ Pico em pé (s)", -2.0, 2.0, _get(st.session_state.adj_stand, sel_cycle), 0.01)
        d_si = st.sidebar.number_input("Δ Pico para sentar (s)", -2.0, 2.0, _get(st.session_state.adj_sit, sel_cycle), 0.01)
        st.session_state.adj_onset[sel_cycle] = d_on
        st.session_state.adj_offset[sel_cycle] = d_off
        st.session_state.adj_stand[sel_cycle] = d_st
        st.session_state.adj_sit[sel_cycle] = d_si

        if len(peaks) > 0:
            sel_peak = st.sidebar.number_input("Pico (mínimo) 0-index", min_value=0, max_value=len(peaks)-1, value=0, step=1)
            d_pk = st.sidebar.number_input("Δ Mínimo (s)", -2.0, 2.0, float(st.session_state.adj_peaks.get(sel_peak, 0.0)), 0.01)
            st.session_state.adj_peaks[sel_peak] = d_pk

        cr1, cr2 = st.sidebar.columns(2)
        if cr1.button("Reset ciclo (cinemática)"):
            for k in ("adj_onset","adj_offset","adj_stand","adj_sit"):
                st.session_state[k].pop(sel_cycle, None)
        if cr2.button("Reset tudo (cinemática)"):
            for k in ("adj_onset","adj_offset","adj_stand","adj_sit","adj_peaks"):
                st.session_state[k].clear()

        # tempos base e ajustados
        onset_times = [t[i] for i in onsets[:num_ciclos]]
        offset_times = [t[i] for i in offsets[:num_ciclos]]
        peak_times = [t[i] for i in peaks]
        onset_adj = [clamp(v + st.session_state.adj_onset.get(i,0.0), t_min, t_max) for i,v in enumerate(onset_times)]
        offset_adj = [clamp(v + st.session_state.adj_offset.get(i,0.0), t_min, t_max) for i,v in enumerate(offset_times)]
        stand_adj = [clamp(v + st.session_state.adj_stand.get(i,0.0), t_min, t_max) for i,v in enumerate(stand_times)]
        sit_adj = [clamp(v + st.session_state.adj_sit.get(i,0.0), t_min, t_max) for i,v in enumerate(sit_times)]
        peak_adj = [clamp(v + st.session_state.adj_peaks.get(i,0.0), t_min, t_max) for i,v in enumerate(peak_times)]

        # PLOTS
        fig1, ax1 = plt.subplots(figsize=(10, 2))
        ax1.plot(t[:2000], disp_z[:2000], 'k-', label="disp_z")
        ax1.axvline(0, color='r', label="t=0")
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Amplitude (m)")
        ax1.legend(loc="lower left")
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        with col1:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(t, disp_y, 'k-', label="Desloc. AP")
            for i in range(num_ciclos):
                on, of = onset_adj[i], offset_adj[i]
                ax2.axvline(on, ls='--', color='orange', label='Início' if i==0 else "")
                ax2.axvline(of, ls='--', color='green', label='Fim' if i==0 else "")
                ax2.axvspan(on, of, color='gray', alpha=0.3, label='Teste' if i==0 else "")
                if i < len(stand_adj): ax2.axvline(stand_adj[i], ls='--', color='red', label='Pico em pé' if i==0 else "")
                if i < len(sit_adj):   ax2.axvline(sit_adj[i],   ls='--', color='black', label='Pico para sentar' if i==0 else "")
            for k, tp in enumerate(peak_adj):
                ax2.axvline(tp, ls='--', color='blue', label='Mínimos' if k==0 else "")
            ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Amplitude (m)")
            ax2.legend(loc="lower left"); st.pyplot(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(t, disp_z, 'k-', label="Desloc. vertical")
            for i in range(num_ciclos):
                on, of = onset_adj[i], offset_adj[i]
                ax3.axvline(on, ls='--', color='orange', label='Início' if i==0 else "")
                ax3.axvline(of, ls='--', color='green', label='Fim' if i==0 else "")
                ax3.axvspan(on, of, color='gray', alpha=0.3, label='Teste' if i==0 else "")
                if i < len(stand_adj): ax3.axvline(stand_adj[i], ls='--', color='red', label='Pico em pé' if i==0 else "")
                if i < len(sit_adj):   ax3.axvline(sit_adj[i],   ls='--', color='black', label='Pico para sentar' if i==0 else "")
            for k, tp in enumerate(peak_adj):
                ax3.axvline(tp, ls='--', color='blue', label='Mínimos' if k==0 else "")
            ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Amplitude (m)")
            ax3.legend(loc="lower left"); st.pyplot(fig3)

        # Tabela de tempos por ciclo (cinemática) + download
        rows = []
        for i in range(num_ciclos):
            t_on, t_off = onset_adj[i], offset_adj[i]
            t_st = stand_adj[i] if i < len(stand_adj) else np.nan
            t_si = sit_adj[i]   if i < len(sit_adj)   else np.nan
            t_min = first_min_within(peak_adj, t_on, t_off)
            rows.append({"ciclo": i, "onset_s": t_on, "offset_s": t_off,
                         "pico_em_pe_s": t_st, "pico_para_sentar_s": t_si, "minimo_s": t_min})
        df_tempos = pd.DataFrame(rows)
        st.session_state["cinem_df_tempos"] = df_tempos.copy()
        st.subheader("Tempos por ciclo — Cinemática")
        st.dataframe(df_tempos, use_container_width=True)
        st.download_button(
            "Baixar CSV (Cinemática)",
            df_tempos.to_csv(index=False).encode("utf-8"),
            file_name="tempos_ciclos_cinematica.csv",
            mime="text/csv"
        )
    else:
        st.warning("Faça upload do arquivo de cinemática para visualizar os resultados.")

# =========================
# ACCELERATION (aparece após liberar abas)
# =========================
if st.session_state.show_dyn_tabs:
    with tab_map["Acceleration"]:
        if 'uploaded_file_acc' not in locals():
            # quando a página recarrega, variáveis locais do sidebar são reavaliadas
            uploaded_file_acc = st.session_state.get("acc_file_uploader", None)

        if uploaded_file_acc is not None:
            df_acc = pd.read_csv(uploaded_file_acc, sep=",", engine="python")

            # Detecta se há coluna de tempo: 4 colunas => [tempo, ax, ay, az]; 3 colunas => [ax, ay, az]
            if df_acc.shape[1] >= 4:
                # tenta: tempo + 3 eixos
                try:
                    t_acc = df_acc.iloc[:,0].astype(float).values
                    ax = df_acc.iloc[:,1].astype(float).values
                    ay = df_acc.iloc[:,2].astype(float).values
                    az = df_acc.iloc[:,3].astype(float).values
                except Exception:
                    st.error("As quatro primeiras colunas devem ser numéricas (tempo, ax, ay, az).")
                    st.stop()
            elif df_acc.shape[1] >= 3:
                try:
                    ax = df_acc.iloc[:,0].astype(float).values
                    ay = df_acc.iloc[:,1].astype(float).values
                    az = df_acc.iloc[:,2].astype(float).values
                except Exception:
                    st.error("As três primeiras colunas devem ser numéricas (ax, ay, az).")
                    st.stop()
                t_acc = build_time_vector(len(ax), fs_acc)
            else:
                st.error("Arquivo de aceleração precisa ter 3 ou 4 colunas (tempo opcional + ax, ay, az).")
                st.stop()

            # Alinha tempo (trigger em segundos)
            t_acc = t_acc - trigger_acc
            t_min_acc, t_max_acc = t_acc[0], t_acc[-1]

            # Pré-processamento
            if do_detrend_acc:
                ax = detrend(ax); ay = detrend(ay); az = detrend(az)
            if do_filter_acc:
                ax = low_pass_filter(ax, cutoff_acc, fs_acc)
                ay = low_pass_filter(ay, cutoff_acc, fs_acc)
                az = low_pass_filter(az, cutoff_acc, fs_acc)

            # Seleciona eixo para eventos
            axis_map = {"ax": ax, "ay": ay, "az": az}
            sig = axis_map[axis_acc]

            # Detecção de mínimos
            pk_kwargs_acc = {}
            if prominence_acc > 0: pk_kwargs_acc["prominence"] = float(prominence_acc)
            if min_distance_samples_acc > 1: pk_kwargs_acc["distance"] = int(min_distance_samples_acc)
            peaks_acc, _ = find_peaks(-sig, **pk_kwargs_acc)

            # Onsets/Offsets no mesmo eixo escolhido
            onsets_acc, offsets_acc = [], []
            for p in peaks_acc:
                for j in range(p, 1, -1):
                    if sig[j] > sig[j-1]:
                        onsets_acc.append(j); break
                for j in range(p, len(sig)-1):
                    if sig[j] > sig[j+1]:
                        offsets_acc.append(j); break

            ncycles_acc = min(len(onsets_acc), len(offsets_acc))

            # ===== Ajustes finos — Aceleração (por ciclo e por pico) =====
            st.sidebar.subheader("Ajustes finos — Aceleração")
            sel_cycle_acc = st.sidebar.number_input(
                "Ciclo (acc) 0-index", min_value=0, max_value=max(ncycles_acc-1,0), value=0, step=1
            )
            def _get_acc(store, idx): return float(store.get(idx, 0.0))
            d_on_acc = st.sidebar.number_input("Δ Onset (acc) (s)", -2.0, 2.0, _get_acc(st.session_state.adj_onset_acc, sel_cycle_acc), 0.01)
            d_off_acc = st.sidebar.number_input("Δ Offset (acc) (s)", -2.0, 2.0, _get_acc(st.session_state.adj_offset_acc, sel_cycle_acc), 0.01)
            st.session_state.adj_onset_acc[sel_cycle_acc] = d_on_acc
            st.session_state.adj_offset_acc[sel_cycle_acc] = d_off_acc

            if len(peaks_acc) > 0:
                sel_peak_acc = st.sidebar.number_input("Pico (mínimo) acc 0-index", min_value=0, max_value=len(peaks_acc)-1, value=0, step=1)
                d_pk_acc = st.sidebar.number_input("Δ Mínimo (acc) (s)", -2.0, 2.0,
                                                   float(st.session_state.adj_peak_acc.get(sel_peak_acc, 0.0)), 0.01)
                st.session_state.adj_peak_acc[sel_peak_acc] = d_pk_acc

            cr3, cr4 = st.sidebar.columns(2)
            if cr3.button("Reset ciclo (acc)"):
                st.session_state.adj_onset_acc.pop(sel_cycle_acc, None)
                st.session_state.adj_offset_acc.pop(sel_cycle_acc, None)
            if cr4.button("Reset tudo (acc)"):
                st.session_state.adj_onset_acc.clear()
                st.session_state.adj_offset_acc.clear()
                st.session_state.adj_peak_acc.clear()

            # Tempos base e ajustados
            onset_acc_t = [t_acc[i] for i in onsets_acc[:ncycles_acc]]
            offset_acc_t = [t_acc[i] for i in offsets_acc[:ncycles_acc]]
            peaks_acc_t = [t_acc[i] for i in peaks_acc]

            onset_acc_adj = [clamp(v + st.session_state.adj_onset_acc.get(i,0.0), t_min_acc, t_max_acc) for i,v in enumerate(onset_acc_t)]
            offset_acc_adj = [clamp(v + st.session_state.adj_offset_acc.get(i,0.0), t_min_acc, t_max_acc) for i,v in enumerate(offset_acc_t)]
            peaks_acc_adj = [clamp(v + st.session_state.adj_peak_acc.get(i,0.0), t_min_acc, t_max_acc) for i,v in enumerate(peaks_acc_t)]

            # PLOTS — sinal do eixo escolhido + marcações
            figA, axA = plt.subplots(figsize=(10,4))
            axA.plot(t_acc, sig, 'k-', label=f"{axis_acc}")
            for i in range(ncycles_acc):
                on, of = onset_acc_adj[i], offset_acc_adj[i]
                axA.axvline(on, ls='--', color='orange', label='Início' if i==0 else "")
                axA.axvline(of, ls='--', color='green',  label='Fim' if i==0 else "")
                axA.axvspan(on, of, color='gray', alpha=0.3, label='Teste' if i==0 else "")
            for k, tp in enumerate(peaks_acc_adj):
                axA.axvline(tp, ls='--', color='blue', label='Mínimos' if k==0 else "")
            axA.set_xlabel("Tempo (s)"); axA.set_ylabel("Aceleração (unid.)")
            axA.legend(loc="lower left"); st.pyplot(figA)

            # Tabela de tempos por ciclo (acc) + download
            rows_acc = []
            for i in range(ncycles_acc):
                t_on, t_off = onset_acc_adj[i], offset_acc_adj[i]
                t_min_ciclo = first_min_within(peaks_acc_adj, t_on, t_off)
                rows_acc.append({"ciclo": i, "onset_s": t_on, "offset_s": t_off, "minimo_s": t_min_ciclo})
            df_tempos_acc = pd.DataFrame(rows_acc)
            st.subheader("Tempos por ciclo — Aceleração")
            st.dataframe(df_tempos_acc, use_container_width=True)
            st.download_button(
                "Baixar CSV (Aceleração)",
                df_tempos_acc.to_csv(index=False).encode("utf-8"),
                file_name="tempos_ciclos_aceleracao.csv",
                mime="text/csv"
            )

            # Plots auxiliares dos 3 eixos
            figE, axE = plt.subplots(figsize=(10,2))
            axE.plot(t_acc, ax, label="ax"); axE.plot(t_acc, ay, label="ay"); axE.plot(t_acc, az, label="az")
            axE.set_xlabel("Tempo (s)"); axE.set_ylabel("Aceleração"); axE.legend(loc="lower left")
            st.pyplot(figE)
        else:
            st.info("Faça upload do arquivo de aceleração na barra lateral.")

    with tab_map["Angular velocity"]:
        st.write("Conteúdo de Angular velocity (a definir).")
