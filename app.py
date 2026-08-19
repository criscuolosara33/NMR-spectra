import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_ketcher import st_ketcher
from rdkit import Chem

# --- CONFIGURAZIONE E STILI ESTETICI ---
st.set_page_config(page_title="NMR Elucidation Studio", layout="wide")

BORDEAUX = '#6B1422'
BORDEAUX_HOVER = '#822433'

st.markdown(f"""
<style>
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, table, th, td {{
        font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif !important;
    }}
    div.stButton > button:first-child {{ 
        background-color: {BORDEAUX}; color: white; border: none; border-radius: 4px; font-weight: bold; transition: all 0.2s ease-in-out;
    }}
    div.stButton > button:hover {{ 
        background-color: {BORDEAUX_HOVER}; color: white;
    }}
    .fragment-box {{
        background-color: #f8f9fa; border-left: 5px solid {BORDEAUX}; padding: 10px 15px; margin-bottom: 10px; border-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)

# --- INIZIALIZZAZIONE STATO ---
if "exp_peaks" not in st.session_state:
    # Pre-popolato con lo spettro dell'Etil acetato come esempio
    st.session_state.exp_peaks = pd.DataFrame([
        {"Shift (ppm)": 1.26, "Molteplicità": "t", "Integrale": 3.0},
        {"Shift (ppm)": 2.04, "Molteplicità": "s", "Integrale": 3.0},
        {"Shift (ppm)": 4.12, "Molteplicità": "q", "Integrale": 2.0}
    ])

# --- FUNZIONI DI CALCOLO E DIAGNOSTICA ---
def lorentziana(x, x0, area, gamma=0.03):
    """Genera una curva lorentziana per simulare un picco NMR."""
    return (area * gamma / np.pi) / ((x - x0)**2 + gamma**2)

def genera_spettro(peaks_df, x_ppm):
    """Genera lo spettro dai dati inseriti (assumendo pattern di splitting ideali)."""
    y = np.zeros_like(x_ppm)
    for _, row in peaks_df.iterrows():
        shift = row["Shift (ppm)"]
        integ = row["Integrale"]
        mult = row["Molteplicità"]
        
        # Semplificazione degli splitting per la visualizzazione
        j_hz_ppm = 7.0 / 400.0 
        
        if mult == 's': offsets, ratios = [0], [1]
        elif mult == 'd': offsets, ratios = [-j_hz_ppm/2, j_hz_ppm/2], [0.5, 0.5]
        elif mult == 't': offsets, ratios = [-j_hz_ppm, 0, j_hz_ppm], [0.25, 0.5, 0.25]
        elif mult == 'q': offsets, ratios = [-1.5*j_hz_ppm, -0.5*j_hz_ppm, 0.5*j_hz_ppm, 1.5*j_hz_ppm], [0.125, 0.375, 0.375, 0.125]
        else: offsets, ratios = [0], [1] # Fallback per m
            
        for off, rat in zip(offsets, ratios):
            y += lorentziana(x_ppm, shift + off, integ * rat)
    return y

def stima_spettro_predetto(smiles, x_ppm):
    """Motore di stima NMR 1H semplificato per la validazione incrociata."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return np.zeros_like(x_ppm)
    mol = Chem.AddHs(mol)
    
    y = np.zeros_like(x_ppm)
    shifts_visti = []
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            c_atom = atom.GetNeighbors()[0]
            # Stima di base molto grezza
            shift = 1.0
            if c_atom.GetAtomicNum() == 8: shift = 4.0
            elif c_atom.GetAtomicNum() == 6:
                if c_atom.GetIsAromatic(): shift = 7.3
                elif c_atom.GetHybridization() == Chem.HybridizationType.SP2: shift = 5.5
                else:
                    # Effetto induttivo vicini
                    for n in c_atom.GetNeighbors():
                        if n.GetAtomicNum() == 8: shift += 2.5
                        elif n.GetAtomicNum() == 7: shift += 1.5
                        elif n.GetIsAromatic(): shift += 1.5
                        elif n.GetHybridization() == Chem.HybridizationType.SP2 and n.GetAtomicNum() == 6: shift += 1.0
            
            # Dispersione per evitare sovrapposizioni perfette di nuclei non equivalenti
            while any(abs(shift - sv) < 0.02 for sv in shifts_visti): shift += 0.05
            shifts_visti.append(shift)
            
            y += lorentziana(x_ppm, shift, 1.0, gamma=0.04)
    return y

def analizza_frammenti(peaks_df):
    """Analizza i picchi e suggerisce sottostrutture."""
    suggerimenti = []
    for _, row in peaks_df.iterrows():
        shift = row["Shift (ppm)"]
        integ = row["Integrale"]
        mult = row["Molteplicità"]
        
        if 9.0 <= shift <= 10.5 and mult == 's':
            suggerimenti.append(f"**δ {shift}**: Probabile protone aldeidico (H-C=O).")
        elif 10.0 <= shift <= 12.0:
            suggerimenti.append(f"**δ {shift}**: Probabile protone acido (COOH) o fenolico fortemente legato.")
        elif 6.5 <= shift <= 8.5:
            suggerimenti.append(f"**δ {shift}**: Zona aromatica. L'integrale ({integ}) suggerisce il livello di sostituzione dell'anello.")
        elif 4.5 <= shift <= 6.5:
            suggerimenti.append(f"**δ {shift}**: Zona alchenica (C=C-H) o protone anomerico nei carboidrati.")
        elif 3.0 <= shift <= 4.5:
            if mult == 'q' and integ == 2:
                suggerimenti.append(f"**δ {shift}**: Gruppo -CH2- fortemente deschermato (probabilmente legato a Ossigeno, es. etere o estere). Quartetto compatibile con un gruppo etilico adiacente.")
            elif mult == 's' and integ == 3:
                suggerimenti.append(f"**δ {shift}**: Gruppo metossilico (-OCH3).")
            else:
                suggerimenti.append(f"**δ {shift}**: Protone in alfa a un eteroatomo (O, N, Alogeno).")
        elif 1.8 <= shift <= 2.8:
            suggerimenti.append(f"**δ {shift}**: Protone in alfa a un gruppo carbonilico (C=O) o anello aromatico (benzilico).")
        elif 0.5 <= shift <= 1.8:
            if mult == 't' and integ == 3:
                suggerimenti.append(f"**δ {shift}**: Gruppo metilico terminale (-CH3) adiacente a un -CH2-.")
            else:
                suggerimenti.append(f"**δ {shift}**: Catena alifatica standard.")
    return suggerimenti

# --- LAYOUT UI ---
st.title("NMR Elucidation Studio")

col_input, col_fragments = st.columns([0.4, 0.6])

with col_input:
    st.markdown("### 1. Dati Sperimentali 1H-NMR")
    st.caption("Inserisci i picchi rilevati dallo spettro incognito.")
    
    edited_df = st.data_editor(
        st.session_state.exp_peaks,
        num_rows="dynamic",
        column_config={
            "Shift (ppm)": st.column_config.NumberColumn("Shift (ppm)", min_value=-1.0, max_value=15.0, format="%.2f"),
            "Molteplicità": st.column_config.SelectboxColumn("Molteplicità", options=["s", "d", "t", "q", "m", "br s"]),
            "Integrale": st.column_config.NumberColumn("Integrale", min_value=0.1, max_value=20.0, format="%.1f")
        },
        use_container_width=True
    )

with col_fragments:
    st.markdown("### 2. Analisi Diagnostica")
    if not edited_df.empty:
        frammenti = analizza_frammenti(edited_df)
        if frammenti:
            for fr in frammenti:
                st.markdown(f"<div class='fragment-box'>{fr}</div>", unsafe_allow_html=True)
        else:
            st.info("Aggiungi picchi per ottenere suggerimenti strutturali.")

st.markdown("---")
st.markdown("### 3. Validazione Strutturale (Experimental vs Predicted)")

col_draw, col_plot = st.columns([0.4, 0.6])

with col_draw:
    st.markdown("**Disegna la tua ipotesi strutturale:**")
    smiles_ipotesi = st_ketcher("CCO", height=400) # Preimpostato con etanolo per test
    
with col_plot:
    x_ppm = np.linspace(-0.5, 12.5, 2000)
    
    # 1. Calcolo Spettro Sperimentale (dalla tabella)
    y_exp = genera_spettro(edited_df, x_ppm)
    
    # 2. Calcolo Spettro Predetto (dal disegno)
    y_pred = stima_spettro_predetto(smiles_ipotesi, x_ppm)
    
    fig = go.Figure()
    
    # Traccia Sperimentale (Tabella)
    fig.add_trace(go.Scatter(
        x=x_ppm, y=y_exp, 
        mode='lines', name='Spettro Sperimentale',
        line=dict(color='black', width=2),
        fill='tozeroy', fillcolor='rgba(0,0,0,0.1)'
    ))
    
    # Traccia Predetta (Ketcher)
    if smiles_ipotesi:
        fig.add_trace(go.Scatter(
            x=x_ppm, y=y_pred, 
            mode='lines', name='Spettro Predetto',
            line=dict(color=BORDEAUX, width=2, dash='dash'),
            fill='tozeroy', fillcolor='rgba(107, 20, 34, 0.2)'
        ))
        
    fig.update_layout(
        xaxis_title="Chemical Shift δ (ppm)",
        yaxis_title="Intensità",
        xaxis=dict(autorange="reversed"), # Convenzione NMR
        plot_bgcolor='white',
        hovermode='x',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', range=[12.5, -0.5])
    fig.update_yaxes(showgrid=False, showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)

