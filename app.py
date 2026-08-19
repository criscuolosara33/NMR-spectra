import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_ketcher import st_ketcher
from rdkit import Chem

# --- CONFIGURAZIONE ESTETICA E SETUP ---
st.set_page_config(page_title="Simulatore Esame Organica 3", layout="wide")
BORDEAUX = '#6B1422'
BORDEAUX_HOVER = '#822433'

st.markdown(f"""
<style>
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, table, th, td {{
        font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif !important;
    }}
    div.stButton > button:first-child {{ 
        background-color: {BORDEAUX}; color: white; border: none; border-radius: 4px; font-weight: bold; 
    }}
    div.stButton > button:hover {{ background-color: {BORDEAUX_HOVER}; }}
    .stTextArea textarea {{ font-family: 'Palatino', serif !important; font-size: 16px; }}
</style>
""", unsafe_allow_html=True)

# Inizializzazione stati per le tabelle
if "exp_1h" not in st.session_state:
    st.session_state.exp_1h = pd.DataFrame([{"Shift (ppm)": None, "Molteplicità": "s", "Integrale": None}])
if "exp_13c" not in st.session_state:
    st.session_state.exp_13c = pd.DataFrame([{"Shift (ppm)": None, "Tipo (DEPT)": "Cq"}])
if "exp_ir" not in st.session_state:
    st.session_state.exp_ir = pd.DataFrame([{"Frequenza (cm-1)": None, "Intensità": "Forte", "Forma": "Stretta"}])

st.title("Esame di Chimica Organica 3: Elucidazione Strutturale")

# --- ARCHITETTURA A TAB ---
tab1, tab2, tab3 = st.tabs(["📊 Dati Sperimentali", "🧩 Tavolo di Lavoro", "⚖️ The 32nd Evaluation"])

# ==========================================
# TAB 1: INSERIMENTO DATI MULTITECNICA
# ==========================================
with tab1:
    st.markdown("### Profilo Analitico del Composto Incognito")
    
    col_ms, col_uv = st.columns(2)
    with col_ms:
        with st.expander("Spettrometria di Massa (MS)", expanded=True):
            st.number_input("Ione Molecolare M+ (m/z)", min_value=1.0, format="%.4f")
            st.text_input("Picchi isotopici rilevanti (es. M+2, M+4)")
            st.text_input("Formula Bruta ipotizzata (es. C8H8O2)")
    
    with col_uv:
        with st.expander("Spettroscopia UV-Vis", expanded=True):
            st.number_input("Lambda max (nm)", min_value=100, max_value=800, value=254)
            st.number_input("Assorbività molare (ε)", min_value=0)
            
    with st.expander("Spettroscopia IR (Infrarosso)", expanded=True):
        st.data_editor(
            st.session_state.exp_ir, num_rows="dynamic", use_container_width=True,
            column_config={
                "Frequenza (cm-1)": st.column_config.NumberColumn(min_value=400, max_value=4000),
                "Intensità": st.column_config.SelectboxColumn(options=["Forte", "Media", "Debole"]),
                "Forma": st.column_config.SelectboxColumn(options=["Stretta", "Allargata"])
            }
        )

    col_1h, col_13c = st.columns(2)
    with col_1h:
        with st.expander("Dati 1H-NMR", expanded=True):
            st.data_editor(
                st.session_state.exp_1h, num_rows="dynamic", use_container_width=True,
                column_config={
                    "Shift (ppm)": st.column_config.NumberColumn(format="%.2f"),
                    "Molteplicità": st.column_config.SelectboxColumn(options=["s", "d", "t", "q", "m", "br s"]),
                    "Integrale": st.column_config.NumberColumn(format="%.1f")
                }
            )
            
    with col_13c:
        with st.expander("Dati 13C-NMR & DEPT", expanded=True):
            st.data_editor(
                st.session_state.exp_13c, num_rows="dynamic", use_container_width=True,
                column_config={
                    "Shift (ppm)": st.column_config.NumberColumn(format="%.1f"),
                    "Tipo (DEPT)": st.column_config.SelectboxColumn(options=["Cq", "CH", "CH2", "CH3"])
                }
            )

# ==========================================
# TAB 2: ASSEMBLAGGIO E DEDUZIONE
# ==========================================
with tab2:
    st.markdown("### Area di Assemblaggio Strutturale")
    st.caption("Usa questo spazio per manipolare i frammenti chimici e redigere il ragionamento logico dell'esame.")
    
    col_draw, col_text = st.columns([0.5, 0.5])
    
    with col_draw:
        st.markdown("**Ipotesi Strutturale**")
        smiles_ipotesi = st_ketcher("C", height=500)
        
    with col_text:
        st.markdown("**Diario di Bordo (Ragionamento Logico)**")
        ragionamento = st.text_area(
            "Giustifica ogni assegnazione. Inizia dai gruppi funzionali primari (IR/13C) e assembla lo scheletro carbonioso (1H/DEPT).", 
            height=435, 
            placeholder="Es. Lo spettro IR mostra una banda forte a 1715 cm-1 indicativa di un carbonile chetonico. Il 13C-NMR conferma con un segnale a 210 ppm..."
        )

# ==========================================
# TAB 3: THE 32ND EVALUATION
# ==========================================
with tab3:
    st.markdown("### The 32nd Evaluation: Validazione Strutturale Inversa")
    st.info("Questo modulo calcola lo spettro teorico della molecola disegnata e lo confronta con i parametri sperimentali inseriti per verificarne la validità.")
    
    if smiles_ipotesi and smiles_ipotesi != "C":
        try:
            mol = Chem.MolFromSmiles(smiles_ipotesi)
            mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            mol_weight = Chem.Descriptors.MolWt(mol)
            
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Formula Calcolata", mol_formula)
            col_kpi2.metric("Massa Calcolata", f"{mol_weight:.2f} g/mol")
            
            # Qui andrà calcolato il vero DBE e confrontato
            col_kpi3.metric("Congruenza Insaturazioni (DBE)", "In attesa di calcolo")
            
            st.markdown("#### Sovrapposizione 1H-NMR")
            # Placeholder grafico temporaneo
            fig = go.Figure()
            fig.update_layout(
                xaxis_title="Chemical Shift δ (ppm)", yaxis_title="Intensità Relativa",
                xaxis=dict(autorange="reversed", range=[12.5, -0.5]), height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Genera Report d'Esame (PDF)", use_container_width=True):
                st.success("Funzione di esportazione PDF da implementare nella prossima iterazione.")
                
        except Exception as e:
            st.error("Disegna una struttura valida per attivare la valutazione.")
    else:
        st.warning("Disegna una molecola nel Tavolo di Lavoro per attivare The 32nd Evaluation.")

