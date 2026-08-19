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
    st.session_state.exp_peaks = pd.DataFrame([
        {"Shift (ppm)": 1.26, "Molteplicità": "t", "Integrale": 3.0},
        {"Shift (ppm)": 2.04, "Molteplicità": "s", "Integrale": 3.0},
        {"Shift (ppm)": 4.12, "Molteplicità": "q", "Integrale": 2.0}
    ])

# --- ARCHITETTURA OOP: SPIN SYSTEM & DYNAMICS ---
class Nucleus:
    def __init__(self, atom_idx, element, shift_base, chem_eq_class, is_exch, attached_c):
        self.id = atom_idx
        self.element = element
        self.shift = shift_base 
        self.chem_eq = chem_eq_class
        self.mag_eq = None
        self.is_exchangeable = is_exch
        self.attached_c = attached_c
        self.couplings = {}

class SpinSystemEngine:
    def __init__(self, mol_h, freq_mhz, temperature):
        self.mol = mol_h
        self.freq = freq_mhz
        self.temperature = temperature
        self.nuclei = {}
        self.couplings = []
        self.debug_log = []
        self._build_engine()

    def _stima_shift_base(self, atom):
        c_atom = atom.GetNeighbors()[0]
        if c_atom.GetAtomicNum() in [7, 8, 16]:
            if c_atom.GetAtomicNum() == 8: return 11.0 if any(b.GetBondType() == Chem.BondType.DOUBLE for b in c_atom.GetBonds()) else 4.0
            elif c_atom.GetAtomicNum() == 7: return 2.5
            elif c_atom.GetAtomicNum() == 16: return 1.5
        if c_atom.GetAtomicNum() != 6: return 2.0
        if c_atom.GetIsAromatic(): return 7.3
        elif c_atom.GetHybridization() == Chem.HybridizationType.SP2:
            return 9.8 if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c_atom).GetAtomicNum() == 8 for b in c_atom.GetBonds()) else 5.3
        elif c_atom.GetHybridization() == Chem.HybridizationType.SP: return 2.5
        
        num_H = sum(1 for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 1)
        shift = {3: 0.9, 2: 1.2, 1: 1.5}.get(num_H, 1.5)
        
        for neighbor in c_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1: continue
            atomic_num = neighbor.GetAtomicNum()
            if atomic_num == 6:
                if neighbor.GetIsAromatic(): shift += 1.5
                elif neighbor.GetHybridization() == Chem.HybridizationType.SP2:
                    shift += 1.0 if any(b.GetOtherAtom(neighbor).GetAtomicNum() == 8 for b in neighbor.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE) else 0.8
                elif neighbor.GetHybridization() == Chem.HybridizationType.SP: shift += 0.9
                for beta in neighbor.GetNeighbors():
                    if beta.GetIdx() == c_atom.GetIdx() or beta.GetAtomicNum() == 1: continue
                    b_atomic_num = beta.GetAtomicNum()
                    if b_atomic_num == 8: shift += 0.2
                    elif b_atomic_num in [9, 17, 35, 53]: shift += 0.3
                    elif b_atomic_num == 6 and beta.GetHybridization() == Chem.HybridizationType.SP2:
                        if any(b.GetOtherAtom(beta).GetAtomicNum() == 8 for b in beta.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE): shift += 0.2
            elif atomic_num == 8: shift += 3.0 if any(b.GetBondType() == Chem.BondType.DOUBLE for b in neighbor.GetBonds()) else 2.5
            elif atomic_num == 7: shift += 1.5
            elif atomic_num == 9: shift += 3.0
            elif atomic_num == 17: shift += 2.2
            elif atomic_num == 35: shift += 2.1
            elif atomic_num == 53: shift += 1.7
            elif atomic_num == 16: shift += 1.2
        return shift

    def _build_engine(self):
        ranks = list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))
        shifts_visti = []
        amide_matches = self.mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3](C)(C)"))
        amide_methyl_carbons = [m for match in amide_matches for m in (match[3], match[4])] if amide_matches else []
        
        R, kB, h = 8.314, 1.38e-23, 6.626e-34
        T_K = self.temperature + 273.15
        k_exchange = (kB * T_K / h) * np.exp(-75000 / (R * T_K))

        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                idx = atom.GetIdx()
                c_idx = atom.GetNeighbors()[0].GetIdx()
                is_exch = atom.GetNeighbors()[0].GetAtomicNum() in [7, 8, 16]
                shift = self._stima_shift_base(atom)
                
                if c_idx in amide_methyl_carbons:
                    if k_exchange > 1000:
                        shift = 2.9
                    else:
                        shift = 2.8 if c_idx == amide_methyl_carbons[0] else 3.0

                while any(abs(shift - sv) < 0.05 for sv in shifts_visti): shift += 0.1
                shifts_visti.append(shift)
                
                r = "dynamic_avg" if c_idx in amide_methyl_carbons and k_exchange > 1000 else ranks[idx]
                self.nuclei[idx] = Nucleus(idx, '1H', shift, r, is_exch, c_idx + 1)

        h_ids = list(self.nuclei.keys())
        for i in range(len(h_ids)):
            for j in range(i + 1, len(h_ids)):
                n1, n2 = h_ids[i], h_ids[j]
                if self.nuclei[n1].is_exchangeable or self.nuclei[n2].is_exchangeable: continue
                path = Chem.GetShortestPath(self.mol, n1, n2)
                plen = len(path) - 1
                j_val = 12.0 if plen == 2 else (7.5 if plen == 3 else (2.0 if plen == 4 and any(self.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in path) else 0.0))
                if j_val > 0:
                    self.nuclei[n1].couplings[n2] = self.nuclei[n2].couplings[n1] = j_val

        chem_groups = {}
        for nuc in self.nuclei.values(): chem_groups.setdefault(nuc.chem_eq, []).append(nuc)

        mag_eq_counter = 0
        for eq_class, nucs in chem_groups.items():
            if len(nucs) == 1:
                nucs[0].mag_eq = mag_eq_counter
                mag_eq_counter += 1
                continue
            mag_groups = {}
            for nuc in nucs:
                sig = tuple(sorted((self.nuclei[tid].chem_eq, jv) for tid, jv in nuc.couplings.items() if self.nuclei[tid].chem_eq != eq_class))
                mag_groups.setdefault(sig, []).append(nuc)
            for sig, m_nucs in mag_groups.items():
                for mn in m_nucs: mn.mag_eq = mag_eq_counter
                mag_eq_counter += 1

    def get_signals_for_ui(self):
        signals = []
        gruppi_mag = {}
        for nuc in self.nuclei.values(): gruppi_mag.setdefault(nuc.mag_eq, []).append(nuc)
            
        for mag_class, nucs in gruppi_mag.items():
            rep = nucs[0]
            integral = len(nucs)
            if rep.is_exchangeable:
                signals.append(self._format_signal(rep, integral, nucs, 'br s', [], [], None))
                continue

            j_vicini = []
            coupled_nuclei = []
            for target_id, j_val in rep.couplings.items():
                if self.nuclei[target_id].mag_eq != mag_class: 
                    j_vicini.append(j_val)
                    coupled_nuclei.append(self.nuclei[target_id])
            j_vicini.sort(reverse=True)

            roofing_params = None
            if rep.chem_eq == rep.mag_eq:
                for target_nuc in coupled_nuclei:
                    j_val = rep.couplings[target_nuc.id]
                    delta_nu = abs(rep.shift - target_nuc.shift) * self.freq
                    ratio = delta_nu / j_val if j_val > 0 else 999
                    
                    if 0 < ratio < 10:
                        C = np.sqrt(delta_nu**2 + j_val**2)
                        roofing_params = {'C': C, 'inner': 1 + j_val/C, 'outer': 1 - j_val/C, 'is_higher_freq': rep.shift > target_nuc.shift}
                        break

            counts = {}
            for jv in j_vicini: counts[jv] = counts.get(jv, 0) + 1
            
            tree_chars, tree_js = [], []
            for jv, num in counts.items():
                tree_chars.append({1:'d', 2:'t', 3:'q'}.get(num, 'm'))
                tree_js.append(jv)
                
            mult = 's' if not tree_chars else ('m' if 'm' in tree_chars or sum(counts.values()) > 6 else "".join(tree_chars))

            signals.append(self._format_signal(rep, integral, nucs, mult, tree_chars, tree_js, roofing_params))
            
        return signals

    def _format_signal(self, rep, integral, nucs, mult, tree_chars, tree_js, roofing_params):
        sig = {'delta': rep.shift, 'multiplicity': mult, 'integral': integral, 'is_exchangeable': rep.is_exchangeable}
        flat_j_vals = []
        for c, jv in zip(tree_chars, tree_js):
            flat_j_vals.extend([jv] * {'d':1, 't':2, 'q':3, 'm':4}.get(c, 1))
        sig['sub_peaks'] = self._genera_sotto_picchi(sig['delta'], mult, float(integral), self.freq, flat_j_vals, roofing_params)
        return sig

    def _genera_sotto_picchi(self, center, mult, integral, freq, flat_j_vals, roofing_params):
        if mult in ['s', 'br s']: return [(center, integral)]
        if mult == 'm':
            j_std = 7.5 / freq
            return [(center + o, r * integral) for o, r in zip(np.linspace(-1.5*j_std, 1.5*j_std, 5), [0.1, 0.25, 0.3, 0.25, 0.1])]

        def ottieni_offset(carattere, j_val_hz):
            j_ppm = j_val_hz / freq
            if carattere == 'd': return [-j_ppm/2, j_ppm/2], [0.5, 0.5]
            elif carattere == 't': return [-j_ppm, 0, j_ppm], [0.25, 0.5, 0.25]
            elif carattere == 'q': return [-1.5*j_ppm, -0.5*j_ppm, 0.5*j_ppm, 1.5*j_ppm], [0.125, 0.375, 0.375, 0.125]
            return [0.0], [1.0]

        picchi = [(center, integral)]
        for i, c in enumerate([ch for ch in mult if ch in 'dtq']):
            j = flat_j_vals[i] if i < len(flat_j_vals) else 7.5
            nuovi_picchi = []
            off, rat = ottieni_offset(c, j)
            if roofing_params and c == 'd' and i == 0:
                rat = [roofing_params['inner']/2, roofing_params['outer']/2] if roofing_params['is_higher_freq'] else [roofing_params['outer']/2, roofing_params['inner']/2]
                j_ppm, c_ppm = j / freq, roofing_params['C'] / freq
                off = [-(c_ppm - j_ppm)/2, (c_ppm + j_ppm)/2] if roofing_params['is_higher_freq'] else [-(c_ppm + j_ppm)/2, (c_ppm - j_ppm)/2]
            for p_shift, p_int in picchi:
                for o, r in zip(off, rat): nuovi_picchi.append((p_shift + o, p_int * r))
            picchi = nuovi_picchi
        return picchi


# --- FUNZIONI DI CALCOLO SPETTRALE UI ---

def genera_spettro_sperimentale(peaks_df, x_ppm, freq=400.0):
    """Genera lo spettro dai dati inseriti usando la lorentziana originale dell'app."""
    y = np.zeros_like(x_ppm)
    gamma = 0.0025 * (500.0 / freq)
    
    for _, row in peaks_df.iterrows():
        shift = row["Shift (ppm)"]
        integ = row["Integrale"]
        mult = row["Molteplicità"]
        
        if pd.isna(shift) or pd.isna(integ) or pd.isna(mult):
            continue
            
        j_hz = 7.5
        j_ppm = j_hz / freq
        
        if mult == 's' or mult == 'br s': offsets, ratios = [0], [1]
        elif mult == 'd': offsets, ratios = [-j_ppm/2, j_ppm/2], [0.5, 0.5]
        elif mult == 't': offsets, ratios = [-j_ppm, 0, j_ppm], [0.25, 0.5, 0.25]
        elif mult == 'q': offsets, ratios = [-1.5*j_ppm, -0.5*j_ppm, 0.5*j_ppm, 1.5*j_ppm], [0.125, 0.375, 0.375, 0.125]
        else: offsets, ratios = [0], [1] 
        
        gamma_app = max(0.06, gamma) if mult == 'br s' else gamma
        
        for off, rat in zip(offsets, ratios):
            # Stessa formula di rendering del tuo codice principale per coerenza visiva
            y += (integ * rat) / (1.0 + ((x_ppm - (shift + off)) / gamma_app)**2)
    return y

def genera_spettro_predetto_oop(smiles, x_ppm, freq=400.0, temp=25):
    """Richiama il motore SpinSystemEngine e converte in traccia plot."""
    y = np.zeros_like(x_ppm)
    if not smiles: return y
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return y
    mol_h = Chem.AddHs(mol)
    
    try:
        engine = SpinSystemEngine(mol_h, freq, temp)
        signals = engine.get_signals_for_ui()
        gamma = 0.0025 * (500.0 / freq)
        
        for sig in signals:
            gamma_app = max(0.06, gamma) if sig.get('is_exchangeable', False) else gamma
            for p_shift, p_int in sig['sub_peaks']:
                y += p_int / (1.0 + ((x_ppm - p_shift) / gamma_app)**2)
    except Exception as e:
        st.error(f"Errore nella stima strutturale: {e}")
        
    return y

def analizza_frammenti(peaks_df):
    """Analizza i picchi e suggerisce sottostrutture ignorando le righe vuote."""
    suggerimenti = []
    for _, row in peaks_df.iterrows():
        shift = row["Shift (ppm)"]
        integ = row["Integrale"]
        mult = row["Molteplicità"]
        
        if pd.isna(shift) or pd.isna(integ) or pd.isna(mult):
            continue
            
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

col_draw, col_plot = st.columns([0.5, 0.5])

with col_draw:
    st.markdown("**Disegna la tua ipotesi strutturale:**")
    smiles_ipotesi = st_ketcher("CCO", height=500) 
    
with col_plot:
    x_ppm = np.linspace(-0.5, 12.5, 2000)
    
    # 1. Calcolo Spettro Sperimentale (dalla tabella)
    y_exp = genera_spettro_sperimentale(edited_df, x_ppm, freq=400.0)
    
    # 2. Calcolo Spettro Predetto (motore ad oggetti)
    y_pred = genera_spettro_predetto_oop(smiles_ipotesi, x_ppm, freq=400.0, temp=25)
    
    fig = go.Figure()
    
    # Traccia Sperimentale
    fig.add_trace(go.Scatter(
        x=x_ppm, y=y_exp, 
        mode='lines', name='Sperimentale (Tabella)',
        line=dict(color='black', width=2),
        fill='tozeroy', fillcolor='rgba(0,0,0,0.1)'
    ))
    
    # Traccia Predetta (Ketcher)
    if smiles_ipotesi:
        fig.add_trace(go.Scatter(
            x=x_ppm, y=y_pred, 
            mode='lines', name='Predetto (Motore OOP)',
            line=dict(color=BORDEAUX, width=2, dash='dash'),
            fill='tozeroy', fillcolor='rgba(107, 20, 34, 0.2)'
        ))
        
    fig.update_layout(
        xaxis_title="Chemical Shift δ (ppm)",
        yaxis_title="Intensità Relativa",
        xaxis=dict(autorange="reversed"),
        plot_bgcolor='white',
        hovermode='x',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', range=[12.5, -0.5])
    fig.update_yaxes(showgrid=False, showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)
