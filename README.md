# Flow Assure — IPR Curve Generator

**Flow Assure** is an interactive web application for petroleum engineering students and professionals to generate **Inflow Performance Relationship (IPR) curves** using industry-standard correlations.

Enter well/reservoir/fluid data → auto-calculate PVT properties → generate IPR curves → export results.

---

## Features

### IPR Models
| Model | Equation | Best For |
|-------|----------|----------|
| **Simple Darcy / PI** | `q = J(Pr − Pwf)` | Undersaturated, single-phase oil |
| **Vogel** | `q/qmax = 1 − 0.2(Pwf/Pr) − 0.8(Pwf/Pr)²` | Saturated solution-gas-drive |
| **Composite** | Darcy above Pb + Vogel below Pb | Undersaturated reservoirs |
| **Standing (FE)** | Vogel with flow efficiency correction | Damaged / stimulated wells |
| **Fetkovich** | `q = C(Pr² − Pwf²)ⁿ` | Back-pressure type wells |

### PVT Correlations
| Property | Correlations Available |
|----------|----------------------|
| Solution GOR (Rs) | Standing, Vasquez-Beggs |
| Bubble Point (Pb) | Standing, Vasquez-Beggs |
| Oil FVF (Bo) | Standing, Vasquez-Beggs |
| Oil Viscosity (μo) | Beggs-Robinson |
| Gas Z-factor | Dranchuk-Abou-Kassem |
| Gas Viscosity (μg) | Lee-Gonzalez-Eakin |
| Gas FVF (Bg) | Direct calculation |
| Water Properties | McCain |

### Additional Features
- **Compare** all IPR methods on a single chart
- **Sensitivity analysis** for Pr, Pb, skin, FE, and Fetkovich n
- **Export** IPR data (CSV), plots (PNG), and calculation summaries (TXT)
- **Smart warnings** for invalid inputs and correlation validity limits
- **Source tagging** — every calculated value shows whether it's user-input, correlation-derived, or default

---

## Quick Start

### Prerequisites
- Python 3.9 or later
- pip

### Installation

```bash
# Clone or download the project
cd Flow Assure

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Default Example
The app loads with a pre-filled example case:

| Parameter | Value |
|-----------|-------|
| Pr | 3000 psia |
| Pb | 2000 psia |
| T | 180 °F |
| API | 35 °API |
| Gas SG | 0.75 |
| qtest | 800 STB/day |
| Pwf_test | 1500 psia |
| k | 50 md |
| h | 40 ft |
| re | 1000 ft |
| rw | 0.328 ft |
| Skin | 0 |

---

## Project Structure

```
Flow Assure/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── backend/
│   ├── __init__.py
│   ├── pvt.py                # PVT correlations (Rs, Pb, Bo, μo, Z, Bg, μg)
│   ├── ipr.py                # IPR models (Darcy, Vogel, Composite, Standing, Fetkovich)
│   ├── validation.py         # Input validation and warning system
│   ├── plots.py              # Plotly interactive chart builders
│   └── utils.py              # Unit conversions, constants, helpers
├── data/
│   └── sample_cases.csv      # Example well input cases
└── tests/
    ├── __init__.py
    ├── test_pvt.py            # PVT correlation tests
    └── test_ipr.py            # IPR model tests
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests verify:
- Physical reasonability of all PVT correlations
- Boundary conditions (q=0 at Pwf=Pr, q=qmax at Pwf=0)
- Composite IPR continuity at bubble point
- Standing FE=1 ≡ Vogel equivalence
- Roundtrip consistency (Rs → Pb → Rs)
- Manual override behavior

---

## Equations Reference

### Darcy (Productivity Index)
```
q = J × (Pr − Pwf)
J = kh / [141.2 × μo × Bo × (ln(re/rw) − 0.75 + s)]
```

### Vogel
```
q = qmax × [1 − 0.2×(Pwf/Pr) − 0.8×(Pwf/Pr)²]
qmax = qtest / [1 − 0.2×(Pwf_test/Pr) − 0.8×(Pwf_test/Pr)²]
```

### Composite
```
Above Pb:  q = J × (Pr − Pwf)
Below Pb:  q = qb + (J×Pb/1.8) × [1 − 0.2×(Pwf/Pb) − 0.8×(Pwf/Pb)²]
where qb = J × (Pr − Pb)
```

### Standing (Flow Efficiency)
```
Pwf' = Pr − FE × (Pr − Pwf)
q = qmax_ideal × [1 − 0.2×(Pwf'/Pr) − 0.8×(Pwf'/Pr)²]
```

### Fetkovich
```
q = C × (Pr² − Pwf²)ⁿ
qmax = C × Pr^(2n)
```

---

## Assumptions & Limitations

1. All PVT correlations are **empirical** — they have validity limits.
2. IPR curves assume **steady-state or pseudo-steady-state** flow.
3. Results are **engineering estimates** — not replacements for calibrated reservoir models.
4. Gas properties use the **Dranchuk-Abou-Kassem** Z-factor correlation (sweet gas assumed).
5. No compositional PVT modeling — black-oil approach only.

---

## License

Educational project — use freely for learning and teaching.

---

*Built with Streamlit, Plotly, NumPy, and Pandas.*
