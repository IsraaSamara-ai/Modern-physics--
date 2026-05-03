# ======================================================================
# Modern Physics Interactive - Particle Nature of Light
# الفيزياء الحديثة التفاعلية - الطبيعة الجسيمية للضوء
# إعداد: Israa Youssuf Samara
# ======================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Arc
import matplotlib.patheffects as pe
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="الفيزياء الحديثة - الطبيعة الجسيمية للضوء",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# CUSTOM CSS
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 30px 20px;
        border-radius: 15px;
        text-align: center;
        color: #fff;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 2.2em;
        font-weight: 700;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #f7971e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header h2 {
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.1em;
        font-weight: 300;
        color: #b8c6db;
    }
    .author-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
        padding: 8px 25px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.05em;
        margin-top: 12px;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    .concept-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #3a7bd5;
        margin: 15px 0;
        color: #e0e0e0;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    .explain-box {
        background: linear-gradient(135deg, #1b1b3a, #2d2d5e);
        padding: 18px 22px;
        border-radius: 12px;
        border: 1px solid #3a7bd5;
        margin: 12px 0;
        color: #e8e8e8;
        font-family: 'Noto Sans Arabic', sans-serif;
        line-height: 1.9;
    }
    .result-box {
        background: linear-gradient(135deg, #0a3d0a, #1a5c1a);
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #2ecc71;
        color: #a8f0c8;
        margin: 10px 0;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    .warning-box {
        background: linear-gradient(135deg, #3d0a0a, #5c1a1a);
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #e74c3c;
        color: #f0a8a8;
        margin: 10px 0;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    .formula-box {
        background: #0d1b2a;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #1b4965;
        text-align: center;
        font-size: 1.2em;
        color: #00d4ff;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        direction: ltr;
    }
    .step-indicator {
        display: inline-block;
        background: #3a7bd5;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-left: 8px;
        font-size: 0.9em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        color: #b8c6db;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 0.95em;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #302b63, #24243e) !important;
        color: #00d2ff !important;
        border-bottom: 3px solid #00d2ff;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e, #16213e);
    }
    div[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    .stSlider [data-baseweb="slider"] {
        background: #302b63;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# PHYSICAL CONSTANTS
# ======================================================================
H_PLANCK = 6.63e-34       # Planck's constant (J·s)
C_LIGHT = 3.0e8           # Speed of light (m/s)
E_CHARGE = 1.6e-19        # Electron charge (C)
M_ELECTRON = 9.11e-31     # Electron mass (kg)
K_BOLTZMANN = 1.38e-23    # Boltzmann constant (J/K)
EV_TO_J = 1.6e-19         # eV to Joules conversion
WIEN_B = 2.898e-3         # Wien's displacement constant (m·K)
COMPTON_WL = 2.43e-12     # Compton wavelength (m)

# ======================================================================
# MATERIALS DATABASE
# ======================================================================
MATERIALS = {
    "سيزيوم (Cs)":   {"phi_eV": 2.14, "symbol": "Cs", "color": "#FFD700"},
    "صوديوم (Na)":   {"phi_eV": 2.28, "symbol": "Na", "color": "#C0C0C0"},
    "بوتاسيوم (K)":  {"phi_eV": 2.30, "symbol": "K",  "color": "#D4D4D4"},
    "كالسيوم (Ca)":  {"phi_eV": 2.90, "symbol": "Ca", "color": "#F5F5DC"},
    "ألمنيوم (Al)":  {"phi_eV": 4.00, "symbol": "Al", "color": "#A8A9AD"},
    "نحاس (Cu)":     {"phi_eV": 4.70, "symbol": "Cu", "color": "#B87333"},
    "تنغستون (W)":   {"phi_eV": 4.55, "symbol": "W",  "color": "#71797E"},
    "ذهب (Au)":      {"phi_eV": 5.10, "symbol": "Au", "color": "#FFD700"},
    "بيريليوم (Be)": {"phi_eV": 5.00, "symbol": "Be", "color": "#8E8E8E"},
    "بلاتين (Pt)":   {"phi_eV": 5.65, "symbol": "Pt", "color": "#E5E4E2"},
}

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def planck_radiance(wavelength_m, T):
    """Planck's law: spectral radiance B(λ,T)"""
    a = 2.0 * H_PLANCK * C_LIGHT**2
    b = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZMANN * T)
    with np.errstate(over='ignore', divide='ignore'):
        return a / (wavelength_m**5 * (np.exp(np.clip(b, 0, 500)) - 1.0))


def wavelength_to_rgb(wl_nm):
    """Convert wavelength (nm) to approximate RGB color"""
    w = wl_nm
    R = G = B = 0.0
    if 380 <= w < 440:
        R = -(w - 440) / 60.0; G = 0.0; B = 1.0
    elif 440 <= w < 490:
        R = 0.0; G = (w - 440) / 50.0; B = 1.0
    elif 490 <= w < 510:
        R = 0.0; G = 1.0; B = -(w - 510) / 20.0
    elif 510 <= w < 580:
        R = (w - 510) / 70.0; G = 1.0; B = 0.0
    elif 580 <= w < 645:
        R = 1.0; G = -(w - 645) / 65.0; B = 0.0
    elif 645 <= w <= 780:
        R = 1.0; G = 0.0; B = 0.0
    if 380 <= w < 420:
        f = 0.3 + 0.7 * (w - 380) / 40.0
    elif 645 < w <= 780:
        f = 0.3 + 0.7 * (780 - w) / 135.0
    elif 420 <= w <= 645:
        f = 1.0
    else:
        f = 0.0
    return (R * f, G * f, B * f)


def blackbody_rgb(T):
    """Approximate visible color of a blackbody at temperature T"""
    peak_nm = WIEN_B / T * 1e9
    if peak_nm < 380:
        r, g, b = 0.6, 0.6, 1.0
    elif peak_nm > 780:
        frac = min(1.0, (peak_nm - 780) / 3000.0)
        r, g, b = 1.0, max(0.2, 0.8 - frac * 0.6), max(0.0, 0.3 - frac * 0.3)
    else:
        r, g, b = wavelength_to_rgb(peak_nm)
    brightness = min(1.0, (T / 6000.0) ** 0.5)
    return (min(1, r * brightness), min(1, g * brightness), min(1, b * brightness))


def compute_threshold(phi_eV):
    """Compute threshold frequency and wavelength from work function"""
    f0 = phi_eV * EV_TO_J / H_PLANCK
    lam0 = C_LIGHT / f0
    return f0, lam0


def compute_photoelectric(phi_eV, freq_Hz):
    """Compute photoelectric effect results"""
    f0, lam0 = compute_threshold(phi_eV)
    photon_E_eV = H_PLANCK * freq_Hz / EV_TO_J
    emitting = freq_Hz >= f0
    KE_max_eV = max(0, photon_E_eV - phi_eV) if emitting else 0
    Vs = KE_max_eV if emitting else 0  # stopping potential in Volts (since KEmax = eVs, Vs in V = KEmax in eV)
    return {
        "f0": f0, "lam0": lam0, "photon_E_eV": photon_E_eV,
        "emitting": emitting, "KE_max_eV": KE_max_eV, "Vs": Vs
    }


def compute_compton(lambda_i_m, theta_rad):
    """Compute Compton scattering results"""
    dlambda = COMPTON_WL * (1 - np.cos(theta_rad))
    lambda_f = lambda_i_m + dlambda
    Ei = H_PLANCK * C_LIGHT / lambda_i_m
    Ef = H_PLANCK * C_LIGHT / lambda_f
    Ee = Ei - Ef
    # Recoil electron angle
    hf_i = Ei
    if np.tan(theta_rad / 2) != 0:
        cot_phi = (1 + hf_i / (M_ELECTRON * C_LIGHT**2)) * (1 / np.tan(theta_rad / 2))
        phi = np.arctan(1 / cot_phi) if cot_phi != 0 else np.pi / 2
    else:
        phi = np.pi / 2
    return {
        "dlambda": dlambda, "lambda_f": lambda_f,
        "Ei_eV": Ei / EV_TO_J, "Ef_eV": Ef / EV_TO_J, "Ee_eV": Ee / EV_TO_J,
        "phi_rad": phi
    }


# ======================================================================
# HTML ANIMATION: PHOTOELECTRIC EFFECT
# ======================================================================

def photoelectric_animation_html(emitting, photon_color, intensity_level, photon_wavelength_nm):
    """Generate Canvas animation for photoelectric effect"""
    intensity_val = intensity_level / 100.0
    html_code = f"""
    <div style="direction:ltr; text-align:center;">
    <canvas id="peCanvas" width="700" height="340" style="
        background: linear-gradient(180deg, #0a0a1a 0%, #111133 100%);
        border-radius: 12px; border: 1px solid #333; max-width:100%;
    "></canvas>
    </div>
    <script>
    (function() {{
        const canvas = document.getElementById('peCanvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const emitting = {str(emitting).lower()};
        const pColor = '{photon_color}';
        const intensity = {intensity_val};
        const wl = {photon_wavelength_nm};
        
        const metalX = 430, metalW = 35, metalTop = 30, metalBot = 310;
        let photons = [], electrons = [], frame = 0;
        const maxPhotons = Math.floor(intensity * 15) + 2;
        
        function spawnPhoton() {{
            if (photons.filter(p => !p.hit).length < maxPhotons) {{
                photons.push({{
                    x: -5 - Math.random() * 30,
                    y: metalTop + 20 + Math.random() * (metalBot - metalTop - 40),
                    vx: 2.5 + Math.random() * 1.5,
                    waveOff: Math.random() * Math.PI * 2,
                    hit: false, alpha: 1
                }});
            }}
        }}
        
        function drawWavyLine(x, y, len, color, alpha, waveOff) {{
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.shadowColor = color;
            ctx.shadowBlur = 8;
            ctx.beginPath();
            const amp = Math.max(3, 8 - (wl - 200) / 100);
            const freq = Math.max(0.05, 0.4 - (wl - 200) / 2000);
            for (let i = 0; i <= len; i += 2) {{
                ctx.lineTo(x - i, y + Math.sin((i * freq + waveOff + frame * 0.15)) * amp);
            }}
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }}
        
        function drawElectron(x, y, vx, vy, alpha) {{
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.fillStyle = '#00ccff';
            ctx.shadowColor = '#00ccff';
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#66ddff';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.shadowBlur = 0;
            // minus sign
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x - 2, y);
            ctx.lineTo(x + 2, y);
            ctx.stroke();
            ctx.restore();
        }}
        
        function drawMetal() {{
            const grad = ctx.createLinearGradient(metalX, 0, metalX + metalW, 0);
            grad.addColorStop(0, '#666');
            grad.addColorStop(0.5, '#999');
            grad.addColorStop(1, '#777');
            ctx.fillStyle = grad;
            ctx.fillRect(metalX, metalTop, metalW, metalBot - metalTop);
            ctx.strokeStyle = '#aaa';
            ctx.lineWidth = 1;
            ctx.strokeRect(metalX, metalTop, metalW, metalBot - metalTop);
            // label
            ctx.fillStyle = '#ccc';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Metal', metalX + metalW/2, metalBot + 18);
        }}
        
        function drawLabels() {{
            ctx.fillStyle = '#888';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';
            ctx.fillText('Light Source', 20, 25);
            // Arrow showing light direction
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.setLineDash([4,4]);
            ctx.beginPath();
            ctx.moveTo(80, 28);
            ctx.lineTo(metalX - 20, 28);
            ctx.stroke();
            ctx.setLineDash([]);
            
            if (emitting) {{
                ctx.fillStyle = '#00ccff';
                ctx.font = '12px Arial';
                ctx.textAlign = 'left';
                ctx.fillText('e\\u207B emitted', metalX + metalW + 20, 25);
            }} else {{
                ctx.fillStyle = '#ff4444';
                ctx.font = '13px Arial';
                ctx.textAlign = 'left';
                ctx.fillText('NO emission!', metalX + metalW + 15, 25);
                ctx.font = '11px Arial';
                ctx.fillText('f < f\\u2080', metalX + metalW + 15, 42);
            }}
        }}
        
        function animate() {{
            ctx.clearRect(0, 0, W, H);
            drawMetal();
            drawLabels();
            
            // Spawn photons
            const spawnRate = Math.max(2, Math.floor(20 / maxPhotons));
            if (frame % spawnRate === 0) spawnPhoton();
            
            // Update photons
            for (let i = photons.length - 1; i >= 0; i--) {{
                let p = photons[i];
                p.x += p.vx;
                if (p.x >= metalX - 5 && !p.hit) {{
                    p.hit = true;
                    if (emitting) {{
                        const speed = 1.5 + Math.random() * 2;
                        const angle = (Math.random() - 0.5) * 1.2;
                        electrons.push({{
                            x: metalX + metalW + 5,
                            y: p.y,
                            vx: Math.cos(angle) * speed,
                            vy: Math.sin(angle) * speed,
                            alpha: 1
                        }});
                    }}
                }}
                if (p.hit) {{
                    p.alpha -= 0.08;
                    if (p.alpha <= 0) {{ photons.splice(i, 1); continue; }}
                }}
                if (!p.hit) {{
                    drawWavyLine(p.x, p.y, 30, pColor, 1, p.waveOff);
                }}
            }}
            
            // Update electrons
            for (let i = electrons.length - 1; i >= 0; i--) {{
                let el = electrons[i];
                el.x += el.vx;
                el.y += el.vy;
                if (el.x > W + 10 || el.y < -10 || el.y > H + 10) {{
                    electrons.splice(i, 1); continue;
                }}
                drawElectron(el.x, el.y, el.vx, el.vy, el.alpha);
            }}
            
            frame++;
            requestAnimationFrame(animate);
        }}
        animate();
    }})();
    </script>
    """
    return html_code


# ======================================================================
# HTML ANIMATION: BLACKBODY CAVITY
# ======================================================================

def blackbody_cavity_html(T):
    """Generate Canvas animation for blackbody cavity"""
    r, g, b = blackbody_rgb(T)
    color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(min(255, r * 255)), int(min(255, g * 255)), int(min(255, b * 255))
    )
    num_particles = min(60, max(10, int(T / 100)))
    html_code = f"""
    <div style="direction:ltr; text-align:center;">
    <canvas id="bbCanvas" width="400" height="350" style="
        background: #0a0a1a; border-radius: 12px; border: 1px solid #333; max-width:100%;
    "></canvas>
    </div>
    <script>
    (function() {{
        const canvas = document.getElementById('bbCanvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const color = '{color_hex}';
        const nParts = {num_particles};
        const T = {T};
        
        const cx = 200, cy = 175, cw = 220, ch = 200;
        const holeX = cx + cw/2, holeY = cy - 15, holeW = 25, holeH = 30;
        
        let particles = [];
        for (let i = 0; i < nParts; i++) {{
            particles.push({{
                x: cx - cw/2 + 15 + Math.random() * (cw - 30),
                y: cy - ch/2 + 15 + Math.random() * (ch - 30),
                vx: (Math.random() - 0.5) * 3,
                vy: (Math.random() - 0.5) * 3,
                escaped: false, alpha: 1
            }});
        }}
        
        let escapedParticles = [];
        let frame = 0;
        
        function drawCavity() {{
            ctx.fillStyle = '#1a1a2e';
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.rect(cx - cw/2, cy - ch/2, cw, ch);
            ctx.stroke();
            
            // Hole opening
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(holeX - holeW/2, cy - ch/2 - 2, holeW, ch/2 - holeY + ch/2 + 4);
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(holeX - holeW/2, cy - ch/2);
            ctx.lineTo(holeX - holeW/2, holeY);
            ctx.lineTo(holeX + holeW/2, holeY);
            ctx.lineTo(holeX + holeW/2, cy - ch/2);
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#aaa';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Hole', holeX, cy - ch/2 - 8);
            ctx.fillText('Cavity', cx, cy + ch/2 + 20);
        }}
        
        function animate() {{
            ctx.clearRect(0, 0, W, H);
            drawCavity();
            
            for (let p of particles) {{
                if (!p.escaped) {{
                    p.x += p.vx;
                    p.y += p.vy;
                    const lx = cx - cw/2 + 5, rx = cx + cw/2 - 5;
                    const ty = cy - ch/2 + 5, by = cy + ch/2 - 5;
                    
                    if (p.x <= lx) {{ p.x = lx; p.vx *= -1; }}
                    if (p.x >= rx) {{
                        if (p.y < holeY && p.x >= holeX - holeW/2 && p.x <= holeX + holeW/2) {{
                            p.escaped = true;
                            escapedParticles.push({{
                                x: p.x, y: p.y,
                                vx: 1.5 + Math.random(), vy: -0.5 - Math.random() * 0.5,
                                alpha: 1
                            }});
                            p.x = cx - cw/2 + 15 + Math.random() * (cw - 30);
                            p.y = cy - ch/2 + 15 + Math.random() * (ch - 30);
                        }} else {{
                            p.x = rx; p.vx *= -1;
                        }}
                    }}
                    if (p.y <= ty) {{ p.y = ty; p.vy *= -1; }}
                    if (p.y >= by) {{ p.y = by; p.vy *= -1; }}
                    
                    ctx.save();
                    ctx.fillStyle = color;
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 10;
                    ctx.globalAlpha = 0.85;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.restore();
                }}
            }}
            
            for (let i = escapedParticles.length - 1; i >= 0; i--) {{
                let ep = escapedParticles[i];
                ep.x += ep.vx;
                ep.y += ep.vy;
                ep.alpha -= 0.008;
                if (ep.alpha <= 0 || ep.x > W + 10 || ep.y < -10) {{
                    escapedParticles.splice(i, 1); continue;
                }}
                ctx.save();
                ctx.fillStyle = color;
                ctx.shadowColor = color;
                ctx.shadowBlur = 15;
                ctx.globalAlpha = ep.alpha;
                ctx.beginPath();
                ctx.arc(ep.x, ep.y, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }}
            
            // Temperature label
            ctx.fillStyle = '#ddd';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('T = ' + T + ' K', cx, H - 15);
            
            frame++;
            requestAnimationFrame(animate);
        }}
        animate();
    }})();
    </script>
    """
    return html_code


# ======================================================================
# HTML ANIMATION: COMPTON SCATTERING
# ======================================================================

def compton_animation_html(theta_deg, lambda_i_nm):
    """Generate Canvas animation for Compton scattering"""
    theta_rad = np.radians(theta_deg)
    res = compute_compton(lambda_i_nm * 1e-9, theta_rad)
    phi_deg = np.degrees(res["phi_rad"])
    html_code = f"""
    <div style="direction:ltr; text-align:center;">
    <canvas id="compCanvas" width="600" height="400" style="
        background: #0a0a1a; border-radius: 12px; border: 1px solid #333; max-width:100%;
    "></canvas>
    </div>
    <script>
    (function() {{
        const canvas = document.getElementById('compCanvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const theta = {theta_rad};
        const thetaDeg = {theta_deg};
        const phiDeg = {phi_deg:.1f};
        const phi = {res["phi_rad"]};
        let frame = 0;
        let phase = 0; // 0=approach, 1=collision, 2=scatter
        
        const cx = 200, cy = 200;
        const arrowLen = 130;
        
        function drawArrow(x1, y1, x2, y2, color, label, lw) {{
            const angle = Math.atan2(y2 - y1, x2 - x1);
            const headLen = 12;
            ctx.save();
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = lw || 3;
            ctx.shadowColor = color;
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
            ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fill();
            ctx.shadowBlur = 0;
            if (label) {{
                ctx.font = 'bold 13px Arial';
                ctx.textAlign = 'center';
                const lx = (x1 + x2) / 2 + 15 * Math.cos(angle + Math.PI/2);
                const ly = (y1 + y2) / 2 + 15 * Math.sin(angle + Math.PI/2);
                ctx.fillText(label, lx, ly);
            }}
            ctx.restore();
        }}
        
        function drawElectron(x, y, label) {{
            ctx.save();
            ctx.fillStyle = '#00ccff';
            ctx.shadowColor = '#00ccff';
            ctx.shadowBlur = 15;
            ctx.beginPath();
            ctx.arc(x, y, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x - 5, y);
            ctx.lineTo(x + 5, y);
            ctx.stroke();
            if (label) {{
                ctx.fillStyle = '#00ccff';
                ctx.font = '12px Arial';
                ctx.fillText(label, x, y + 25);
            }}
            ctx.restore();
        }}
        
        function drawWavy(x1, y1, x2, y2, color) {{
            const dx = x2 - x1, dy = y2 - y1;
            const len = Math.sqrt(dx*dx + dy*dy);
            const steps = Math.floor(len / 3);
            const angle = Math.atan2(dy, dx);
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.shadowColor = color;
            ctx.shadowBlur = 5;
            ctx.beginPath();
            for (let i = 0; i <= steps; i++) {{
                const t = i / steps;
                const bx = x1 + dx * t;
                const by = y1 + dy * t;
                const perp = Math.sin(t * 15 + frame * 0.2) * 6;
                const px = bx + perp * Math.cos(angle + Math.PI/2);
                const py = by + perp * Math.sin(angle + Math.PI/2);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }}
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.restore();
        }}
        
        function animate() {{
            ctx.clearRect(0, 0, W, H);
            
            // Target label
            ctx.fillStyle = '#888';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Graphite Target', cx, cy + 50);
            
            // Incident photon
            const incX = cx - arrowLen;
            drawWavy(incX - 40, cy, cx - 15, cy, '#ffaa00');
            ctx.fillStyle = '#ffaa00';
            ctx.font = 'bold 13px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('\\u03BB\\u1D62 (incident)', incX - 10, cy - 15);
            
            // Electron at center
            drawElectron(cx, cy, 'e\\u207B (at rest)');
            
            // Scattered photon
            const scX = cx + arrowLen * Math.cos(-theta);
            const scY = cy + arrowLen * Math.sin(-theta);
            drawWavy(cx + 15, cy, scX, scY, '#ff6644');
            ctx.fillStyle = '#ff6644';
            ctx.font = 'bold 13px Arial';
            const scLabelX = scX + 15 * Math.cos(-theta + Math.PI/4);
            const scLabelY = scY + 15 * Math.sin(-theta + Math.PI/4);
            ctx.fillText('\\u03BBf (scattered)', scLabelX, scLabelY);
            
            // Recoil electron
            const elLen = arrowLen * 0.7;
            const elX = cx + elLen * Math.cos(phi);
            const elY = cy + elLen * Math.sin(phi);
            drawArrow(cx + 15, cy, elX, elY, '#00ccff', 'recoil e\\u207B', 2);
            drawElectron(elX, elY, '');
            
            // Angle arcs
            ctx.save();
            ctx.strokeStyle = '#ffaa00';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3,3]);
            ctx.beginPath();
            ctx.arc(cx, cy, 50, 0, -theta, theta > 0);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#ffaa00';
            ctx.font = '12px Arial';
            ctx.fillText('\\u03B8 = ' + thetaDeg + '\\u00B0', cx + 55, cy - 10);
            
            ctx.strokeStyle = '#00ccff';
            ctx.setLineDash([3,3]);
            ctx.beginPath();
            ctx.arc(cx, cy, 35, 0, phi, false);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#00ccff';
            ctx.fillText('\\u03C6 = ' + phiDeg + '\\u00B0', cx + 40, cy + 40);
            ctx.restore();
            
            // Incident line (dashed extension)
            ctx.save();
            ctx.strokeStyle = '#555';
            ctx.setLineDash([4,4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + arrowLen + 20, cy);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
            
            frame++;
            requestAnimationFrame(animate);
        }}
        animate();
    }})();
    </script>
    """
    return html_code


# ======================================================================
# MAIN APPLICATION
# ======================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>⚛️ الفيزياء الحديثة - الطبيعة الجسيمية للضوء</h1>
    <h2>Modern Physics - Particle Nature of Light</h2>
    <div class="author-badge">إعداد Israa Youssuf Samara</div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🟢 الجسم الأسود | Blackbody",
    "💛 الكهروضوئية | Photoelectric",
    "🔵 تفسير أينشتين | Einstein",
    "🟣 جهد الإيقاف | Stopping Potential",
    "🔴 كومبتون | Compton"
])

# ======================================================================
# TAB 1: BLACKBODY RADIATION
# ======================================================================
with tab1:
    st.markdown("""
    <div class="concept-card">
        <h3>🔥 ما هو الجسم الأسود؟ | What is a Blackbody?</h3>
        <p>الجسم الأسود هو <b>جسم مثالي</b> يمتصّ <b>جميع</b> الأشعة الكهرمغناطيسية الساقطة عليه بغضّ النظر عن تردّدها، 
        ويُشعّها كلها بالكفاءة نفسها. يعتمد الإشعاع المنبعث <b>فقط</b> على درجة حرارته.</p>
        <p>يمثّل الثقب الصغير في جسم أجوف مثالاً تقريبياً للجسم الأسود: الأشعة الداخلة عبر الثقب تتشتّت داخل التجويف 
        وتمتصّ بالكامل.</p>
    </div>
    """, unsafe_allow_html=True)

    col_anim, col_curve = st.columns([1, 1.3])

    with col_anim:
        st.markdown("#### 🎬 محاكاة تجويف الجسم الأسود")
        temperature = st.slider("درجة الحرارة T (K)", 1000, 8000, 3000, step=100,
                                key="bb_temp")
        components.html(blackbody_cavity_html(temperature), height=370)

        r, g, b = blackbody_rgb(temperature)
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(min(255, r * 255)), int(min(255, g * 255)), int(min(255, b * 255))
        )
        peak_nm = WIEN_B / temperature * 1e9

        st.markdown(f"""
        <div class="explain-box">
            <b>🔍 ما الذي يحدث عند T = {temperature} K؟</b><br>
            • لون التوهّج المقترب: <span style="color:{color_hex}; font-weight:bold;">■</span><br>
            • أقصى طول موجي للإشعاع (قانون فين): λ<sub>max</sub> = <b>{peak_nm:.0f} nm</b><br>
            {"• معظم الإشعاع في منطقة <b>الأشعة تحت الحمراء</b> (غير مرئي)" if peak_nm > 700 else
             "• الإشعاع يدخل منطقة <b>الضوء المرئي</b>" if peak_nm > 380 else
             "• الإشعاع في منطقة <b>الأشعة فوق البنفسجية</b>"}
        </div>
        """, unsafe_allow_html=True)

    with col_curve:
        st.markdown("#### 📈 منحنى إشعاع الجسم الأسود - خطوة بخطوة")
        step = st.slider("اختَر الخطوة لترى التوضيح تدريجياً", 1, 6, 1, key="bb_step")

        wavelengths = np.linspace(1e-7, 3e-6, 500)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#aaa')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Wavelength (nm)', color='#ccc', fontsize=11)
        ax.set_ylabel('Spectral Radiance', color='#ccc', fontsize=11)

        temps_show = [3000, 4000, 5000, 6000]
        colors_t = ['#ff4444', '#ff8844', '#ffcc44', '#ffffff']
        labels_t = ['3000 K', '4000 K', '5000 K', '6000 K']

        if step >= 1:
            rad1 = planck_radiance(wavelengths, temps_show[0])
            ax.plot(wavelengths * 1e9, rad1 / 1e13, color=colors_t[0], linewidth=2, label=labels_t[0])
            ax.fill_between(wavelengths * 1e9, rad1 / 1e13, alpha=0.15, color=colors_t[0])

        if step >= 2:
            rad2 = planck_radiance(wavelengths, temps_show[1])
            ax.plot(wavelengths * 1e9, rad2 / 1e13, color=colors_t[1], linewidth=2, label=labels_t[1])

        if step >= 3:
            rad3 = planck_radiance(wavelengths, temps_show[2])
            ax.plot(wavelengths * 1e9, rad3 / 1e13, color=colors_t[2], linewidth=2, label=labels_t[2])

        if step >= 4:
            rad4 = planck_radiance(wavelengths, temps_show[3])
            ax.plot(wavelengths * 1e9, rad4 / 1e13, color=colors_t[3], linewidth=2, label=labels_t[3])

        if step >= 5:
            for i, T in enumerate(temps_show):
                pk = WIEN_B / T * 1e9
                ax.axvline(pk, color=colors_t[i], linestyle='--', alpha=0.5, linewidth=1)
                ax.annotate(f'λ_max={pk:.0f}', xy=(pk, ax.get_ylim()[1] * 0.9),
                           fontsize=8, color=colors_t[i], rotation=90, ha='right')

        if step >= 6:
            rad_total = planck_radiance(wavelengths, 6000)
            ax.fill_between(wavelengths * 1e9, rad_total / 1e13, alpha=0.2, color='#00aaff')
            ax.annotate('Total Energy\n(area under curve)', xy=(1500, ax.get_ylim()[1] * 0.7),
                       fontsize=10, color='#00aaff', ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#00aaff', alpha=0.8))

        # Visible spectrum band
        ax.axvspan(380, 780, alpha=0.08, color='white')
        ax.text(580, ax.get_ylim()[1] * 0.05, 'Visible', fontsize=9, color='#666', ha='center')
        ax.text(200, ax.get_ylim()[1] * 0.05, 'UV', fontsize=9, color='#666', ha='center')
        ax.text(2000, ax.get_ylim()[1] * 0.05, 'IR', fontsize=9, color='#666', ha='center')

        ax.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', edgecolor='#444',
                  labelcolor='#ccc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        explanations = {
            1: "📌 <b>الخطوة 1:</b> عند T = 3000 K، معظم الإشعاع في منطقة الأشعة تحت الحمراء (λ > 700 nm). الجسم يبدو أحمر خافت لأنه يشعّ القليل من الضوء المرئي.",
            2: "📌 <b>الخطوة 2:</b> عند T = 4000 K، يزداد الإشعاع الكلي وتبدأ القمة بالانزاح نحو الأطوال الموجية الأقصر. يظهر اللون البرتقالي.",
            3: "📌 <b>الخطوة 3:</b> عند T = 5000 K، القمة تقترب من منطقة الضوء المرئي. يظهر اللون الأصفر.",
            4: "📌 <b>الخطوة 4:</b> عند T = 6000 K (≈ درجة حرارة سطح الشمس)، القمة في منطقة الضوء المرئي. اللون أبيض.",
            5: "📌 <b>الخطوة 5:</b> الخطوط المتقطعة تُظهر قمة كل منحنى. مع زيادة الحرارة، تنزاح القمة نحو الأطوال الموجية الأقصر (قانون فين: λ_max = b/T).",
            6: "📌 <b>الخطوة 6:</b> المساحة تحت المنحنى تمثّل الطاقة الكلية المشعّة. المساحة عند 6000 K أكبر بكثير من 3000 K، مما يعني أن الجسم الأ hotter يشعّ طاقة أكبر."
        }
        st.markdown(f'<div class="explain-box">{explanations[step]}</div>', unsafe_allow_html=True)

    # Rayleigh-Jeans vs Planck comparison
    st.markdown("---")
    st.markdown("#### ⚠️ كارثة الأشعة فوق البنفسجية ومبدأ بلانك")
    col_rj, col_explain = st.columns([1, 1])

    with col_rj:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        ax2.tick_params(colors='#aaa')
        ax2.spines['bottom'].set_color('#444')
        ax2.spines['left'].set_color('#444')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        wl = np.linspace(1e-7, 5e-6, 600)
        planck_val = planck_radiance(wl, 5000) / 1e13
        rj_val = 2 * K_BOLTZMANN * 5000 / (wl**4) / 1e13
        rj_val = np.clip(rj_val, 0, planck_val.max() * 3)

        ax2.plot(wl * 1e9, planck_val, color='#00ff88', linewidth=2.5, label="Planck (experimental)")
        ax2.plot(wl * 1e9, rj_val, color='#ff4444', linewidth=2, linestyle='--',
                 label="Rayleigh-Jeans")
        ax2.set_xlabel('Wavelength (nm)', color='#ccc', fontsize=11)
        ax2.set_ylabel('Spectral Radiance', color='#ccc', fontsize=11)
        ax2.set_ylim(0, planck_val.max() * 1.5)
        ax2.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc')
        ax2.annotate('UV Catastrophe\n→ ∞ !!!', xy=(200, planck_val.max() * 1.3),
                     fontsize=11, color='#ff4444', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='#2a0a0a', edgecolor='#ff4444'))
        ax2.axvspan(100, 380, alpha=0.1, color='#ff00ff')
        ax2.text(240, planck_val.max() * 0.1, 'UV', fontsize=10, color='#ff66ff')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_explain:
        st.error("❌ لماذا فشلت الفيزياء الكلاسيكية؟")
        st.markdown("""
نموذج **رايلي-جينز** (الخط الأحمر المتقطع) يتفق مع التجربة في منطقة الأشعة تحت الحمراء فقط، لكنه يتوقع أن شدة الإشعاع **تؤول إلى اللانهاية** عند الأطوال الموجية القصيرة (الأشعة فوق البنفسجية)!

هذا مستحيل فيزيائياً وعُرف بـ **"كارثة الأشعة فوق البنفسجية"**.
""")

        st.success("✅ حلّ بلانك: تكمية الطاقة")
        st.markdown("""
افترض بلانك أن الطاقة لا تُشعّ أو تُمتصّ بشكل متّصل، بل على شكل **حزم منفصلة** تُسمّى **كمّات (Quanta)**:
""")

        st.code("E = hf     و     Eₙ = nhf", language="text")

        st.markdown("""
حيث:
- **h = 6.63 × 10⁻³⁴ J·s** (ثابت بلانك)
- **f** : التردد
- **n** : عدد صحيح موجب (1, 2, 3, ...)

كل كمة طاقة تساوي **E = hf**، والطاقة الكلية عند تردد معيّن تكون **nhf** (أي hf, 2hf, 3hf, ...) وليست أي قيمة بينها.
""")


# ======================================================================
# TAB 2: PHOTOELECTRIC EFFECT
# ======================================================================
with tab2:
    st.markdown("""
    <div class="concept-card">
        <h3>⚡ ما هي الظاهرة الكهروضوئية؟ | What is the Photoelectric Effect?</h3>
        <p>انبعاث <b>إلكترونات</b> من سطح فلزّ عند سقوط إشعاع كهرمغناطيسي بتردّد <b>مناسب</b> عليه. 
        الإلكترونات المنبعثة تُسمّى <b>الإلكترونات الضوئية</b>.</p>
        <p>اكتشفها <b>هيرتز</b> عام 1887، ودرسها <b>لينارد</b> تجريبياً.</p>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_anim_pe = st.columns([0.8, 1.2])

    with col_ctrl:
        st.markdown("#### 🎛️ التحكم بالتجربة")
        selected_material = st.selectbox("اختر المادة (الفلز)", list(MATERIALS.keys()),
                                          key="pe_material")
        mat = MATERIALS[selected_material]
        phi_eV = mat["phi_eV"]
        f0, lam0 = compute_threshold(phi_eV)

        wavelength_nm = st.slider("الطول الموجي للضوء الساقط λ (nm)", 100, 800, 400, step=5,
                                   key="pe_wavelength")
        freq_Hz = C_LIGHT / (wavelength_nm * 1e-9)
        intensity = st.slider("شدة الضوء (نسبة)", 10, 100, 50, key="pe_intensity")

        res_pe = compute_photoelectric(phi_eV, freq_Hz)

        # Material info
        st.markdown(f"""
        <div class="explain-box">
            <b>📋 خصائص المادة المختارة ({mat['symbol']}):</b><br>
            • اقتران الشغل: Φ = <b>{phi_eV} eV</b><br>
            • تردّد العتبة: f<sub>0</sub> = <b>{f0:.2e} Hz</b><br>
            • الطول الموجي للعتبة: λ<sub>0</sub> = <b>{lam0*1e9:.1f} nm</b><br>
            • منطقة الطول الموجي للعتبة: <b>{"أشعة فوق بنفسجية" if lam0*1e9 < 380 else "ضوء مرئي" if lam0*1e9 < 780 else "أشعة تحت حمراء"}</b>
        </div>
        """, unsafe_allow_html=True)

        # Photon info
        photon_E = res_pe["photon_E_eV"]
        st.markdown(f"""
        <div class="explain-box">
            <b>💡 خصائص الفوتون الساقط:</b><br>
            • الطول الموجي: λ = <b>{wavelength_nm} nm</b><br>
            • التردد: f = <b>{freq_Hz:.2e} Hz</b><br>
            • طاقة الفوتون: E = <b>{photon_E:.2f} eV</b><br>
            • مقارنة: {"E > Φ ✅" if res_pe["emitting"] else "E < Φ ❌"} 
            ({"f > f₀" if res_pe["emitting"] else "f < f₀"})
        </div>
        """, unsafe_allow_html=True)

    with col_anim_pe:
        st.markdown("#### 🎬 محاكاة الظاهرة الكهروضوئية")
        # Photon color
        if 380 <= wavelength_nm <= 780:
            pr, pg, pb = wavelength_to_rgb(wavelength_nm)
            p_color = '#{:02x}{:02x}{:02x}'.format(int(pr*255), int(pg*255), int(pb*255))
        elif wavelength_nm < 380:
            p_color = '#aa44ff'
        else:
            p_color = '#ff2222'

        components.html(
            photoelectric_animation_html(res_pe["emitting"], p_color, intensity, wavelength_nm),
            height=370
        )

    # Results
    if res_pe["emitting"]:
        st.markdown(f"""
        <div class="result-box">
            <h4>✅ يحدث انبعاث إلكترونات!</h4>
            <p>• لأن طاقة الفوتون ({photon_E:.2f} eV) > اقتران الشغل ({phi_eV} eV)</p>
            <p>• الطاقة الحركية العظمى: KE<sub>max</sub> = hf - Φ = <b>{res_pe["KE_max_eV"]:.2f} eV</b></p>
            <p>• عدد الإلكترونات المتحررة يعتمد على <b>شدة</b> الضوء ({intensity}%)</p>
            <p>• الطاقة الحركية العظمى تعتمد على <b>تردد</b> الضوء ({freq_Hz:.2e} Hz) وليس شدته</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            <h4>❌ لا يحدث انبعاث إلكترونات!</h4>
            <p>• لأن طاقة الفوتون ({photon_E:.2f} eV) < اقتران الشغل ({phi_eV} eV)</p>
            <p>• التردد ({freq_Hz:.2e} Hz) < تردد العتبة ({f0:.2e} Hz)</p>
            <p>• <b>حتى لو زادت الشدة إلى أقصى حد!</b> لأن زيادة الشدة تزيد عدد الفوتونات فقط، 
            وليس طاقة كل فوتون.</p>
        </div>
        """, unsafe_allow_html=True)

    # Classical vs Quantum comparison
    st.markdown("---")
    st.markdown("#### 🆚 الفيزياء الكلاسيكية مقابل النتائج التجريبية")
    col_cl, col_qm = st.columns(2)
    with col_cl:
        st.markdown("""
        <div class="warning-box">
            <h4>❌ تنبّؤات النموذج الموجي (كلاسيكي)</h4>
            <p>1. الإلكترونات تنبعث عند <b>أي تردد</b> بشرط شدة كافية</p>
            <p>2. الانبعاث <b>ليس فورياً</b> - يحتاج وقتاً لامتصاص الطاقة</p>
            <p>3. زيادة الشدة → زيادة <b>الطاقة الحركية</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col_qm:
        st.markdown("""
        <div class="result-box">
            <h4>✅ النتائج التجريبية الفعلية</h4>
            <p>1. الإلكترونات تنبعث فقط عند f ≥ f<sub>0</sub> <b>مهما كانت الشدة</b></p>
            <p>2. الانبعاث <b>فوري</b> عند f ≥ f<sub>0</sub></p>
            <p>3. زيادة الشدة → زيادة <b>عدد</b> الإلكترونات فقط، لا طاقتها</p>
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# TAB 3: EINSTEIN'S EXPLANATION
# ======================================================================
with tab3:
    st.markdown("""
    <div class="concept-card">
        <h3>🧠 تفسير أينشتين للظاهرة الكهروضوئية (1905)</h3>
        <p>اعتمد أينشتين على فرضية بلانك وافترض أن الأشعة الكهرمغناطيسية تتكوّن من <b>جسيمات</b> تُسمّى 
        <b>فوتونات</b>، طاقة كل منها:</p>
    </div>
    <div class="formula-box">E = hf</div>
    <div class="explain-box">
        <p>• كل فوتون يعطي <b>طاقته كاملة</b> لإلكترون <b>واحد فقط</b></p>
        <p>• إذا كانت طاقة الفوتون <b>أكبر</b> من اقتران الشغل Φ، يتحرر الإلكترون واكتسب فائض الطاقة كطاقة حركية</p>
        <p>• إذا كانت طاقة الفوتون <b>أصغر</b> من Φ، لا يتحرر أي إلكترون <b>مهما كانت الشدة</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">KE<sub>max</sub> = hf - Φ &nbsp;&nbsp;⟹&nbsp;&nbsp; hf = Φ + ½mv<sub>max</sub>²</div>
    """, unsafe_allow_html=True)

    col_sel3, col_graph3 = st.columns([0.7, 1.3])

    with col_sel3:
        st.markdown("#### 🎛️ اختر المادة")
        mat3_name = st.selectbox("اختر الفلز", list(MATERIALS.keys()), key="ein_material")
        mat3 = MATERIALS[mat3_name]
        phi3 = mat3["phi_eV"]
        f0_3, lam0_3 = compute_threshold(phi3)

        # Also show comparison materials
        st.markdown("#### 🔍 خصائص المادة")
        st.markdown(f"""
        <div class="explain-box">
            <b>{mat3_name}:</b><br>
            • Φ = <b>{phi3} eV</b><br>
            • f<sub>0</sub> = <b>{f0_3:.2e} Hz</b><br>
            • λ<sub>0</sub> = <b>{lam0_3*1e9:.1f} nm</b>
        </div>
        """, unsafe_allow_html=True)

        # Table of all materials
        st.markdown("#### 📊 اقتران الشغل لجميع الفلزات")
        table_data = []
        for name, data in MATERIALS.items():
            f0_t, _ = compute_threshold(data["phi_eV"])
            table_data.append({
                "الفلز": name,
                "Φ (eV)": data["phi_eV"],
                "f₀ (×10¹⁴ Hz)": f"{f0_t/1e14:.2f}"
            })
        st.table(table_data)

    with col_graph3:
        st.markdown("#### 📈 العلاقة بين KE_max والتردد - خطوة بخطوة")
        step3 = st.slider("الخطوة", 1, 6, 1, key="ein_step")

        freqs = np.linspace(0, 15e14, 300)
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        fig3.patch.set_facecolor('#0d1117')
        ax3.set_facecolor('#0d1117')
        ax3.tick_params(colors='#aaa')
        ax3.spines['bottom'].set_color('#444')
        ax3.spines['left'].set_color('#444')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_xlabel('Frequency f (×10¹⁴ Hz)', color='#ccc', fontsize=11)
        ax3.set_ylabel('KE_max (eV)', color='#ccc', fontsize=11)
        ax3.axhline(y=0, color='#444', linewidth=0.8)
        ax3.axvline(x=0, color='#444', linewidth=0.8)

        f0_plot = f0_3 / 1e14  # in units of 10^14 Hz

        if step3 >= 1:
            ax3.axvline(x=f0_plot, color='#ff4444', linestyle='--', linewidth=1.5, alpha=0.8)
            ax3.annotate(f'f₀ = {f0_plot:.2f}×10¹⁴ Hz',
                        xy=(f0_plot, -0.3), fontsize=9, color='#ff4444', ha='center')

        if step3 >= 2:
            # Draw the forbidden region
            ax3.axvspan(0, f0_plot, alpha=0.08, color='#ff0000')
            ax3.text(f0_plot / 2, ax3.get_ylim()[1] * 0.5 if ax3.get_ylim()[1] > 0 else 2,
                    'No emission\n(f < f₀)', fontsize=10, color='#ff6666',
                    ha='center', va='center', style='italic')

        if step3 >= 3:
            # Draw the line: KE = hf - Phi (only for f >= f0)
            f_line = np.linspace(f0_plot, 15, 200)
            KE_line = H_PLANCK * f_line * 1e14 / EV_TO_J - phi3
            ax3.plot(f_line, KE_line, color='#00ff88', linewidth=2.5,
                    label=f'{mat3["symbol"]}: KE = hf - Φ')

        if step3 >= 4:
            # Show slope = h
            ax3.annotate('slope = h\n(Planck constant)',
                        xy=(10, H_PLANCK * 10e14 / EV_TO_J - phi3),
                        xytext=(11, H_PLANCK * 10e14 / EV_TO_J - phi3 - 1),
                        fontsize=10, color='#ffaa00',
                        arrowprops=dict(arrowstyle='->', color='#ffaa00', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ffaa00'))

        if step3 >= 5:
            # Show intercept = -Phi
            y_lim = ax3.get_ylim()
            ax3.plot([0, f0_plot], [-phi3, 0], color='#00ff88', linewidth=2, linestyle=':')
            ax3.annotate(f'intercept = -Φ = -{phi3} eV',
                        xy=(0, -phi3), xytext=(1.5, -phi3 - 0.5),
                        fontsize=10, color='#ff88ff',
                        arrowprops=dict(arrowstyle='->', color='#ff88ff', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ff88ff'))

        if step3 >= 6:
            # Add multiple materials for comparison
            for name, data in list(MATERIALS.items())[:6]:
                if name == mat3_name:
                    continue
                f0_t = data["phi_eV"] * EV_TO_J / H_PLANCK / 1e14
                f_l = np.linspace(f0_t, 15, 200)
                KE_l = H_PLANCK * f_l * 1e14 / EV_TO_J - data["phi_eV"]
                KE_l = np.clip(KE_l, 0, None)
                ax3.plot(f_l, KE_l, color=data["color"], linewidth=1.2, alpha=0.6,
                        linestyle='--', label=f'{data["symbol"]}: Φ={data["phi_eV"]} eV')

        ax3.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc',
                  loc='upper left')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        exp3 = {
            1: f"📌 <b>الخطوة 1:</b> نحدّد أولاً تردد العتبة f₀ = {f0_plot:.2f}×10¹⁴ Hz على المحور الأفقي. هذا أدنى تردد يلزم لتحرير إلكترون من {mat3['symbol']}.",
            2: f"📌 <b>الخطوة 2:</b> المنطقة المظللة بالأحمر (f < f₀) هي <b>منطقة ممنوعة</b>: لا يحدث أي انبعاث مهما زادت الشدة. هذا ما عجزت الفيزياء الكلاسيكية عن تفسيره.",
            3: f"📌 <b>الخطوة 3:</b> عند f ≥ f₀، العلاقة خطّية: KE_max = hf - Φ. كل نقطة على الخط تمثّل قياساً تجريبياً عند تردد معيّن.",
            4: "📌 <b>الخطوة 4:</b> <b>ميل الخط = h</b> (ثابت بلانك)! هذا يعني أن الميل هو نفسه لجميع الفلزات. أثبت ميليكان ذلك تجريبياً عام 1916.",
            5: f"📌 <b>الخطوة 5:</b> امتداد الخط ليقطع محور الطاقة عند <b>-Φ = -{phi3} eV</b>. هذا التقاطع يحدد اقتران الشغل للفلز.",
            6: "📌 <b>الخطوة 6:</b> عند مقارنة عدة فلزات: جميع الخطوط <b>متوازية</b> (نفس الميل h) لكنها تبدأ من نقاط مختلفة (اقترانات شغل مختلفة). الفلز ذو Φ الأصغر له f₀ الأصغر."
        }
        st.markdown(f'<div class="explain-box">{exp3[step3]}</div>', unsafe_allow_html=True)


# ======================================================================
# TAB 4: STOPPING POTENTIAL & THRESHOLD
# ======================================================================
with tab4:
    st.markdown("""
    <div class="concept-card">
        <h3>🔋 جهد الإيقاف وتردد العتبة | Stopping Potential & Threshold Frequency</h3>
        <p><b>جهد الإيقاف V<sub>s</sub>:</b> فرق الجهد الذي يصبح عنده التيار الكهرضوئي <b>صفراً</b>. 
        يوقف الإلكترونات ذات الطاقة الحركية العظمى فقط.</p>
        <p><b>تردد العتبة f<sub>0</sub>:</b> أدنى تردد يلزم لتحرير إلكترون من سطح الفلز دون إكسابه طاقة حركية.</p>
    </div>
    <div class="formula-box">
        KE<sub>max</sub> = eV<sub>s</sub> &nbsp;&nbsp;|&nbsp;&nbsp; f<sub>0</sub> = Φ / h
    </div>
    """, unsafe_allow_html=True)

    col_ctrl4, col_graph4 = st.columns([0.7, 1.3])

    with col_ctrl4:
        st.markdown("#### 🎛️ التحكم")
        mat4_name = st.selectbox("اختر الفلز", list(MATERIALS.keys()), key="sp_material")
        mat4 = MATERIALS[mat4_name]
        phi4 = mat4["phi_eV"]
        f0_4, lam0_4 = compute_threshold(phi4)

        wl4_nm = st.slider("الطول الموجي λ (nm)", 100, 800, 300, step=5, key="sp_wl")
        freq4 = C_LIGHT / (wl4_nm * 1e-9)
        res4 = compute_photoelectric(phi4, freq4)

        st.markdown(f"""
        <div class="explain-box">
            <b>📋 النتائج لـ {mat4_name} عند λ = {wl4_nm} nm:</b><br><br>
            • تردد الضوء: f = <b>{freq4:.2e} Hz</b><br>
            • تردد العتبة: f<sub>0</sub> = <b>{f0_4:.2e} Hz</b><br>
            • طاقة الفوتون: E = <b>{res4['photon_E_eV']:.2f} eV</b><br>
            • اقتران الشغل: Φ = <b>{phi4} eV</b><br><br>
        """, unsafe_allow_html=True)

        if res4["emitting"]:
            ke4 = res4["KE_max_eV"]
            vs4 = res4["Vs"]
            st.markdown(f"""
            <div class="result-box">
                <b>✅ يحدث انبعاث:</b><br>
                • KE<sub>max</sub> = hf - Φ = <b>{ke4:.3f} eV</b><br>
                • جهد الإيقاف: V<sub>s</sub> = KE<sub>max</sub>/e = <b>{vs4:.3f} V</b><br><br>
                <i>معنى V<sub>s</sub>: نحتاج فرق جهد سالب بقيمة {vs4:.3f} V بين الجامع والباعث 
                لإيقاف الإلكترونات الأسرع عن الوصول إلى الجامع.</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <b>❌ لا يحدث انبعاث</b><br>
                • f &lt; f<sub>0</sub> → V<sub>s</sub> = <b>0 V</b><br>
                • لا حاجة لجهد إيقاف لأنه لا توجد إلكترونات متحررة
            </div>
            """, unsafe_allow_html=True)

    with col_graph4:
        st.markdown("#### 📈 العلاقة بين جهد الإيقاف والتردد - خطوة بخطوة")
        step4 = st.slider("الخطوة", 1, 5, 1, key="sp_step")

        freqs4 = np.linspace(0, 15e14, 300)
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        fig4.patch.set_facecolor('#0d1117')
        ax4.set_facecolor('#0d1117')
        ax4.tick_params(colors='#aaa')
        ax4.spines['bottom'].set_color('#444')
        ax4.spines['left'].set_color('#444')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.set_xlabel('Frequency f (×10¹⁴ Hz)', color='#ccc', fontsize=11)
        ax4.set_ylabel('Stopping Potential Vs (V)', color='#ccc', fontsize=11)
        ax4.axhline(y=0, color='#444', linewidth=0.8)
        ax4.axvline(x=0, color='#444', linewidth=0.8)

        f0_p4 = f0_4 / 1e14

        if step4 >= 1:
            ax4.axvline(x=f0_p4, color='#ff4444', linestyle='--', linewidth=1.5)
            ax4.annotate(f'f₀ = {f0_p4:.2f}', xy=(f0_p4, -0.15), fontsize=9,
                        color='#ff4444', ha='center')

        if step4 >= 2:
            f_l4 = np.linspace(f0_p4, 15, 200)
            Vs_line = (H_PLANCK * f_l4 * 1e14 / EV_TO_J - phi4)
            ax4.plot(f_l4, Vs_line, color='#00ff88', linewidth=2.5,
                    label=f'{mat4["symbol"]}: Vs = (hf - Φ)/e')

        if step4 >= 3:
            ax4.annotate(f'slope = h/e\n= {H_PLANCK/E_CHARGE:.2e} V·s',
                        xy=(10, H_PLANCK * 10e14 / EV_TO_J - phi4),
                        xytext=(11.5, H_PLANCK * 10e14 / EV_TO_J - phi4 - 1.5),
                        fontsize=9, color='#ffaa00',
                        arrowprops=dict(arrowstyle='->', color='#ffaa00', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ffaa00'))

        if step4 >= 4:
            if res4["emitting"]:
                freq_plot = freq4 / 1e14
                ax4.plot(freq_plot, vs4, 'o', color='#ff4444', markersize=10, zorder=5)
                ax4.annotate(f'({freq_plot:.1f}, {vs4:.2f} V)',
                            xy=(freq_plot, vs4),
                            xytext=(freq_plot + 1, vs4 + 0.5),
                            fontsize=9, color='#ff6666',
                            arrowprops=dict(arrowstyle='->', color='#ff6666'),
                            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#ff6666'))

        if step4 >= 5:
            for name, data in list(MATERIALS.items())[:6]:
                if name == mat4_name:
                    continue
                f0_t = data["phi_eV"] * EV_TO_J / H_PLANCK / 1e14
                f_l = np.linspace(f0_t, 15, 200)
                Vs_l = H_PLANCK * f_l * 1e14 / EV_TO_J - data["phi_eV"]
                ax4.plot(f_l, Vs_l, color=data["color"], linewidth=1.2, alpha=0.6,
                        linestyle='--', label=f'{data["symbol"]}')

        ax4.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        vs_display = "V_s = " + str(round(vs4, 2)) + " V" if res4["emitting"] else "لا يوجد انبعاث"
        exp4 = {
            1: f"📌 <b>الخطوة 1:</b> نحدد تردد العتبة f₀ = {f0_p4:.2f}×10¹⁴ Hz. عند هذا التردد بالضبط، V<sub>s</sub> = 0 (الإلكترونات تتحرر لكن بدون طاقة حركية).",
            2: f"📌 <b>الخطوة 2:</b> العلاقة بين V<sub>s</sub> و f خطّية: V<sub>s</sub> = (hf - Φ)/e = (h/e)f - Φ/e. تبدأ من f₀ وتزداد مع التردد.",
            3: f"📌 <b>الخطوة 3:</b> ميل الخط = h/e = {H_PLANCK/E_CHARGE:.2e} V·s. من قياس الميل تجريبياً يمكن حساب ثابت بلانك h!",
            4: f"📌 <b>الخطوة 4:</b> النقطة الحمراء تمثّل حالة الاختيار الحالية: λ = {wl4_nm} nm. {vs_display}",
            5: "📌 <b>الخطوة 5:</b> مقارنة عدة فلزات: جميع الخطوط متوازية (نفس الميل h/e) لكنها تبدأ من f₀ مختلف. الفلز ذو Φ الأصغر يبدأ أبكر (f₀ أصغر)."
        }
        st.markdown(f'<div class="explain-box">{exp4[step4]}</div>', unsafe_allow_html=True)
    # Calculator section
    st.markdown("---")
    st.markdown("#### 🧮 آلة حاسبة للظاهرة الكهروضوئية")
    c1, c2, c3 = st.columns(3)
    with c1:
        calc_mode = st.radio("أدخل:", ["التردد f (Hz)", "الطول الموجي λ (nm)"])
    with c2:
        if calc_mode == "التردد f (Hz)":
            calc_f = st.number_input("أدخل التردد (×10¹⁴ Hz)", 0.1, 30.0, 8.0, step=0.1, key="calc_f")
            calc_freq = calc_f * 1e14
        else:
            calc_wl = st.number_input("أدخل الطول الموجي (nm)", 50, 900, 400, step=5, key="calc_wl")
            calc_freq = C_LIGHT / (calc_wl * 1e-9)
    with c3:
        calc_mat = st.selectbox("اختر الفلز", list(MATERIALS.keys()), key="calc_mat")

    calc_phi = MATERIALS[calc_mat]["phi_eV"]
    calc_res = compute_photoelectric(calc_phi, calc_freq)
    calc_photon_E = calc_res["photon_E_eV"]

    st.markdown(f"""
    <div class="formula-box" style="font-size:1em;">
        f = {calc_freq:.2e} Hz &nbsp;|&nbsp; E<sub>photon</sub> = {calc_photon_E:.3f} eV &nbsp;|&nbsp; 
        Φ = {calc_phi} eV &nbsp;|&nbsp; f<sub>0</sub> = {calc_res["f0"]:.2e} Hz
    </div>
    """, unsafe_allow_html=True)

    if calc_res["emitting"]:
        ke_val = calc_res["KE_max_eV"]
        vs_val = calc_res["Vs"]
        v_max = np.sqrt(2 * ke_val * EV_TO_J / M_ELECTRON)
        st.markdown(f"""
        <div class="result-box">
            <b>KE<sub>max</sub> = hf - Φ = {calc_photon_E:.3f} - {calc_phi} = {ke_val:.3f} eV</b><br>
            <b>V<sub>s</sub> = {vs_val:.3f} V</b><br>
            <b>v<sub>max</sub> = √(2 × KE<sub>max</sub> / m<sub>e</sub>) = {v_max:.2e} m/s</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            <b>f &lt; f₀ → لا انبعاث</b> (E_photon = {calc_photon_E:.3f} eV &lt; Φ = {calc_phi} eV)
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# TAB 5: COMPTON EFFECT
# ======================================================================
with tab5:
    st.markdown("""
    <div class="concept-card">
        <h3>🌀 ظاهرة كومبتون | Compton Effect</h3>
        <p>عند سقوط <b>أشعة سينيّة</b> على هدف (غرافيت)، لاحظ كومبتون أن الأشعة المشتَّتة لها 
        <b>طول موجي أطول</b> من الأشعة الساقطة (λ<sub>f</sub> > λ<sub>i</sub>).</p>
        <p>النموذج الموجي للضوء <b>عجز</b> عن تفسير ذلك، لكن النموذج الجسيمي نجح باستخدام 
        <b>حفظ الطاقة والزخم الخطي</b>.</p>
    </div>
    <div class="formula-box">
        Δλ = λ<sub>f</sub> - λ<sub>i</sub> = (h / m<sub>e</sub>c)(1 - cosθ) = λ<sub>C</sub>(1 - cosθ)
    </div>
    <div class="explain-box">
        <p>حيث <b>λ<sub>C</sub> = h/m<sub>e</sub>c = 2.43 × 10<sup>-12</sup> m</b> (طول موجة كومبتون للإلكترون) 
        و <b>θ</b> زاوية تشتّت الفوتون.</p>
        <p>الزخم الخطي للفوتون: <b>p = E/c = h/λ = hf/c</b></p>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl5, col_anim5 = st.columns([0.7, 1.3])

    with col_ctrl5:
        st.markdown("#### 🎛️ التحكم بالتجربة")
        lambda_i_nm = st.number_input("طول موجة الأشعة الساقطة λ_i (pm)",
                                        1.0, 100.0, 10.0, step=0.5, key="comp_wl")
        lambda_i_m = lambda_i_nm * 1e-12
        theta_deg = st.slider("زاوية التشتت θ (°)", 0, 180, 60, key="comp_theta")
        theta_rad = np.radians(theta_deg)

        res5 = compute_compton(lambda_i_m, theta_rad)

        dlambda_pm = res5["dlambda"] * 1e12
        lambdaf_pm = res5["lambda_f"] * 1e12
        Ei_keV = res5["Ei_eV"]
        Ef_keV = res5["Ef_eV"]
        Ee_keV = res5["Ee_eV"]
        phi_deg_val = np.degrees(res5["phi_rad"])
        p_i_val = H_PLANCK / lambda_i_m

        st.markdown(f"""
        <div class="explain-box">
            <b>📋 مدخلات التجربة:</b><br>
            • λ<sub>i</sub> = <b>{lambda_i_nm} pm</b> = {lambda_i_m:.2e} m<br>
            • θ = <b>{theta_deg}°</b><br>
            • طاقة الفوتون الساقط: E<sub>i</sub> = <b>{Ei_keV:.1f} keV</b><br>
            • زخم الفوتون الساقط: p<sub>i</sub> = <b>{p_i_val:.2e} kg·m/s</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
            <b>📋 نتائج التشتت:</b><br><br>
            • Δλ = λ<sub>C</sub>(1 - cosθ) = <b>{dlambda_pm:.4f} pm</b><br>
            • λ<sub>f</sub> = λ<sub>i</sub> + Δλ = <b>{lambdaf_pm:.4f} pm</b><br>
            • طاقة الفوتون المشتت: E<sub>f</sub> = <b>{Ef_keV:.1f} keV</b><br>
            • طاقة الإلكترون الارتدادي: E<sub>e</sub> = <b>{Ee_keV:.1f} keV</b><br>
            • زاوية ارتداد الإلكترون: φ = <b>{phi_deg_val:.1f}°</b><br><br>
            <i>ملاحظة: λ<sub>f</sub> &gt; λ<sub>i</sub> → الفوتون المشتت أقل طاقة (طاقة أقل → تردد أقل → طول موجي أطول)</i>
        </div>
        """, unsafe_allow_html=True)

        if theta_deg == 0:
            explain_comp = "عند θ = 0°: لا يحدث تصادم فعلي، Δλ = 0 (الفوتون يستمر بنفس الطاقة)."
        elif theta_deg == 180:
            explain_comp = (f"عند θ = 180° (تشتيث خلفي): يحدث <b>أقصى تغيّر</b> في الطول الموجي. "
                          f"Δλ = 2λ<sub>C</sub> = {dlambda_pm:.4f} pm. الفوتون يرتدّ عكسياً ويفقد أكبر كمية من الطاقة.")
        else:
            explain_comp = (f"عند θ = {theta_deg}°: يفقد الفوتون جزءاً من طاقته للإلكترون. "
                          f"كلما زادت θ، زاد Δλ وزادت الطاقة المفقودة.")

        st.markdown(f'<div class="explain-box"><b>🔍 ما الذي يحدث؟</b><br>{explain_comp}</div>',
                    unsafe_allow_html=True)

    with col_anim5:
        st.markdown("#### 🎬 محاكاة تشتت كومبتون")
        components.html(compton_animation_html(theta_deg, lambda_i_nm), height=420)

    # Compton curve
    st.markdown("---")
    st.markdown("#### 📈 العلاقة بين Δλ وزاوية التشتت θ")
    col_comp_graph, comp_explain_col = st.columns([1.2, 0.8])

    with col_comp_graph:
        thetas = np.linspace(0, 180, 200)
        dlamdas = COMPTON_WL * (1 - np.cos(np.radians(thetas))) * 1e12

        fig5, ax5 = plt.subplots(figsize=(9, 5))
        fig5.patch.set_facecolor('#0d1117')
        ax5.set_facecolor('#0d1117')
        ax5.tick_params(colors='#aaa')
        ax5.spines['bottom'].set_color('#444')
        ax5.spines['left'].set_color('#444')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.set_xlabel('Scattering Angle θ (°)', color='#ccc', fontsize=11)
        ax5.set_ylabel('Δλ (pm)', color='#ccc', fontsize=11)

        ax5.plot(thetas, dlamdas, color='#ff6644', linewidth=2.5, label='Δλ = λC(1 - cosθ)')
        ax5.fill_between(thetas, dlamdas, alpha=0.1, color='#ff6644')

        current_dlam = COMPTON_WL * (1 - np.cos(theta_rad)) * 1e12
        ax5.plot(theta_deg, current_dlam, 'o', color='#00ff88', markersize=12, zorder=5)
        ax5.annotate(f'θ={theta_deg}°, Δλ={current_dlam:.3f} pm',
                    xy=(theta_deg, current_dlam),
                    xytext=(theta_deg + 15, current_dlam + 1),
                    fontsize=10, color='#00ff88',
                    arrowprops=dict(arrowstyle='->', color='#00ff88'),
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88'))

        ax5.plot(0, 0, 's', color='#ffaa00', markersize=8)
        ax5.annotate('θ=0°: Δλ=0', xy=(0, 0), xytext=(15, 0.5),
                    fontsize=9, color='#ffaa00',
                    arrowprops=dict(arrowstyle='->', color='#ffaa00'))
        ax5.plot(180, COMPTON_WL * 2 * 1e12, 's', color='#ff4444', markersize=8)
        ax5.annotate(f'θ=180°: Δλ=2λC={COMPTON_WL*2*1e12:.3f} pm',
                    xy=(180, COMPTON_WL * 2 * 1e12), xytext=(130, COMPTON_WL * 2 * 1e12 + 0.5),
                    fontsize=9, color='#ff4444',
                    arrowprops=dict(arrowstyle='->', color='#ff4444'))

        ax5.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with comp_explain_col:
        st.markdown("""
        <div class="explain-box">
            <b>📖 تفسير ظاهرة كومبتون:</b><br><br>
            <b>1.</b> الفوتون الساقط يتصادم مع إلكترون (ساكن تقريباً)<br><br>
            <b>2.</b> يطبّق قانونا <b>حفظ الطاقة</b> و<b>حفظ الزخم الخطي</b><br><br>
            <b>3.</b> الفوتون يفقد جزءاً من طاقته → λ<sub>f</sub> &gt; λ<sub>i</sub><br><br>
            <b>4.</b> الإلكترون يكتسب الطاقة المفقودة ويصطرد<br><br>
            <b>5.</b> عند θ = 0°: لا تصادم فعلي<br>
            <b>6.</b> عند θ = 180°: أقصى فقدان للطاقة
        </div>
        <div class="warning-box">
            <b>❌ لماذا فشل النموذج الموجي؟</b><br>
            الموجة لا يمكنها "تفقد" جزءاً من طاقها عند التشتت بشكل يعتمد على الزاوية. 
            لكن الفوتون (جسيم) يمكنه نقل جزء من طاقته وزخمه للإلكترون كما في أي تصادم.
        </div>
        <div class="result-box">
            <b>✅ أهمية ظاهرة كومبتون:</b><br>
            دليل قوي على أن للضوء <b>طبيعة جسيمية</b>: 
            الفوتون يحمل <b>زخماً خطياً</b> p = h/λ 
            ويطبّق عليه قوانين التصادم!
        </div>
        """, unsafe_allow_html=True)

    # Compton calculator
    st.markdown("---")
    st.markdown("#### 🧮 آلة حاسبة لظاهرة كومبتون")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        comp_calc_wl = st.number_input("λ_i (pm)", 1.0, 200.0, 71.2, step=0.1, key="comp_calc_wl")
    with cc2:
        comp_calc_theta = st.number_input("θ (°)", 0, 180, 90, key="comp_calc_theta")
    with cc3:
        comp_calc_m = comp_calc_wl * 1e-12
        comp_calc_r = compute_compton(comp_calc_m, np.radians(comp_calc_theta))
        ccr_dl = comp_calc_r["dlambda"] * 1e12
        ccr_lf = comp_calc_r["lambda_f"] * 1e12
        st.markdown(f"""
        <div class="formula-box" style="font-size:0.9em;">
            Δλ = {ccr_dl:.4f} pm<br>
            λ_f = {ccr_lf:.4f} pm
        </div>
        """, unsafe_allow_html=True)
    with cc4:
        ccr_ei = comp_calc_r["Ei_eV"]
        ccr_ef = comp_calc_r["Ef_eV"]
        ccr_ee = comp_calc_r["Ee_eV"]
        st.markdown(f"""
        <div class="formula-box" style="font-size:0.9em;">
            E_i = {ccr_ei:.2f} keV<br>
            E_f = {ccr_ef:.2f} keV<br>
            E_e = {ccr_ee:.2f} keV
        </div>
        """, unsafe_allow_html=True)

    # Detailed step-by-step Compton explanation
    st.markdown("---")
    st.markdown("#### 📝 تحليل خطوة بخطوة لظاهرة كومبتون")
    comp_step = st.slider("اختر الخطوة التفصيلية", 1, 7, 1, key="comp_detail_step")

    comp_steps_explain = {
        1: f"""
        <div class="explain-box">
            <span class="step-indicator">1</span>
            <b>الفوتون الساقط:</b><br>
            فوتون أشعة سينية بطول موجي λ<sub>i</sub> = {lambda_i_nm} pm يسقط على إلكترون ساكن في هدف الغرافيت.<br><br>
            طاقة الفوتون: E<sub>i</sub> = hc/λ<sub>i</sub> = <b>{Ei_keV:.1f} keV</b><br>
            زخمه الخطي: p<sub>i</sub> = h/λ<sub>i</sub> = <b>{p_i_val:.2e} kg·m/s</b><br><br>
            <i>ملاحظة: طاقة الإلكترون في الغرافيت صغيرة جداً مقارنة بطاقة الفوتون، لذلك نعتبر الإلكترون ساكناً.</i>
        </div>
        """,
        2: f"""
        <div class="explain-box">
            <span class="step-indicator">2</span>
            <b>التصادم:</b><br>
            يتصادم الفوتون بالإلكترون كما يتصادم أي جسيمان. الفوتون ينحرف بزاوية θ = <b>{theta_deg}°</b> 
            ويكتسب الإلكترون طاقة وزخماً.<br><br>
            <i>هذا السلوك لا يمكن تفسيره بالنموذج الموجي! الموجة لا تتصادم مع جسيم وتنحرف بزاوية محددة.</i>
        </div>
        """,
        3: """
        <div class="explain-box">
            <span class="step-indicator">3</span>
            <b>حفظ الزخم الخطي:</b><br>
            الزخم كمية متجهة، لذلك نحلّل إلى مركّبتين:<br>
            • المحور x: p<sub>i</sub> = p<sub>f</sub>cosθ + p<sub>e</sub>cosφ<br>
            • المحور y: 0 = p<sub>f</sub>sinθ - p<sub>e</sub>sinφ<br><br>
            حيث p<sub>f</sub> = h/λ<sub>f</sub> (زخم الفوتون المشتت) و p<sub>e</sub> (زخم الإلكترون الارتدادي)
        </div>
        """,
        4: f"""
        <div class="explain-box">
            <span class="step-indicator">4</span>
            <b>حفظ الطاقة:</b><br>
            E<sub>i</sub> = E<sub>f</sub> + E<sub>e</sub><br>
            {Ei_keV:.1f} keV = {Ef_keV:.1f} keV + {Ee_keV:.1f} keV<br><br>
            <i>الطاقة التي يفقدها الفوتون تنتقل كاملة إلى الإلكترون الارتدادي.</i>
        </div>
        """,
        5: f"""
        <div class="explain-box">
            <span class="step-indicator">5</span>
            <b>حساب التغيّر في الطول الموجي:</b><br>
            بدمج قانونَي الحفظ نحصل على:<br>
            <div class="formula-box" style="margin: 10px 0;">Δλ = (h/m<sub>e</sub>c)(1 - cosθ)</div>
            Δλ = {COMPTON_WL*1e12:.3f} × (1 - cos{theta_deg}°) = <b>{dlambda_pm:.4f} pm</b><br><br>
            <i>لاحظ أن Δλ لا يعتمد على λ<sub>i</sub>! يعتمد فقط على زاوية التشتت θ.</i>
        </div>
        """,
        6: f"""
        <div class="explain-box">
            <span class="step-indicator">6</span>
            <b>الطول الموجي للفوتون المشتت:</b><br>
            λ<sub>f</sub> = λ<sub>i</sub> + Δλ = {lambda_i_nm} + {dlambda_pm:.4f} = <b>{lambdaf_pm:.4f} pm</b><br><br>
            بما أن λ<sub>f</sub> &gt; λ<sub>i</sub> فإن:<br>
            • f<sub>f</sub> &lt; f<sub>i</sub> (التردد قلّ)<br>
            • E<sub>f</sub> &lt; E<sub>i</sub> (الطاقة قلّت)<br>
            • السرعة تبقى c (فوتون دائماً يتحرك بسرعة الضوء)
        </div>
        """,
        7: """
        <div class="result-box">
            <span class="step-indicator">7</span>
            <b>الخلاصة - لماذا تُعدّ ظاهرة كومبتون دليلاً حاسماً؟</b><br><br>
            ✅ الفوتون يتصرف كـ<b>جسيم</b> يحمل زخماً خطياً p = h/λ<br>
            ✅ يطبّق عليه قوانين التصادم (حفظ الطاقة والزخم)<br>
            ✅ النموذج الموجي عجز تماماً عن تفسير زيادة الطول الموجي<br>
            ✅ تُكمّل ظاهرة كومبتون الظاهرة الكهروضوئية في إثبات <b>الطبيعة الجسيمية للضوء</b><br><br>
            <b>الضوء إذن له ثنائية: موجيّة (حيود، تداخل) وجسيميّة (كهروضوئية، كومبتون)</b>
        </div>
        """
    }
    st.markdown(comp_steps_explain[comp_step], unsafe_allow_html=True)


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #0f0c29, #302b63);
border-radius: 12px; color: #888; font-family: 'Noto Sans Arabic', sans-serif;">
    <p style="color: #b8c6db; font-size: 1.1em;">
        الفيزياء الحديثة - الطبيعة الجسيمية للضوء | Modern Physics - Particle Nature of Light
    </p>
    <p style="color: #f7971e; font-size: 1.3em; font-weight: 700; margin: 8px 0;">
        إعداد Israa Youssuf Samara
    </p>
    <p style="font-size: 0.85em; color: #666;">
        الوَحْدة: الفيزياء الحديثة | الدرس الأول: الطبيعة الجسيمية للضوء
    </p>
</div>
""", unsafe_allow_html=True)
