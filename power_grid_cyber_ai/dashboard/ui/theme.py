"""PowerGrid Cyber AI - shared "Cyber-Grid Command Center" theme.

Usage (one line per page, right after st.set_page_config):
    from ui.theme import apply_theme
    apply_theme()            # Arabic pages: apply_theme(rtl=True)

Only CSS / HTML / Plotly. No program logic lives here.
"""
from __future__ import annotations

import html as _html
import random
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Plotly is required by every chart helper in this theme module.
# Import it directly so a missing/broken installation fails with a clear
# message instead of causing: AttributeError: 'NoneType' object has no attribute 'Figure'
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Palette (semantic state colors are constant across every page)
# --------------------------------------------------------------------------
CYAN = "#00e5ff"
VIOLET = "#7c4dff"
STATE_COLORS = {
    "Normal": "#00ff9d",
    "Fault": "#ffb020",
    "Attack": "#ff2d55",
}
STATE_ICONS = {"Normal": "🟢", "Fault": "🟠", "Attack": "🔴"}
CLASS_ORDER = ["Normal", "Fault", "Attack"]
PLOT_FONT = "Cairo, Orbitron, sans-serif"


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _lerp(c1: str, c2: str, t: float) -> str:
    a = [int(c1.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _pack(markup: str) -> str:
    """Collapse markup so Streamlit's markdown parser never sees blank lines
    or 4-space indents (which would turn HTML into a code block)."""
    markup = re.sub(r"\n\s*", "", markup.strip())
    return markup


def render(markup: str) -> None:
    st.markdown(_pack(markup), unsafe_allow_html=True)


def _esc(value) -> str:
    return _html.escape(str(value))


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Cairo:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
@property --num{syntax:'<integer>';initial-value:0;inherits:false}
:root{
 --bg0:#05070d;--bg1:#0a0f1f;--cyan:#00e5ff;--violet:#7c4dff;
 --normal:#00ff9d;--fault:#ffb020;--attack:#ff2d55;
 --state:#00e5ff;--state-rgb:0,229,255;
 --text:#d7e6ff;--muted:#7f93b5;--line:rgba(0,229,255,.22);
 --glass:rgba(10,16,34,.66);
 --f-display:'Orbitron','Cairo',sans-serif;--f-ui:'Cairo','Tajawal',sans-serif;--f-mono:'JetBrains Mono',ui-monospace,monospace;
}
html,body{background:radial-gradient(1200px 800px at 50% -10%,#0d1530 0%,var(--bg0) 62%) fixed!important;color:var(--text)}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stMain"],.main{background:transparent!important}
.stApp{font-family:var(--f-ui);color:var(--text)}
[data-testid="stAppViewContainer"],[data-testid="stSidebar"],[data-testid="stHeader"]{position:relative;z-index:2}
.block-container{padding-top:2.2rem;max-width:1240px}
h1,h2,h3,h4{font-family:var(--f-display);letter-spacing:.03em;color:#eaf6ff}
h2,h3{text-shadow:0 0 18px rgba(var(--state-rgb),.35)}
code,pre,[data-testid="stCode"]{font-family:var(--f-mono)!important}
.num{direction:ltr;unicode-bidi:isolate;display:inline-block}
::selection{background:rgba(var(--state-rgb),.35)}
a{color:var(--cyan)}
:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}

/* ---------- animated background ---------- */
div[data-testid="stElementContainer"]:has(.pg-bg),div.element-container:has(.pg-bg){position:absolute!important;height:0!important;width:0!important;margin:0!important;padding:0!important;overflow:visible!important}
.pg-bg{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.pg-bg *{pointer-events:none}
.pg-glow{position:absolute;width:60vmax;height:60vmax;border-radius:50%;filter:blur(90px);will-change:transform}
.pg-g1{left:-18vmax;top:-24vmax;opacity:.30;background:radial-gradient(circle,var(--state) 0%,transparent 65%);animation:pgDrift1 26s ease-in-out infinite alternate}
.pg-g2{right:-24vmax;bottom:-28vmax;opacity:.26;background:radial-gradient(circle,var(--violet) 0%,transparent 65%);animation:pgDrift2 34s ease-in-out infinite alternate}
.pg-g3{left:32vw;top:34vh;width:38vmax;height:38vmax;opacity:.13;background:radial-gradient(circle,var(--cyan) 0%,transparent 65%);animation:pgDrift3 40s ease-in-out infinite alternate}
@keyframes pgDrift1{to{transform:translate3d(20vw,16vh,0) scale(1.15)}}
@keyframes pgDrift2{to{transform:translate3d(-22vw,-14vh,0) scale(1.2)}}
@keyframes pgDrift3{to{transform:translate3d(-16vw,10vh,0) scale(.85)}}
.pg-gridwrap{position:absolute;left:-40%;right:-40%;bottom:0;height:64vh;perspective:380px;perspective-origin:50% 0%;overflow:hidden;
 -webkit-mask-image:linear-gradient(to top,#000 5%,transparent 92%);mask-image:linear-gradient(to top,#000 5%,transparent 92%)}
.pg-grid{position:absolute;left:0;right:0;top:-80px;height:260%;transform-origin:50% 0%;transform:rotateX(64deg);will-change:transform;
 background-image:linear-gradient(rgba(var(--state-rgb),.55) 1px,transparent 1px),linear-gradient(90deg,rgba(var(--state-rgb),.55) 1px,transparent 1px);
 background-size:64px 64px;animation:pgGrid 2.6s linear infinite}
@keyframes pgGrid{from{transform:rotateX(64deg) translateY(0)}to{transform:rotateX(64deg) translateY(64px)}}
.pg-circuit{position:absolute;inset:0;width:100%;height:100%;opacity:.55}
.pg-circuit .tr{stroke:rgba(var(--state-rgb),.20);stroke-width:1.2;fill:none}
.pg-circuit .pu{stroke:var(--state);stroke-width:2.4;fill:none;stroke-linecap:round;stroke-dasharray:5 95;filter:drop-shadow(0 0 4px var(--state));animation:pgPulse 7s linear infinite}
.pg-circuit .nd{fill:var(--state);opacity:.7;animation:pgNode 3.2s ease-in-out infinite}
@keyframes pgPulse{from{stroke-dashoffset:100}to{stroke-dashoffset:0}}
@keyframes pgNode{50%{opacity:.15}}
.pg-p{position:absolute;bottom:-12px;left:var(--x);width:var(--s);height:var(--s);border-radius:50%;background:var(--state);box-shadow:0 0 10px var(--state);opacity:0;animation:pgRise var(--d) linear var(--dl) infinite;will-change:transform,opacity}
@keyframes pgRise{0%{transform:translate3d(0,0,0);opacity:0}10%{opacity:.85}90%{opacity:.85}100%{transform:translate3d(var(--dx),-112vh,0);opacity:0}}
.pg-scan{position:absolute;left:0;right:0;top:0;height:180px;background:linear-gradient(to bottom,transparent,rgba(var(--state-rgb),.06) 60%,rgba(var(--state-rgb),.20) 99%,transparent);animation:pgScan 10s linear infinite;will-change:transform}
@keyframes pgScan{from{transform:translateY(-220px)}to{transform:translateY(105vh)}}
.pg-crt{position:absolute;inset:0;background:repeating-linear-gradient(to bottom,rgba(255,255,255,.022) 0 1px,transparent 1px 3px);opacity:.6}
.pg-noise{position:absolute;inset:-10%;opacity:.045;background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");animation:pgNoise .9s steps(4) infinite}
@keyframes pgNoise{25%{transform:translate(-3%,2%)}50%{transform:translate(2%,-3%)}75%{transform:translate(-2%,-2%)}}
.pg-vignette{position:absolute;inset:0;background:radial-gradient(ellipse at center,transparent 55%,rgba(0,0,0,.55) 100%)}
.pg-alert{position:absolute;inset:0;opacity:0;box-shadow:inset 0 0 180px 30px rgba(var(--state-rgb),.55)}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(8,12,26,.93),rgba(5,7,13,.96))!important;border-inline-end:1px solid var(--line);backdrop-filter:blur(14px)}
[data-testid="stSidebarNav"] a,[data-testid="stSidebarNavLink"]{border-radius:2px;border:1px solid transparent;transition:transform .2s,border-color .2s,background .2s}
[data-testid="stSidebarNav"] a:hover,[data-testid="stSidebarNavLink"]:hover{background:rgba(0,229,255,.08)!important;transform:translateX(3px)}
[data-testid="stSidebarNav"] a[aria-current="page"],[data-testid="stSidebarNavLink"][aria-current="page"]{background:linear-gradient(90deg,rgba(0,229,255,.18),transparent)!important;border-color:rgba(0,229,255,.55);box-shadow:0 0 18px rgba(0,229,255,.25)}
.brand{display:flex;align-items:center;gap:12px;padding:6px 2px 10px}
.brand-logo{position:relative;width:46px;height:46px;flex:none;display:grid;place-items:center;font-size:22px}
.brand-logo::before{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid transparent;border-top-color:var(--cyan);border-right-color:var(--violet);animation:pgSpin 3.2s linear infinite;box-shadow:0 0 14px rgba(0,229,255,.45)}
.brand-logo b{animation:pgBolt 2.4s ease-in-out infinite;filter:drop-shadow(0 0 8px var(--cyan))}
@keyframes pgSpin{to{transform:rotate(360deg)}}
@keyframes pgBolt{50%{transform:scale(1.18);opacity:.75}}
.brand-name{font-family:var(--f-display);font-weight:900;font-size:15px;line-height:1.1;color:#fff;letter-spacing:.06em}
.brand-name small{display:block;font-weight:500;font-size:10px;color:var(--cyan);letter-spacing:.22em;margin-top:4px}
.online{display:flex;align-items:center;gap:8px;font-family:var(--f-mono);font-size:11px;color:var(--normal);letter-spacing:.14em;margin:4px 0 8px}
.sim{border:1px dashed rgba(255,176,32,.6);background:rgba(255,176,32,.07);color:#ffd27a;padding:7px 10px;font-size:12px;line-height:1.5}
.sim b{font-family:var(--f-display);letter-spacing:.12em;font-size:11px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--state);box-shadow:0 0 10px var(--state);animation:pgDot 1.6s ease-in-out infinite;flex:none}
@keyframes pgDot{50%{transform:scale(.55);opacity:.45}}

/* ---------- hero ---------- */
.hero{position:relative;padding:8px 0 22px;margin-bottom:6px}
.hero-title{position:relative;display:inline-block;margin:0;font-family:var(--f-display);font-weight:900;font-size:clamp(28px,5vw,54px);line-height:1.15;letter-spacing:.04em;
 background:linear-gradient(90deg,var(--cyan),#ffffff 30%,var(--violet) 55%,var(--cyan));background-size:220% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:pgShine 7s linear infinite}
.hero-title::before,.hero-title::after{content:attr(data-text);position:absolute;inset:0;background:none;-webkit-text-fill-color:currentColor;opacity:0;pointer-events:none}
.hero-title::before{color:#ff2d55;animation:pgGlitchA 6s steps(1) infinite}
.hero-title::after{color:#00e5ff;animation:pgGlitchB 6s steps(1) infinite}
@keyframes pgShine{to{background-position:220% 0}}
@keyframes pgGlitchA{0%,92%,100%{opacity:0;transform:none}93%{opacity:.8;transform:translate(-4px,1px)}95%{opacity:.8;transform:translate(3px,-1px)}97%{opacity:0}}
@keyframes pgGlitchB{0%,92%,100%{opacity:0;transform:none}94%{opacity:.8;transform:translate(4px,-1px)}96%{opacity:.8;transform:translate(-3px,1px)}98%{opacity:0}}
.hero-sub{margin:8px 0 0;color:var(--muted);font-size:16px;max-width:760px;clip-path:inset(0 100% 0 0);animation:pgType 2.4s steps(36,end) .5s forwards}
.rtl .hero-sub{clip-path:inset(0 0 0 100%);animation-name:pgTypeR}
@keyframes pgType{to{clip-path:inset(0 0 0 0)}}
@keyframes pgTypeR{to{clip-path:inset(0 0 0 0)}}
.hero-line{height:2px;margin-top:16px;background:linear-gradient(90deg,var(--cyan),var(--violet),transparent);box-shadow:0 0 14px var(--cyan);width:min(520px,70%)}
.rtl .hero-line{background:linear-gradient(270deg,var(--cyan),var(--violet),transparent)}

/* ---------- section header ---------- */
.sec{display:flex;align-items:center;gap:12px;margin:26px 0 10px}
.sec-bar{width:4px;height:26px;background:linear-gradient(var(--state),var(--violet));box-shadow:0 0 12px var(--state)}
.sec h3{margin:0;font-size:20px}
.sec p{margin:0;color:var(--muted);font-size:13px}

/* ---------- HUD cards ---------- */
.hud{--cut:14px;position:relative;padding:1px;height:100%;
 background:linear-gradient(135deg,rgba(var(--state-rgb),.75),rgba(124,77,255,.35) 55%,rgba(var(--state-rgb),.25));
 clip-path:polygon(var(--cut) 0,100% 0,100% calc(100% - var(--cut)),calc(100% - var(--cut)) 100%,0 100%,0 var(--cut));
 transition:transform .25s ease}
.hud:hover{transform:translateY(-4px)}
.hud>.in{height:100%;box-sizing:border-box;padding:16px 18px;background:linear-gradient(160deg,rgba(14,22,46,.82),rgba(7,11,24,.88));backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
 clip-path:polygon(var(--cut) 0,100% 0,100% calc(100% - var(--cut)),calc(100% - var(--cut)) 100%,0 100%,0 var(--cut))}
.mc-top{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
.mc-ico{font-size:17px;filter:drop-shadow(0 0 6px var(--state))}
.mc-val{display:block;margin-top:8px;font-family:var(--f-display);font-weight:900;font-size:clamp(26px,3.2vw,40px);color:#fff;text-shadow:0 0 16px rgba(var(--state-rgb),.7);line-height:1.1;word-break:break-word}
.mc-hint{margin-top:6px;color:var(--muted);font-size:12px;min-height:16px}
.mc-bar{height:3px;margin-top:10px;background:rgba(255,255,255,.07);overflow:hidden}
.mc-bar i{display:block;height:100%;width:100%;background:linear-gradient(90deg,var(--state),transparent);transform-origin:left;animation:pgBar 1.4s ease-out both}
.rtl .mc-bar i{transform-origin:right}
@keyframes pgBar{from{transform:scaleX(0)}}
.cnt{--num:var(--to);counter-reset:n var(--num);animation:pgCount 1.3s ease-out both}
.cnt::after{content:counter(n)}
@keyframes pgCount{from{--num:0}}
.result{display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.result .big{display:block;font-family:var(--f-display);font-weight:900;font-size:clamp(34px,5vw,60px);color:var(--state);text-shadow:0 0 26px var(--state);line-height:1}
.result .sub{color:var(--muted);font-size:13px;margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:9px;padding:6px 16px;font-family:var(--f-display);font-weight:700;font-size:14px;letter-spacing:.12em;color:var(--state);
 border:1px solid var(--state);background:rgba(var(--state-rgb),.10);box-shadow:0 0 18px rgba(var(--state-rgb),.35),inset 0 0 12px rgba(var(--state-rgb),.12)}
.badge.sm{padding:2px 10px;font-size:11px;letter-spacing:.08em}
.badge.sm .dot{width:7px;height:7px}
.flow{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 4px}
.flow span{font-size:13px;color:var(--muted);padding:5px 12px;border:1px solid var(--line);background:rgba(10,16,34,.55)}
.flow span.on{color:#04121a;background:linear-gradient(90deg,var(--cyan),var(--violet));border-color:transparent;font-weight:700;box-shadow:0 0 16px rgba(0,229,255,.45)}

/* ---------- confusion matrix ---------- */
.cm{direction:ltr;display:grid;gap:6px;font-family:var(--f-mono);margin:8px 0}
.cm .h{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;padding:6px;letter-spacing:.08em}
.cm .c{position:relative;display:grid;place-items:center;min-height:84px;font-family:var(--f-display);font-weight:900;font-size:26px;color:#fff;border:1px solid rgba(0,229,255,.18);background:rgba(var(--rgb),var(--a))}
.cm .c small{display:block;font-family:var(--f-mono);font-weight:400;font-size:10px;color:rgba(255,255,255,.7)}
.cm .crit{border:2px solid var(--attack);animation:pgCrit 1.5s ease-in-out infinite}
.cm .crit::before{content:"CRITICAL";position:absolute;top:4px;left:6px;font-family:var(--f-mono);font-size:9px;letter-spacing:.14em;color:#ff7d94}
@keyframes pgCrit{0%,100%{box-shadow:0 0 0 rgba(255,45,85,0),inset 0 0 0 rgba(255,45,85,0)}50%{box-shadow:0 0 22px rgba(255,45,85,.85),inset 0 0 18px rgba(255,45,85,.35)}}
.cm-axis{direction:ltr;display:flex;justify-content:space-between;color:var(--muted);font-size:12px;font-family:var(--f-mono);margin-top:2px}

/* ---------- tables ---------- */
.tbl-wrap{overflow-x:auto;border:1px solid var(--line);box-shadow:0 0 26px rgba(0,229,255,.07);background:rgba(8,12,26,.55);backdrop-filter:blur(10px)}
table.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl th{position:sticky;top:0;padding:11px 14px;text-align:start;font-family:var(--f-display);font-size:12px;letter-spacing:.1em;color:var(--cyan);background:rgba(0,229,255,.08);text-shadow:0 0 10px rgba(0,229,255,.7);border-bottom:1px solid var(--line);white-space:nowrap}
.tbl td{padding:10px 14px;border-bottom:1px solid rgba(0,229,255,.07);font-family:var(--f-mono);font-size:13px;word-break:break-word}
.tbl tr:nth-child(even) td{background:rgba(124,77,255,.06)}
.tbl tr:hover td{background:rgba(0,229,255,.09)}
[data-testid="stDataFrame"]{border:1px solid var(--line);box-shadow:0 0 26px rgba(0,229,255,.07)}

/* ---------- native widgets ---------- */
[data-testid="stMetric"]{position:relative;padding:14px 16px;background:linear-gradient(160deg,rgba(14,22,46,.8),rgba(7,11,24,.86));border:1px solid var(--line);backdrop-filter:blur(14px);transition:transform .25s}
[data-testid="stMetric"]:hover{transform:translateY(-3px)}
[data-testid="stMetric"]::before{content:"";position:absolute;top:-1px;left:-1px;width:14px;height:14px;border-top:2px solid var(--state);border-left:2px solid var(--state)}
[data-testid="stMetric"]::after{content:"";position:absolute;bottom:-1px;right:-1px;width:14px;height:14px;border-bottom:2px solid var(--state);border-right:2px solid var(--state)}
[data-testid="stMetricValue"]{font-family:var(--f-display);font-weight:900;color:#fff;text-shadow:0 0 14px rgba(var(--state-rgb),.6);direction:ltr;unicode-bidi:isolate}
[data-testid="stMetricLabel"]{color:var(--muted)}
.stButton button,.stDownloadButton button,[data-testid^="stBaseButton"]{position:relative;overflow:hidden;border-radius:2px;border:1px solid rgba(0,229,255,.55);background:linear-gradient(135deg,rgba(0,229,255,.14),rgba(124,77,255,.16));color:#eaffff;font-family:var(--f-ui);font-weight:700;letter-spacing:.03em;transition:transform .15s,box-shadow .25s,border-color .25s}
.stButton button:hover,.stDownloadButton button:hover{box-shadow:0 0 24px rgba(0,229,255,.55);border-color:var(--cyan);transform:translateY(-1px);color:#fff}
.stButton button:active,.stDownloadButton button:active{transform:scale(.96)}
.stButton button::after,.stDownloadButton button::after{content:"";position:absolute;inset:0;background:radial-gradient(circle,rgba(255,255,255,.45) 0%,transparent 60%);opacity:0;transform:scale(.2);transition:transform .5s,opacity .5s}
.stButton button:active::after,.stDownloadButton button:active::after{opacity:1;transform:scale(2.4);transition:0s}
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--cyan),var(--violet))!important;color:#031018!important;border-color:transparent!important;box-shadow:0 0 22px rgba(0,229,255,.4)}
[data-testid="stFileUploaderDropzone"]{border:none!important;padding:26px;background:
 linear-gradient(90deg,var(--cyan) 50%,transparent 50%) 0 0/18px 2px repeat-x,
 linear-gradient(90deg,var(--cyan) 50%,transparent 50%) 0 100%/18px 2px repeat-x,
 linear-gradient(var(--cyan) 50%,transparent 50%) 0 0/2px 18px repeat-y,
 linear-gradient(var(--cyan) 50%,transparent 50%) 100% 0/2px 18px repeat-y,
 rgba(0,229,255,.05);animation:pgDash 1.2s linear infinite;transition:box-shadow .25s}
[data-testid="stFileUploaderDropzone"]:hover{box-shadow:0 0 30px rgba(0,229,255,.3)}
@keyframes pgDash{to{background-position:18px 0,-18px 100%,0 18px,100% -18px,0 0}}
button[data-baseweb="tab"]{font-family:var(--f-ui);font-weight:700;color:var(--muted)}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--cyan);text-shadow:0 0 12px rgba(0,229,255,.8)}
[data-baseweb="tab-highlight"]{background:var(--cyan)!important;box-shadow:0 0 14px var(--cyan)}
[data-baseweb="tab-border"]{background:var(--line)!important}
[data-testid="stExpander"] details{border:1px solid var(--line)!important;border-radius:2px!important;background:rgba(10,16,34,.55)}
[data-testid="stExpander"] summary:hover{color:var(--cyan)}
[data-testid="stAlert"]{--k:0,229,255;border-radius:2px;border:1px solid rgba(var(--k),.55);background:rgba(var(--k),.08)!important;box-shadow:0 0 18px rgba(var(--k),.14);backdrop-filter:blur(10px)}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){--k:0,255,157}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){--k:255,176,32}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){--k:255,45,85}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]){border-color:var(--line)}
[data-testid="stSlider"] [role="slider"]{background:var(--state)!important;box-shadow:0 0 14px var(--state)}
[data-baseweb="select"]>div,[data-baseweb="base-input"],[data-baseweb="input"]{background:rgba(10,16,34,.8)!important;border-color:var(--line)!important}
[data-testid="stSpinner"] *{border-top-color:var(--cyan)!important;color:var(--cyan)}
[data-testid="stSpinner"]{filter:drop-shadow(0 0 8px rgba(0,229,255,.7))}
[data-testid="stProgress"] [role="progressbar"]>div>div{background:linear-gradient(90deg,var(--cyan),var(--violet))!important;box-shadow:0 0 12px var(--cyan)}
hr{border-color:var(--line)!important}
[data-testid="stPlotlyChart"]{border:1px solid rgba(0,229,255,.12);background:rgba(8,12,26,.35)}

/* ---------- staggered entrance (page load only) ---------- */
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*{animation:pgIn .6s cubic-bezier(.2,.7,.2,1) both}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(2){animation-delay:.06s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(3){animation-delay:.12s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(4){animation-delay:.18s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(5){animation-delay:.24s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(6){animation-delay:.30s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(n+7){animation-delay:.36s}
@keyframes pgIn{from{opacity:0;transform:translate3d(0,18px,0)}}

/* ---------- RTL ---------- */
.rtl-on .stApp,.rtl-on [data-testid="stSidebar"]{direction:rtl}
.rtl-on [data-testid="stMetric"]{text-align:right}

@media (prefers-reduced-motion:reduce){
 .pg-bg *,.hero-title,.hero-title::before,.hero-title::after,.hero-sub,.hud,.cnt,.mc-bar i,.dot,.brand-logo::before,.brand-logo b,.cm .crit,[data-testid="stFileUploaderDropzone"],[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*{animation:none!important}
 .hero-sub{clip-path:none}
}
"""

_CIRCUITS = [
    "M0 140 H260 L320 200 H640 L700 140 H1020 L1080 200 H1600",
    "M0 330 H180 L240 390 H520 L580 330 H900 L960 270 H1260 L1320 330 H1600",
    "M0 560 H300 L360 620 H760 L820 560 H1180 L1240 620 H1600",
    "M0 760 H220 L280 700 H700 L760 760 H1100 L1160 700 H1600",
    "M420 0 V120 L470 170 V420 L420 470 V900",
    "M1180 0 V210 L1130 260 V560 L1180 610 V900",
]
_NODES = [(260, 140), (640, 200), (1020, 140), (520, 390), (900, 330), (1260, 270),
          (360, 620), (820, 560), (1180, 620), (280, 700), (760, 760), (470, 170), (1130, 260)]


def _background(seed: int = 7) -> str:
    rnd = random.Random(seed)
    traces = "".join(f'<path class="tr" d="{d}"/>' for d in _CIRCUITS)
    pulses = "".join(
        f'<path class="pu" pathLength="100" d="{d}" style="animation-delay:-{i * 1.3:.1f}s;animation-duration:{6 + i}s"/>'
        for i, d in enumerate(_CIRCUITS)
    )
    nodes = "".join(
        f'<circle class="nd" cx="{x}" cy="{y}" r="4" style="animation-delay:-{i * .4:.1f}s"/>'
        for i, (x, y) in enumerate(_NODES)
    )
    particles = "".join(
        f'<i class="pg-p" style="--x:{rnd.randint(2, 98)}vw;--s:{rnd.choice([2, 2, 3, 4])}px;'
        f'--d:{rnd.randint(14, 30)}s;--dl:-{rnd.randint(0, 28)}s;--dx:{rnd.randint(-8, 8)}vw"></i>'
        for _ in range(28)
    )
    return (
        '<div class="pg-bg" aria-hidden="true">'
        '<div class="pg-glow pg-g1"></div><div class="pg-glow pg-g2"></div><div class="pg-glow pg-g3"></div>'
        '<div class="pg-gridwrap"><div class="pg-grid"></div></div>'
        f'<svg class="pg-circuit" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">{traces}{pulses}{nodes}</svg>'
        f'{particles}'
        '<div class="pg-scan"></div><div class="pg-crt"></div><div class="pg-noise"></div>'
        '<div class="pg-vignette"></div><div class="pg-alert"></div></div>'
    )


def _state_vars(state: str | None) -> str:
    color = STATE_COLORS.get(state or "", CYAN)
    return f":root{{--state:{color};--state-rgb:{_rgb(color)}}}"



def _cyber_grid_layer() -> None:
    """Interactive PowerGrid node/data-packet layer injected into the main Streamlit page."""
    components.html(
        """
        <script>
          const doc = window.parent.document;

          const oldCanvas = doc.getElementById("pgGridLayer");
          if (oldCanvas) oldCanvas.remove();

          const canvas = doc.createElement("canvas");
          canvas.id = "pgGridLayer";
          canvas.style.cssText = `
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1;
            pointer-events: none;
            opacity: .8;
          `;
          doc.body.appendChild(canvas);

          const ctx = canvas.getContext("2d");
          let W, H;
          let mouse = { x: -9999, y: -9999 };
          let nodes = [];
          let packets = [];

          const CYAN = "0, 229, 255";
          const VIOLET = "124, 77, 255";

          function resize() {
            W = canvas.width = doc.documentElement.clientWidth;
            H = canvas.height = doc.documentElement.clientHeight;
            buildNodes();
          }

          function buildNodes() {
            nodes = [];
            const count = Math.min(65, Math.floor(W * H / 26000));

            for (let i = 0; i < count; i++) {
              nodes.push({
                x: Math.random() * W,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.16,
                vy: (Math.random() - 0.5) * 0.16,
                r: 1 + Math.random() * 1.5,
                pulse: Math.random() * Math.PI * 2
              });
            }
          }

          function createPacket() {
            if (nodes.length < 2) return;
            const a = nodes[Math.floor(Math.random() * nodes.length)];
            const b = nodes[Math.floor(Math.random() * nodes.length)];
            if (a === b) return;

            packets.push({
              ax: a.x, ay: a.y,
              bx: b.x, by: b.y,
              t: 0,
              speed: 0.005 + Math.random() * 0.009
            });
          }

          doc.addEventListener("mousemove", e => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
          });

          doc.addEventListener("mouseout", () => {
            mouse.x = -9999;
            mouse.y = -9999;
          });

          window.addEventListener("resize", resize);
          resize();

          function draw() {
            ctx.clearRect(0, 0, W, H);

            for (const n of nodes) {
              n.x += n.vx;
              n.y += n.vy;
              n.pulse += 0.022;

              if (n.x < 0 || n.x > W) n.vx *= -1;
              if (n.y < 0 || n.y > H) n.vy *= -1;
            }

            for (let i = 0; i < nodes.length; i++) {
              for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                  const alpha = (1 - dist / 150) * 0.14;

                  ctx.beginPath();
                  ctx.strokeStyle = `rgba(${CYAN}, ${alpha})`;
                  ctx.lineWidth = 0.6;
                  ctx.moveTo(a.x, a.y);
                  ctx.lineTo(b.x, b.y);
                  ctx.stroke();
                }
              }
            }

            if (Math.random() < 0.07) createPacket();

            for (let i = packets.length - 1; i >= 0; i--) {
              const p = packets[i];
              p.t += p.speed;

              if (p.t >= 1) {
                packets.splice(i, 1);
                continue;
              }

              const x = p.ax + (p.bx - p.ax) * p.t;
              const y = p.ay + (p.by - p.ay) * p.t;

              const glow = ctx.createRadialGradient(x, y, 0, x, y, 9);
              glow.addColorStop(0, `rgba(${CYAN}, 0.80)`);
              glow.addColorStop(0.4, `rgba(${VIOLET}, 0.30)`);
              glow.addColorStop(1, "rgba(0,0,0,0)");

              ctx.beginPath();
              ctx.fillStyle = glow;
              ctx.arc(x, y, 9, 0, Math.PI * 2);
              ctx.fill();
            }

            for (const n of nodes) {
              const pulse = 0.5 + Math.sin(n.pulse) * 0.35;

              ctx.beginPath();
              ctx.fillStyle = `rgba(${CYAN}, ${pulse * 0.60})`;
              ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
              ctx.fill();

              ctx.beginPath();
              ctx.strokeStyle = `rgba(${CYAN}, ${pulse * 0.14})`;
              ctx.lineWidth = 1;
              ctx.arc(n.x, n.y, n.r + 4 + Math.sin(n.pulse) * 2, 0, Math.PI * 2);
              ctx.stroke();
            }

            if (mouse.x > -1000) {
              const radius = 200;

              const field = ctx.createRadialGradient(
                mouse.x, mouse.y, 0,
                mouse.x, mouse.y, radius
              );
              field.addColorStop(0, `rgba(${CYAN}, 0.09)`);
              field.addColorStop(0.45, `rgba(${VIOLET}, 0.05)`);
              field.addColorStop(1, "rgba(0,0,0,0)");

              ctx.beginPath();
              ctx.fillStyle = field;
              ctx.arc(mouse.x, mouse.y, radius, 0, Math.PI * 2);
              ctx.fill();

              const time = Date.now() / 1000;

              for (let k = 0; k < 3; k++) {
                const ringR = ((time + k / 3) % 1) * radius;
                const alpha = 0.20 * (1 - ringR / radius);

                ctx.beginPath();
                ctx.strokeStyle = `rgba(${CYAN}, ${alpha})`;
                ctx.lineWidth = 1;
                ctx.arc(mouse.x, mouse.y, ringR, 0, Math.PI * 2);
                ctx.stroke();
              }

              for (const n of nodes) {
                const dx = mouse.x - n.x;
                const dy = mouse.y - n.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < radius && dist > 1) {
                  const force = (1 - dist / radius) * 0.008;
                  n.x += dx * force;
                  n.y += dy * force;
                }
              }
            }

            requestAnimationFrame(draw);
          }

          draw();
        </script>
        """,
        height=0,
    )


def apply_theme(rtl: bool = False, state: str | None = None, sidebar_brand: bool = True) -> None:
    """Inject fonts, CSS, animated background and the sidebar brand.

    state: optional "Normal" | "Fault" | "Attack" to tint the whole UI.
    """
    extra = _state_vars(state)
    if rtl:
        extra += ".stApp,[data-testid='stSidebar']{direction:rtl}[data-testid='stMetric']{text-align:right}"
    render(f"<style>{_CSS}{extra}</style>{_background()}")
    _cyber_grid_layer()
    if sidebar_brand:
        with st.sidebar:
            render(
                '<div class="brand"><div class="brand-logo"><b>⚡</b></div>'
                '<div class="brand-name">POWERGRID<small>CYBER AI</small></div></div>'
                '<div class="online"><i class="dot" style="--state:#00ff9d"></i>SYSTEM ONLINE</div>'
                '<div class="sim"><b>SIMULATION MODE</b><br>لا اتصال بشبكة حقيقية</div>'
            )


def set_state(state: str) -> None:
    """Re-tint the page after a prediction is known (Normal/Fault/Attack)."""
    css = _state_vars(state)
    if state == "Attack":
        css += ".pg-alert{animation:pgAlert 1.4s ease-in-out infinite}@keyframes pgAlert{0%,100%{opacity:.15}50%{opacity:1}}"
    elif state == "Fault":
        css += ".pg-alert{opacity:.18}"
    render(f"<style>{css}</style>")


# --------------------------------------------------------------------------
# HTML components
# --------------------------------------------------------------------------
def hero(title: str, subtitle: str = "", rtl: bool = False) -> None:
    cls = "hero rtl" if rtl else "hero"
    render(
        f'<div class="{cls}"><h1 class="hero-title" data-text="{_esc(title)}">{_esc(title)}</h1>'
        f'<p class="hero-sub">{_esc(subtitle)}</p><div class="hero-line"></div></div>'
    )


def section_header(title: str, sub: str = "", icon: str = "") -> None:
    sub_html = f"<p>{_esc(sub)}</p>" if sub else ""
    render(f'<div class="sec"><span class="sec-bar"></span><div><h3>{_esc(icon + " " if icon else "")}{_esc(title)}</h3>{sub_html}</div></div>')


def metric_card(label: str, value, icon: str = "", state: str | None = None, hint: str = "") -> str:
    """Return HTML for a KPI card. Integers get a CSS count-up animation."""
    color = STATE_COLORS.get(state or "", None)
    style = f' style="--state:{color};--state-rgb:{_rgb(color)}"' if color else ""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        val = f'<span class="cnt" style="--to:{value}" aria-label="{value}"></span>'
    else:
        val = _esc(value)
    return _pack(
        f'<div class="hud"{style}><div class="in"><div class="mc-top"><span class="mc-ico">{_esc(icon)}</span>'
        f'<span>{_esc(label)}</span></div><span class="mc-val num">{val}</span>'
        f'<div class="mc-hint">{_esc(hint)}</div><div class="mc-bar"><i></i></div></div></div>'
    )


def show_metric(container, label: str, value, icon: str = "", state: str | None = None, hint: str = "") -> None:
    container.markdown(metric_card(label, value, icon, state, hint), unsafe_allow_html=True)


def status_badge(name: str, small: bool = False) -> str:
    color = STATE_COLORS.get(name, CYAN)
    cls = "badge sm" if small else "badge"
    return f'<span class="{cls}" style="--state:{color};--state-rgb:{_rgb(color)}"><i class="dot"></i>{_esc(name)}</span>'


def result_card(predicted: str, risk_score: float, p_attack: float) -> None:
    color = STATE_COLORS.get(predicted, CYAN)
    render(
        f'<div class="hud" style="--state:{color};--state-rgb:{_rgb(color)}"><div class="in"><div class="result">'
        f'<div>{status_badge(predicted)}<div class="big num" style="margin-top:12px">{risk_score:.1%}</div>'
        f'<div class="sub">Risk Score &nbsp;|&nbsp; P(Attack) <span class="num">{p_attack:.1%}</span></div></div></div></div></div>'
    )


def flow_steps(active: int, labels: list[str]) -> None:
    items = "".join(f'<span class="{"on" if i + 1 == active else ""}">{i + 1}. {_esc(t)}</span>' for i, t in enumerate(labels))
    render(f'<div class="flow">{items}</div>')


def data_table(df: pd.DataFrame, max_rows: int = 200) -> None:
    """Dark neon HTML table; Available/Missing/Normal/Fault/Attack values become badges."""
    badge_map = {"Available": "#00ff9d", "Missing": "#ff2d55", **STATE_COLORS}
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    body = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for v in row.values:
            s = str(v)
            if s in badge_map:
                c = badge_map[s]
                cells.append(f'<td><span class="badge sm" style="--state:{c};--state-rgb:{_rgb(c)}"><i class="dot"></i>{_esc(s)}</span></td>')
            else:
                cells.append(f"<td>{_esc(s)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    render(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def confusion_heatmap(matrix: pd.DataFrame, labels=("Normal", "Fault", "Attack"),
                      row_title: str = "TRUE", col_title: str = "PREDICTED") -> None:
    """HTML heatmap. Attack->Fault and Fault->Attack cells pulse red (critical errors)."""
    labels = list(labels)
    vmax = max(1, int(matrix.values.max()))
    cols = f"grid-template-columns:90px repeat({len(labels)},1fr)"
    cells = [f'<div class="h">{_esc(row_title)} ↓ / {_esc(col_title)} →</div>']
    cells += [f'<div class="h">{_esc(c)}</div>' for c in labels]
    for r in labels:
        cells.append(f'<div class="h">{_esc(r)}</div>')
        for c in labels:
            v = int(matrix.loc[r, c])
            a = 0.10 + 0.75 * (v / vmax) ** 0.6
            rgb = _rgb(CYAN) if r == c else _rgb(STATE_COLORS["Attack"])
            if r != c and v == 0:
                rgb, a = "124,77,255", 0.06
            crit = " crit" if (r, c) in (("Attack", "Fault"), ("Fault", "Attack")) else ""
            cells.append(f'<div class="c{crit}" style="--rgb:{rgb};--a:{a:.2f}"><span class="num">{v}</span></div>')
    render(f'<div class="cm" style="{cols}">{"".join(cells)}</div>')


# --------------------------------------------------------------------------
# Plotly components
# --------------------------------------------------------------------------
def _layout(fig, height: int = 320, margin=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_FONT, color="#cfe3ff", size=13),
        margin=margin or dict(l=10, r=20, t=30, b=10),
        height=height,
        hoverlabel=dict(bgcolor="#0a1226", bordercolor=CYAN, font=dict(family=PLOT_FONT, color="#fff")),
    )
    return fig


def show_chart(fig, key: str | None = None) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)
    except TypeError:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)


def risk_gauge(risk: float, state: str | None = None, height: int = 300):
    """Semicircular gauge (0-1 input) with a green->yellow->red scale and a white needle."""
    value = max(0.0, min(1.0, float(risk))) * 100
    color = STATE_COLORS.get(state or "", CYAN)
    steps = []
    for i in range(20):
        t = i / 19
        c = _lerp("#00ff9d", "#ffe14d", t * 2) if t < .5 else _lerp("#ffe14d", "#ff2d55", (t - .5) * 2)
        steps.append(dict(range=[i * 5, (i + 1) * 5], color=f"rgba({_rgb(c)},.62)"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix="%", valueformat=".1f", font=dict(family="Orbitron", size=46, color=color)),
        title=dict(text="RISK SCORE", font=dict(family="Orbitron", size=13, color="#7f93b5")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#5b6b8c", tickfont=dict(color="#7f93b5", size=11)),
            bar=dict(color="rgba(0,0,0,0)", thickness=0), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=steps, threshold=dict(line=dict(color="#ffffff", width=6), thickness=0.95, value=value),
        ),
    ))
    return _layout(fig, height, dict(l=30, r=30, t=60, b=10))


def prob_chart(probabilities: dict, height: int = 240):
    names = [n for n in CLASS_ORDER if n in probabilities] + [n for n in probabilities if n not in CLASS_ORDER]
    vals = [float(probabilities[n]) for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=[STATE_COLORS.get(n, CYAN) for n in names], line=dict(color="rgba(255,255,255,.35)", width=1)),
        text=[f"{v:.1%}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", gridcolor="rgba(0,229,255,.08)", zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, height)


def prob_donut(probabilities: dict, height: int = 280):
    names = [n for n in CLASS_ORDER if n in probabilities] + [n for n in probabilities if n not in CLASS_ORDER]
    fig = go.Figure(go.Pie(
        labels=names, values=[float(probabilities[n]) for n in names], hole=.66, sort=False,
        marker=dict(colors=[STATE_COLORS.get(n, CYAN) for n in names], line=dict(color="#05070d", width=3)),
        textinfo="percent", hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    return _layout(fig, height)


def count_chart(series: pd.Series, height: int = 260):
    names = [str(i) for i in series.index]
    fig = go.Figure(go.Bar(
        x=[float(v) for v in series.values], y=names, orientation="h",
        marker=dict(color=[STATE_COLORS.get(n, CYAN) for n in names], line=dict(color="rgba(255,255,255,.3)", width=1)),
        text=[f"{int(v):,}" for v in series.values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(gridcolor="rgba(0,229,255,.08)", zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, height)


def feature_bars(df: pd.DataFrame, label_col: str = "feature", value_col: str = "importance"):
    d = df.sort_values(value_col, ascending=True)
    n = max(1, len(d) - 1)
    colors = [_lerp(VIOLET, CYAN, i / n) for i in range(len(d))]
    fig = go.Figure(go.Bar(
        x=d[value_col], y=d[label_col].astype(str), orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,.25)", width=1)),
        hovertemplate="<b>%{y}</b><br>importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(gridcolor="rgba(0,229,255,.08)", zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(family="JetBrains Mono", size=11))
    return _layout(fig, max(320, 26 * len(d) + 60), dict(l=10, r=20, t=20, b=10))


def accuracy_compare(
    rf: float,
    xgb: float,
    cat: float,
    height: int = 300,
):
    """Compare accuracy for Random Forest, XGBoost and CatBoost."""

    names = ["Random Forest", "XGBoost", "CatBoost"]
    vals = [float(rf), float(xgb), float(cat)]

    fig = go.Figure(go.Bar(
        x=names, y=vals, width=[.42, .42, .42],
        marker=dict(
            color=[CYAN, VIOLET, "#00ff9d"],
            line=dict(color="rgba(255,255,255,.4)", width=1),
        ),
        text=[f"{v:.2%}" for v in vals],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(family="Orbitron", size=16, color="#fff"),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.3%}<extra></extra>",
    ))

    lo = max(0.0, min(vals) - 0.08)

    fig.update_yaxes(
        range=[lo, min(1.04, max(vals) + 0.05)],
        tickformat=".0%",
        gridcolor="rgba(0,229,255,.08)",
        zeroline=False,
    )

    fig.update_xaxes(showgrid=False)

    return _layout(fig, height)
