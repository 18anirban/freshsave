# ====================================================================
# LANDING PAGE: logo, tagline, picture, and the two tabs in the centre.
# Background: put your own photo in assets/background.jpg (or .png) and it
# is used automatically. Without one, a generated fresh-green backdrop is used.
# ====================================================================
import base64
from pathlib import Path

import streamlit as st

from src.data_prep import ROOT

ASSETS = ROOT / "assets"
BRAND = "FreshSave"
TAGLINE = "Know what will go to waste, before it does."
SUBLINE = "Daily risk alerts and smarter orders for fresh-food stores."

LOGO_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#46c98b'/><stop offset='1' stop-color='#1f7a52'/></linearGradient></defs>
<circle cx='32' cy='32' r='30' fill='url(#g)' stroke='white' stroke-width='2'/>
<path d='M17 40 C16 25 28 14 47 14 C48 32 37 45 22 44 Z' fill='white'/>
<path d='M20 41 C28 33 35 26 42 19' stroke='#1f7a52' stroke-width='2.4' fill='none' stroke-linecap='round'/>
<path d='M24 49 l5 5 l11 -12' stroke='white' stroke-width='4' fill='none' stroke-linecap='round' stroke-linejoin='round'/>
</svg>"""

HERO_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 560 300'>
<defs>
<linearGradient id='wood' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#d9a566'/><stop offset='1' stop-color='#a56f3a'/></linearGradient>
<radialGradient id='apple' cx='0.35' cy='0.3' r='0.8'><stop offset='0' stop-color='#ff7a6b'/><stop offset='1' stop-color='#c5201f'/></radialGradient>
<radialGradient id='orange' cx='0.35' cy='0.3' r='0.8'><stop offset='0' stop-color='#ffc15a'/><stop offset='1' stop-color='#e8731a'/></radialGradient>
<radialGradient id='leaf' cx='0.4' cy='0.3' r='0.9'><stop offset='0' stop-color='#8fdc6a'/><stop offset='1' stop-color='#2f8f3a'/></radialGradient>
</defs>
<ellipse cx='280' cy='278' rx='235' ry='14' fill='rgba(0,0,0,0.25)'/>
<!-- leafy greens -->
<g><circle cx='95' cy='150' r='46' fill='url(#leaf)'/><circle cx='130' cy='135' r='42' fill='#58b84a'/><circle cx='108' cy='118' r='36' fill='#7dd25d'/></g>
<!-- milk carton -->
<g><polygon points='392,70 440,70 452,88 404,88' fill='#e9eef5'/><rect x='404' y='88' width='48' height='112' fill='#ffffff'/><polygon points='392,70 404,88 404,200 392,184' fill='#cfd8e3'/>
<rect x='412' y='110' width='32' height='46' rx='5' fill='#3a82e8'/><circle cx='428' cy='126' r='8' fill='#ffffff'/></g>
<!-- bread -->
<g><rect x='300' y='128' width='96' height='62' rx='30' fill='#d79b4f'/><rect x='300' y='128' width='96' height='40' rx='26' fill='#e8b46c'/>
<path d='M322 134 l10 28 M346 132 l10 30 M370 134 l10 28' stroke='#b9792f' stroke-width='5' stroke-linecap='round'/></g>
<!-- apples -->
<g><circle cx='200' cy='168' r='40' fill='url(#apple)'/><path d='M200 130 q6 -18 20 -20' stroke='#5b3a1a' stroke-width='5' fill='none' stroke-linecap='round'/><ellipse cx='222' cy='122' rx='13' ry='7' fill='#4cae4c' transform='rotate(-25 222 122)'/>
<circle cx='262' cy='176' r='36' fill='url(#apple)'/><path d='M262 141 q4 -14 16 -16' stroke='#5b3a1a' stroke-width='5' fill='none' stroke-linecap='round'/></g>
<!-- oranges -->
<g><circle cx='160' cy='196' r='30' fill='url(#orange)'/><circle cx='322' cy='204' r='28' fill='url(#orange)'/></g>
<!-- crate -->
<g><rect x='44' y='190' width='472' height='84' rx='8' fill='url(#wood)'/>
<rect x='44' y='190' width='472' height='14' rx='6' fill='#e7b878'/>
<path d='M44 222 H516 M44 250 H516' stroke='#8a5a2b' stroke-width='3' opacity='0.6'/>
<rect x='60' y='190' width='14' height='84' fill='#8a5a2b' opacity='0.5'/><rect x='486' y='190' width='14' height='84' fill='#8a5a2b' opacity='0.5'/></g>
</svg>"""

BACKDROP_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMid slice'>
<defs><linearGradient id='b' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#0d3b2a'/><stop offset='0.6' stop-color='#14684a'/><stop offset='1' stop-color='#0a2c20'/></linearGradient>
<filter id='blur'><feGaussianBlur stdDeviation='28'/></filter></defs>
<rect width='1600' height='900' fill='url(#b)'/>
<g filter='url(#blur)' opacity='0.85'>
<circle cx='180' cy='160' r='120' fill='#e9573f'/><circle cx='420' cy='700' r='150' fill='#f2a33a'/><circle cx='1380' cy='180' r='140' fill='#7cd35a'/>
<circle cx='1250' cy='740' r='170' fill='#e9573f'/><circle cx='820' cy='90' r='90' fill='#f6d44a'/><circle cx='1520' cy='470' r='110' fill='#2fb37a'/><circle cx='60' cy='520' r='100' fill='#7cd35a'/>
</g></svg>"""


def _uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def background_uri():
    """Your own photo (assets/background.jpg|jpeg|png) if present, else the generated backdrop."""
    for name, mime in [("background.jpg", "jpeg"), ("background.jpeg", "jpeg"), ("background.png", "png")]:
        f = ASSETS / name
        if f.exists():
            return f"data:image/{mime};base64," + base64.b64encode(f.read_bytes()).decode()
    return _uri(BACKDROP_SVG)


def page_css():
    return f"""<style>
.stApp {{ background: linear-gradient(rgba(5,32,22,0.62), rgba(5,32,22,0.82)), url("{background_uri()}") center / cover fixed; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 980px; padding-top: 2.2rem; }}
.hero {{ text-align: center; color: #ffffff; }}
.hero img.logo {{ width: 84px; height: 84px; filter: drop-shadow(0 6px 14px rgba(0,0,0,0.4)); }}
.hero .brand {{ font-size: 3.4rem; font-weight: 800; letter-spacing: 0.5px; margin: 0.2rem 0 0; line-height: 1.1; }}
.hero .tag {{ font-size: 1.75rem; font-weight: 600; margin: 0.6rem 0 0.2rem; color: #d9ffe9; }}
.hero .sub {{ font-size: 1.05rem; opacity: 0.88; margin-bottom: 1.2rem; }}
.hero img.art {{ width: min(520px, 90%); margin: 0.4rem 0 1.4rem; filter: drop-shadow(0 14px 22px rgba(0,0,0,0.45)); }}
.hero .foot {{ font-size: 0.8rem; opacity: 0.7; margin-top: 1.6rem; }}
.st-key-go_details button, .st-key-go_demo button {{ width: 100%; height: 3.4rem; border-radius: 999px; font-weight: 700; border: 2px solid #ffffff;
    box-shadow: 0 8px 20px rgba(0,0,0,0.35); }}
.st-key-go_details button p, .st-key-go_demo button p {{ font-size: 1.15rem; font-weight: 700; }}
.st-key-go_details button {{ background: #ffffff; color: #0d3b2a; }}
.st-key-go_details button p {{ color: #0d3b2a; }}
.st-key-go_demo button {{ background: #2e9e6b; color: #ffffff; }}
.st-key-go_demo button p {{ color: #ffffff; }}
.st-key-go_details button:hover, .st-key-go_demo button:hover {{ transform: translateY(-2px); border-color: #d9ffe9; }}
</style>"""


def hero_html():
    return f"""<div class="hero">
<img class="logo" src="{_uri(LOGO_SVG)}">
<div class="brand">{BRAND}</div>
<div class="tag">{TAGLINE}</div>
<div class="sub">{SUBLINE}</div>
<img class="art" src="{_uri(HERO_SVG)}">
</div>"""


def landing():
    st.markdown(page_css(), unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)
    _, c1, c2, _ = st.columns([1, 2, 2, 1])            # the two tabs, centred and equal width
    if c1.button("Model Train Details", key="go_details"):
        st.session_state["page"] = "details"
        st.rerun()
    if c2.button("Demo", key="go_demo"):
        st.session_state["page"] = "demo"
        st.rerun()
    st.markdown('<div class="hero"><div class="foot">KNN and Decision Tree models - simulated retail data</div></div>', unsafe_allow_html=True)
