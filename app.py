from __future__ import annotations
import os
import unicodedata
from datetime import datetime
from pathlib import Path
import streamlit as st
from ingestion import IngestionPipeline, RAW_DOCS_DIR
from retriever import RetrievalPipeline
import base64
import requests

N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL", "")

NOT_FOUND_MESSAGE = "Answer not found in uploaded sources"
SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt"}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --darkest:   #0A1931;
    --dark:      #1A3D63;
    --mid:       #4A7FA7;
    --light:     #B3CFE5;
    --lightest:  #F6FAFD;
    --white:     #FFFFFF;

    --bg:        #0A1931;
    --surface:   #0f2240;
    --card:      #1A3D63;
    --border:    #2a5580;
    --primary:   #4A7FA7;
    --accent:    #B3CFE5;
    --text:      #F6FAFD;
    --muted:     #7aaec8;
    --success:   #4fc3a1;
    --danger:    #e05c7a;
    --gold:      #f0c96e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Animated background ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    overflow: hidden;
}

/* Deep wave layers */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 120% 80% at -10% 20%, rgba(26,61,99,0.9) 0%, transparent 55%),
        radial-gradient(ellipse 80% 100% at 110% 80%, rgba(74,127,167,0.4) 0%, transparent 50%),
        radial-gradient(ellipse 60% 60% at 50% 110%, rgba(10,25,49,0.95) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 80% 10%, rgba(179,207,229,0.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: waveShift 18s ease-in-out infinite alternate;
}

@keyframes waveShift {
    0%   { opacity: 0.8; transform: scale(1) translateY(0px); }
    33%  { opacity: 1;   transform: scale(1.03) translateY(-20px); }
    66%  { opacity: 0.9; transform: scale(0.98) translateY(10px); }
    100% { opacity: 1;   transform: scale(1.05) translateY(-15px); }
}

/* Animated orbs */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle 180px at 15% 30%, rgba(74,127,167,0.25) 0%, transparent 70%),
        radial-gradient(circle 120px at 80% 15%, rgba(179,207,229,0.15) 0%, transparent 70%),
        radial-gradient(circle 200px at 90% 70%, rgba(26,61,99,0.6) 0%, transparent 70%),
        radial-gradient(circle 90px  at 40% 80%, rgba(74,127,167,0.2) 0%, transparent 70%),
        radial-gradient(circle 150px at 60% 40%, rgba(10,25,49,0.4) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: orbFloat 25s ease-in-out infinite;
}

@keyframes orbFloat {
    0%,100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
    20%      { transform: translateY(-40px) translateX(25px) rotate(5deg); }
    40%      { transform: translateY(-15px) translateX(-30px) rotate(-3deg); }
    60%      { transform: translateY(-55px) translateX(15px) rotate(8deg); }
    80%      { transform: translateY(-20px) translateX(-15px) rotate(-5deg); }
}

/* ── Floating particles ── */
.mw-particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}

.particle {
    position: absolute;
    border-radius: 50%;
    animation: particleDrift linear infinite;
    opacity: 0;
}

.particle:nth-child(1)  { width:3px;  height:3px;  background:#B3CFE5; left:8%;   animation-duration:12s; animation-delay:0s;   }
.particle:nth-child(2)  { width:2px;  height:2px;  background:#4A7FA7; left:18%;  animation-duration:16s; animation-delay:3s;   }
.particle:nth-child(3)  { width:4px;  height:4px;  background:#F6FAFD; left:28%;  animation-duration:10s; animation-delay:1s;   }
.particle:nth-child(4)  { width:2px;  height:2px;  background:#B3CFE5; left:40%;  animation-duration:18s; animation-delay:5s;   }
.particle:nth-child(5)  { width:3px;  height:3px;  background:#4A7FA7; left:52%;  animation-duration:14s; animation-delay:2s;   }
.particle:nth-child(6)  { width:2px;  height:2px;  background:#F6FAFD; left:63%;  animation-duration:11s; animation-delay:7s;   }
.particle:nth-child(7)  { width:4px;  height:4px;  background:#B3CFE5; left:74%;  animation-duration:19s; animation-delay:0.5s; }
.particle:nth-child(8)  { width:2px;  height:2px;  background:#1A3D63; left:82%;  animation-duration:13s; animation-delay:4s;   }
.particle:nth-child(9)  { width:3px;  height:3px;  background:#4A7FA7; left:91%;  animation-duration:15s; animation-delay:6s;   }
.particle:nth-child(10) { width:2px;  height:2px;  background:#F6FAFD; left:35%;  animation-duration:17s; animation-delay:8s;   }
.particle:nth-child(11) { width:3px;  height:3px;  background:#B3CFE5; left:55%;  animation-duration:9s;  animation-delay:1.5s; }
.particle:nth-child(12) { width:4px;  height:4px;  background:#4A7FA7; left:70%;  animation-duration:20s; animation-delay:9s;   }

@keyframes particleDrift {
    0%   { bottom: -20px; opacity: 0; transform: translateX(0px); }
    10%  { opacity: 0.7; }
    50%  { transform: translateX(30px); }
    90%  { opacity: 0.4; }
    100% { bottom: 110vh; opacity: 0; transform: translateX(-20px); }
}

/* ── Flowing data streams ── */
.mw-streams {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}

.stream {
    position: absolute;
    top: -200px;
    width: 1px;
    background: linear-gradient(180deg, transparent, rgba(179,207,229,0.5), rgba(74,127,167,0.3), transparent);
    animation: streamFall linear infinite;
    opacity: 0;
}

.stream:nth-child(1)  { left:5%;   height:160px; animation-duration:6s;  animation-delay:0s;   }
.stream:nth-child(2)  { left:15%;  height:100px; animation-duration:9s;  animation-delay:2s;   opacity:0; }
.stream:nth-child(3)  { left:25%;  height:200px; animation-duration:7s;  animation-delay:1s;   }
.stream:nth-child(4)  { left:38%;  height:130px; animation-duration:11s; animation-delay:4s;   }
.stream:nth-child(5)  { left:50%;  height:170px; animation-duration:8s;  animation-delay:0.5s; }
.stream:nth-child(6)  { left:62%;  height:90px;  animation-duration:10s; animation-delay:3s;   }
.stream:nth-child(7)  { left:75%;  height:150px; animation-duration:6.5s;animation-delay:1.5s; }
.stream:nth-child(8)  { left:85%;  height:120px; animation-duration:12s; animation-delay:5s;   }
.stream:nth-child(9)  { left:93%;  height:180px; animation-duration:7.5s;animation-delay:2.5s; }

@keyframes streamFall {
    0%   { top: -200px; opacity: 0; }
    8%   { opacity: 0.6; }
    92%  { opacity: 0.3; }
    100% { top: 110vh;  opacity: 0; }
}

/* ── Slow-rotating rings ── */
.mw-rings {
    position: fixed;
    pointer-events: none;
    z-index: 0;
}

.ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(74,127,167,0.12);
    animation: ringRotate linear infinite;
}

.ring-1 {
    width: 500px; height: 500px;
    top: -150px; right: -150px;
    border-color: rgba(179,207,229,0.08);
    animation-duration: 40s;
}

.ring-2 {
    width: 350px; height: 350px;
    bottom: -100px; left: -100px;
    border-color: rgba(74,127,167,0.1);
    animation-duration: 30s;
    animation-direction: reverse;
}

.ring-3 {
    width: 220px; height: 220px;
    top: 40%; left: 70%;
    border-color: rgba(179,207,229,0.06);
    animation-duration: 20s;
}

@keyframes ringRotate {
    0%   { transform: rotate(0deg) scale(1); }
    50%  { transform: rotate(180deg) scale(1.05); }
    100% { transform: rotate(360deg) scale(1); }
}

/* ── Grid overlay ── */
.mw-grid {
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(74,127,167,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74,127,167,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── All content above BG ── */
[data-testid="stSidebar"],
[data-testid="stMainBlockContainer"],
.block-container { position: relative; z-index: 10; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 0 2.5rem 4rem 2.5rem !important; max-width: 1000px !important; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Title */
.stApp header + div h1,
.stApp .stMarkdown h1,
h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #F6FAFD 0%, #B3CFE5 50%, #4A7FA7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stApp .stCaption p {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* Section headers h2 */
.stApp h2 {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid rgba(74,127,167,0.3) !important;
    padding-bottom: 0.6rem;
    margin-top: 2.5rem !important;
}

.stApp h3 {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--light) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,25,49,0.97) !important;
    border-right: 1px solid rgba(74,127,167,0.25) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-size: 0.65rem !important;
    border-bottom: none !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.25s ease !important;
    background: linear-gradient(135deg, #1A3D63, #4A7FA7) !important;
    color: #F6FAFD !important;
    border: 1px solid rgba(179,207,229,0.25) !important;
    box-shadow: 0 4px 16px rgba(10,25,49,0.4), 0 0 0 1px rgba(74,127,167,0.1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4A7FA7, #B3CFE5) !important;
    color: #0A1931 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(74,127,167,0.35), 0 0 20px rgba(179,207,229,0.15) !important;
    border-color: rgba(179,207,229,0.5) !important;
}

.stButton > button:disabled {
    background: rgba(26,61,99,0.3) !important;
    color: rgba(179,207,229,0.3) !important;
    border-color: rgba(74,127,167,0.1) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1.5px solid rgba(179,207,229,0.3) !important;
    box-shadow: none !important;
    font-size: 0.72rem !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: rgba(74,127,167,0.15) !important;
    border-color: var(--accent) !important;
    color: #F6FAFD !important;
    transform: none !important;
}

/* ── Text Input ── */
.stTextInput > div > div > input {
    background: rgba(26,61,99,0.6) !important;
    border: 1px solid rgba(74,127,167,0.35) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.1rem !important;
    transition: all 0.25s ease !important;
    backdrop-filter: blur(8px) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--mid) !important;
    box-shadow: 0 0 0 3px rgba(74,127,167,0.15), 0 0 20px rgba(74,127,167,0.08) !important;
    background: rgba(26,61,99,0.8) !important;
}

.stTextInput > div > div > input::placeholder { color: var(--muted) !important; opacity: 0.7; }
.stTextInput > label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(26,61,99,0.4) !important;
    border: 1.5px dashed rgba(74,127,167,0.35) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(179,207,229,0.5) !important;
    background: rgba(26,61,99,0.55) !important;
}

[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }
.stSuccess { background: rgba(79,195,161,0.1) !important; border-left: 3px solid #4fc3a1 !important; color: #4fc3a1 !important; }
.stWarning { background: rgba(240,201,110,0.1) !important; border-left: 3px solid #f0c96e !important; color: #f0c96e !important; }
.stError   { background: rgba(224,92,122,0.1) !important; border-left: 3px solid #e05c7a !important; color: #e05c7a !important; }
.stInfo    { background: rgba(74,127,167,0.12) !important; border-left: 3px solid #4A7FA7 !important; color: #B3CFE5 !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(26,61,99,0.5) !important;
    border: 1px solid rgba(74,127,167,0.2) !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
    font-size: 0.82rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.2s ease;
}

.streamlit-expanderHeader:hover {
    background: rgba(74,127,167,0.2) !important;
    border-color: rgba(179,207,229,0.4) !important;
}

.streamlit-expanderContent {
    background: rgba(10,25,49,0.7) !important;
    border: 1px solid rgba(74,127,167,0.15) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    line-height: 1.75;
}

.streamlit-expanderContent p,
.streamlit-expanderContent div { color: var(--muted) !important; }

/* ── Answer box ── */
.answer-card {
    background: linear-gradient(135deg, rgba(26,61,99,0.8) 0%, rgba(10,25,49,0.9) 100%);
    border: 1px solid rgba(74,127,167,0.4);
    border-left: 4px solid #4A7FA7;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin: 1.25rem 0;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(10,25,49,0.5), 0 0 0 1px rgba(74,127,167,0.1);
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(179,207,229,0.04) 0%, transparent 60%);
    pointer-events: none;
}

.answer-label {
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--mid);
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(74,127,167,0.2);
}

.grounded-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.57rem;
    letter-spacing: 0.12em;
    font-weight: 600;
    background: rgba(79,195,161,0.12);
    border: 1px solid rgba(79,195,161,0.3);
    color: #4fc3a1;
    text-transform: uppercase;
}

.pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4fc3a1;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(79,195,161,0.5); }
    50%      { box-shadow: 0 0 0 6px rgba(79,195,161,0); }
}

.answer-text {
    font-size: 0.93rem !important;
    line-height: 1.85 !important;
    color: #E8F4FF !important;
}

/* ── Citations ── */
.citations-header {
    font-size: 0.57rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-transform: uppercase;
    margin: 2rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.citations-header::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(74,127,167,0.3), transparent);
}

.cite-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin: 0.2rem;
    font-size: 0.65rem;
    background: rgba(74,127,167,0.15);
    border: 1px solid rgba(179,207,229,0.2);
    color: var(--accent);
}
.cite-num { color: var(--muted); }

/* ── Sidebar custom components ── */
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0.7rem;
    background: rgba(26,61,99,0.5);
    border: 1px solid rgba(74,127,167,0.2);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    backdrop-filter: blur(8px);
}
.sb-stat-lbl { font-size: 0.58rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; }
.sb-stat-val { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #B3CFE5; }

.file-chip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0.65rem;
    background: rgba(74,127,167,0.12);
    border: 1px solid rgba(179,207,229,0.15);
    border-radius: 6px;
    margin-bottom: 0.3rem;
}
.file-name { font-size: 0.68rem; color: var(--accent); }
.file-cnt  { font-size: 0.65rem; color: var(--mid); font-weight: 600; }

.stack-badge {
    display: inline-block;
    padding: 0.22rem 0.6rem;
    margin: 2px;
    background: rgba(26,61,99,0.7);
    border: 1px solid rgba(74,127,167,0.25);
    border-radius: 4px;
    font-size: 0.6rem;
    color: var(--accent);
    letter-spacing: 0.07em;
}

.sb-sep { height: 1px; background: rgba(74,127,167,0.2); margin: 0.75rem 0; }
.sb-sec {
    font-size: 0.55rem; letter-spacing: 0.28em; color: var(--muted);
    text-transform: uppercase; margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.35rem; border-bottom: 1px solid rgba(74,127,167,0.2);
}
.sb-session-active {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0.7rem;
    background: rgba(79,195,161,0.08);
    border: 1px solid rgba(79,195,161,0.2);
    border-radius: 8px; margin-bottom: 0.6rem;
}
.sb-session-id { font-size: 0.68rem; color: #4fc3a1; font-family: 'DM Sans', sans-serif; }
.sb-session-none {
    padding: 0.5rem 0.7rem;
    background: rgba(26,61,99,0.4);
    border: 1px solid rgba(74,127,167,0.15);
    border-radius: 8px;
    font-size: 0.68rem; color: var(--muted); margin-bottom: 0.6rem;
}

/* Spinner */
.stSpinner > div { border-top-color: #4A7FA7 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,127,167,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--mid); }

hr { border-color: rgba(74,127,167,0.2) !important; margin: 1.5rem 0 !important; }

.main .block-container {
    padding: 2.5rem 2.5rem 4rem !important;
    max-width: 980px;
}

code {
    background: rgba(26,61,99,0.7) !important;
    color: var(--accent) !important;
    border-radius: 4px !important;
    padding: 0.15em 0.4em !important;
    font-size: 0.82em !important;
}
</style>

<!-- Animated background layers -->
<div class="mw-particles">
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
</div>

<div class="mw-streams">
  <div class="stream"></div><div class="stream"></div><div class="stream"></div>
  <div class="stream"></div><div class="stream"></div><div class="stream"></div>
  <div class="stream"></div><div class="stream"></div><div class="stream"></div>
</div>

<div class="mw-rings">
  <div class="ring ring-1"></div>
  <div class="ring ring-2"></div>
  <div class="ring ring-3"></div>
</div>

<div class="mw-grid"></div>
"""


# ── Unicode safety ─────────────────────────────────────────────────────────────
def _safe_latin1(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── Notes helpers ──────────────────────────────────────────────────────────────
def build_pdf_notes(notes: list[dict]) -> bytes:
    from fpdf import FPDF

    def hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    BG      = hex2rgb("0A1931")
    CARD    = hex2rgb("1A3D63")
    BORDER  = hex2rgb("2a5580")
    PRIMARY = hex2rgb("4A7FA7")
    LIGHT   = hex2rgb("B3CFE5")
    MUTED   = hex2rgb("7aaec8")

    class NotesPDF(FPDF):
        def header(self):
            self.set_fill_color(*BG)
            self.rect(0, 0, 210, 297, "F")

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*MUTED)
            self.cell(0, 6, _safe_latin1(
                f"MindWeave Research Notes  -  Page {self.page_no()}"
            ), align="C")

    pdf = NotesPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    W = 170

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*LIGHT)
    pdf.cell(W, 10, "MindWeave", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(W, 8, "Research Notes", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    n = len(notes)
    pdf.cell(W, 5, _safe_latin1(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  -  {n} note{'s' if n != 1 else ''}"
    ), ln=True)
    pdf.ln(3)
    pdf.set_draw_color(*PRIMARY)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    for i, entry in enumerate(notes, start=1):
        pdf.set_fill_color(*CARD)
        pdf.set_draw_color(*BORDER)
        pdf.set_line_width(0.3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*LIGHT)
        pdf.cell(W, 7, _safe_latin1(f"  Note {i}"), border=1, fill=True, ln=False)
        pdf.ln(7)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*MUTED)
        pdf.cell(W, 4, _safe_latin1(f"  {entry['timestamp']}"), ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(W, 5, "QUESTION", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(W, 5, _safe_latin1(entry["question"]))
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(W, 5, "ANSWER", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(W, 5, _safe_latin1(entry["answer"]))
        pdf.ln(2)
        if entry.get("sources"):
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*PRIMARY)
            pdf.cell(W, 5, "SOURCES", ln=True)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*MUTED)
            for src in entry["sources"]:
                line = (f"  -  {src['file']}  |  Page {src['page']}  |  "
                        f"Section: {src['section']}  |  Score: {src['score']:.4f}")
                pdf.multi_cell(W, 4.5, _safe_latin1(line))
            pdf.ln(1)
        pdf.set_draw_color(*BORDER)
        pdf.set_line_width(0.3)
        pdf.line(20, pdf.get_y() + 2, 190, pdf.get_y() + 2)
        pdf.ln(6)

    return bytes(pdf.output())


def add_to_notes(question: str, answer: str, result) -> None:
    sources = []
    if result.sources:
        for chunk in result.sources:
            sources.append({
                "file": chunk.source_file,
                "page": chunk.page_number if chunk.page_number else "N/A",
                "section": chunk.section or "General",
                "score": chunk.similarity_score,
            })
    st.session_state.notes.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources,
    })


# ── App state ──────────────────────────────────────────────────────────────────
def ensure_state():
    defaults = {
        "session_id": None, "indexed_files": {}, "notes": [],
        "last_answer": None, "query_count": 0, "total_chunks": 0, "groq_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def get_ingestion_pipeline(): return IngestionPipeline()


@st.cache_resource(show_spinner=False)
def get_retrieval_pipeline(): return RetrievalPipeline()


def persist_uploaded_file(uploaded_file, session_id):
    session_dir = RAW_DOCS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    fp = session_dir / Path(uploaded_file.name).name
    with open(fp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return fp


def is_supported(filename):
    return Path(filename).suffix.lower().replace(".", "") in SUPPORTED_EXTENSIONS


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0 0.5rem 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                background:linear-gradient(135deg,#F6FAFD,#B3CFE5,#4A7FA7);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:-0.03em;">MindWeave</div>
            <div style="font-size:0.52rem;letter-spacing:0.28em;color:#7aaec8;
                text-transform:uppercase;margin-top:3px;">Intelligence Platform</div>
        </div>
        <div class="sb-sep"></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-sec">Session</div>', unsafe_allow_html=True)
        if st.session_state.session_id:
            sid = st.session_state.session_id[:14]
            st.markdown(f"""
            <div class="sb-session-active">
                <span class="pulse"></span>
                <span class="sb-session-id">{sid}…</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-session-none">No active session</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("＋ New", key="new_sess"):
                ip = get_ingestion_pipeline()
                st.session_state.update({
                    "session_id": ip.generate_session_id(),
                    "indexed_files": {}, "notes": [], "last_answer": None,
                    "query_count": 0, "total_chunks": 0,
                })
                st.rerun()
        with c2:
            if st.button("⟳ Groq", key="chk_groq"):
                s = get_retrieval_pipeline().get_groq_status()
                st.session_state.groq_status = s

        if st.session_state.groq_status:
            s = st.session_state.groq_status
            color, icon, txt = ("#4fc3a1", "✓", s["model"]) if s["online"] else ("#e05c7a", "✗", "Groq unreachable")
            st.markdown(f'<div style="font-size:0.65rem;color:{color};margin-top:0.3rem;">{icon} {txt}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-sec">Analytics</div>', unsafe_allow_html=True)
        for lbl, val in [("Queries", st.session_state.query_count),
                          ("Chunks",  st.session_state.total_chunks),
                          ("Docs",    len(st.session_state.indexed_files))]:
            st.markdown(f"""
            <div class="sb-stat">
                <span class="sb-stat-lbl">{lbl}</span>
                <span class="sb-stat-val">{val}</span>
            </div>""", unsafe_allow_html=True)

        if st.session_state.indexed_files:
            st.markdown('<div class="sb-sec">Documents</div>', unsafe_allow_html=True)
            for name, cnt in st.session_state.indexed_files.items():
                short = (name[:18] + "…") if len(name) > 18 else name
                st.markdown(f"""
                <div class="file-chip">
                    <span class="file-name">📄 {short}</span>
                    <span class="file-cnt">{cnt}c</span>
                </div>""", unsafe_allow_html=True)

        note_count = len(st.session_state.notes)
        st.markdown('<div class="sb-sec">Research Notes</div>', unsafe_allow_html=True)
        if note_count == 0:
            st.caption("No notes yet.")
        else:
            st.caption(f"{note_count} note{'s' if note_count != 1 else ''} captured")
            if st.button("Clear All Notes", key="clear_notes"):
                st.session_state.notes = []
                st.rerun()
            for i, entry in enumerate(reversed(st.session_state.notes), start=1):
                with st.expander(f"#{note_count - i + 1} · {entry['timestamp']}"):
                    st.markdown(f"**Q:** {entry['question']}")
                    st.markdown(
                        f"**A:** {entry['answer'][:280]}{'...' if len(entry['answer']) > 280 else ''}"
                    )

        st.markdown('<div class="sb-sec">Stack</div>', unsafe_allow_html=True)
        for t in ["llama-3.3-70b", "FAISS", "BM25", "RRF Fusion", "Groq API"]:
            st.markdown(f'<span class="stack-badge">{t}</span>', unsafe_allow_html=True)


# ── Sources ───────────────────────────────────────────────────────────────────
def render_sources(result):
    if not result.sources:
        st.info("No retrieved chunks to display.")
        return

    st.markdown('<div class="citations-header">Citations</div>', unsafe_allow_html=True)
    pills = "".join(
        f'<span class="cite-pill"><span class="cite-num">#{i}</span> '
        f'pg {c.page_number or "—"} · {c.similarity_score:.3f}</span>'
        for i, c in enumerate(result.sources, 1)
    )
    st.markdown(f'<div style="margin-bottom:1.25rem;">{pills}</div>', unsafe_allow_html=True)

    st.markdown('<div class="citations-header">Source Chunks</div>', unsafe_allow_html=True)
    for i, c in enumerate(result.sources, 1):
        pg = c.page_number or "N/A"
        label = f"#{i}  ·  {c.source_file}  ·  pg {pg}  ·  score {c.similarity_score:.4f}  ·  {c.section or 'General'}"
        with st.expander(label):
            st.write(c.text)


# ── Voice helper ──────────────────────────────────────────────────────────────
def speak_answer(answer_text: str) -> None:
    safe = answer_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""
<div id="voice-bar" style="
    display:flex; align-items:center; gap:12px;
    background:rgba(26,61,99,0.7);
    border:1px solid rgba(74,127,167,0.35);
    border-radius:8px; padding:10px 16px; margin:10px 0 6px;
    backdrop-filter:blur(8px);
    font-family:'DM Sans',sans-serif;">
  <span id="voice-icon" style="font-size:1.2rem;">🔊</span>
  <span id="voice-status" style="color:#B3CFE5;font-size:0.78rem;font-weight:600;
    letter-spacing:0.07em;text-transform:uppercase;flex:1;">Speaking answer...</span>
  <button id="stop-btn" onclick="stopSpeech()" style="
    background:transparent;color:#4A7FA7;
    border:1.5px solid rgba(74,127,167,0.4);
    border-radius:6px;padding:4px 12px;
    font-family:'DM Sans',sans-serif;font-size:0.75rem;font-weight:600;
    letter-spacing:0.07em;text-transform:uppercase;cursor:pointer;
    transition:all .2s ease;">Stop</button>
</div>
<script>
(function() {{
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(`{safe}`);
  utter.rate=1.0; utter.pitch=1.0; utter.volume=1.0;
  function pickVoice() {{
    const voices = window.speechSynthesis.getVoices();
    const pref = ["Google US English","Google UK English Female","Microsoft Aria","Samantha"];
    for(const n of pref){{ const v=voices.find(v=>v.name===n); if(v) return v; }}
    return voices.find(v=>v.lang.startsWith("en"))||voices[0]||null;
  }}
  function start() {{ const v=pickVoice(); if(v) utter.voice=v; window.speechSynthesis.speak(utter); }}
  if(window.speechSynthesis.getVoices().length===0) window.speechSynthesis.onvoiceschanged=start;
  else start();
  utter.onend=function(){{
    const s=document.getElementById("voice-status");
    const ic=document.getElementById("voice-icon");
    const b=document.getElementById("stop-btn");
    if(s) s.textContent="Done speaking"; if(ic) ic.textContent="✅"; if(b) b.style.display="none";
  }};
  window.stopSpeech=function(){{
    window.speechSynthesis.cancel();
    const s=document.getElementById("voice-status");
    const ic=document.getElementById("voice-icon");
    const b=document.getElementById("stop-btn");
    if(s) s.textContent="Stopped"; if(ic) ic.textContent="⏹"; if(b) b.style.display="none";
  }};
}})();
</script>
"""
    import streamlit.components.v1 as components
    components.html(html, height=70)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="MindWeave", page_icon="🧠", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("MindWeave")
    st.caption("Powered by Groq llama-3.3-70b · Hybrid FAISS + BM25 Retrieval")
    ensure_state()
    render_sidebar()

    # ── Upload & Index ────────────────────────────────────────────────────────
    st.header("1) Upload and Index Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index Uploaded Documents", disabled=not uploaded_files):
        indexed_now = {}
        with st.spinner("Embedding & indexing documents…"):
            ip = get_ingestion_pipeline()
            if not st.session_state.session_id:
                st.session_state.session_id = ip.generate_session_id()
            for uf in uploaded_files:
                if not is_supported(uf.name):
                    st.error(f"Unsupported: {uf.name}")
                    continue
                try:
                    fp  = persist_uploaded_file(uf, st.session_state.session_id)
                    cnt = ip.ingest(fp, st.session_state.session_id)
                    indexed_now[uf.name] = cnt
                    st.session_state.total_chunks += cnt
                except Exception as e:
                    st.error(f"{uf.name}: {e}")
        st.session_state.indexed_files.update(indexed_now)
        if indexed_now:
            total = sum(indexed_now.values())
            st.success(f"✓ {len(indexed_now)} file(s) · {total} chunks indexed")
            for name, cnt in indexed_now.items():
                st.write(f"- {name}: {cnt} chunks")
        else:
            st.warning("No documents were indexed.")

    # ── Ask Questions ─────────────────────────────────────────────────────────
    st.header("2) Ask Questions")
    question = st.text_input(
        "Ask a question grounded only in uploaded documents",
        placeholder="e.g. What are the key insights in this document?",
    )
    ask_disabled = (
        not question.strip()
        or not st.session_state.session_id
        or not bool(st.session_state.indexed_files)
    )

    col_ask, col_hint = st.columns([1, 4])
    with col_ask:
        ask_clicked = st.button("Get Answer", disabled=ask_disabled, key="ask_btn")
    with col_hint:
        if ask_disabled and not st.session_state.indexed_files:
            st.markdown(
                '<span style="font-size:0.72rem;color:#7aaec8;line-height:2.8rem;display:block;">'
                'Index documents first to enable querying</span>',
                unsafe_allow_html=True,
            )

    if ask_clicked:
        try:
            rp = get_retrieval_pipeline()
            if not rp.check_session_ready(st.session_state.session_id):
                st.error("No indexed knowledge base found for this session.")
                return
            with st.spinner("Retrieving context · Synthesising with Groq…"):
                result = rp.query(
                    question=question.strip(), session_id=st.session_state.session_id
                )
            st.session_state.query_count += 1
            st.session_state.last_answer = result
        except Exception as e:
            st.error(f"Query failed: {e}")

    # ── Answer display ────────────────────────────────────────────────────────
    if st.session_state.last_answer:
        result = st.session_state.last_answer

        if not result.answer_found:
            st.warning(NOT_FOUND_MESSAGE)
        else:
            grounded = ""
            if result.is_grounded:
                grounded = '<span class="grounded-badge"><span class="pulse"></span> Grounded</span>'

            answer_html = result.answer.replace("\n", "<br>")
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-label">Answer {grounded}</div>
                <div class="answer-text">{answer_html}</div>
            </div>
            """, unsafe_allow_html=True)

            speak_answer(result.answer)

            if ask_clicked:
                add_to_notes(question.strip(), result.answer, result)
                n = len(st.session_state.notes)
                st.success(
                    f"Saved to Research Notes — {n} note{'s' if n != 1 else ''} total. "
                    "Scroll down to Section 3 to download."
                )

        render_sources(result)

    # ── Research Notes ────────────────────────────────────────────────────────
    if st.session_state.notes:
        st.markdown("---")
        st.header("3) Research Notes")

        note_count = len(st.session_state.notes)
        fname_base = f"mindweave_notes_{datetime.now().strftime('%Y%m%d_%H%M')}"

        dl_col, _ = st.columns([2, 6])
        with dl_col:
            st.download_button(
                label="Download Notes as PDF",
                data=build_pdf_notes(st.session_state.notes),
                file_name=f"{fname_base}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.caption(f"{note_count} note{'s' if note_count != 1 else ''} captured this session")

        for i, entry in enumerate(st.session_state.notes, start=1):
            label = (
                f"Note {i} · {entry['timestamp']} · "
                f"{entry['question'][:65]}{'...' if len(entry['question']) > 65 else ''}"
            )
            with st.expander(label):
                st.markdown("**Question**")
                st.write(entry["question"])
                st.markdown("**Answer**")
                st.write(entry["answer"])
                if entry["sources"]:
                    st.markdown("**Sources**")
                    for src in entry["sources"]:
                        st.markdown(
                            f"- `{src['file']}` · Page {src['page']} · "
                            f"Section: {src['section']} · Score: `{src['score']:.4f}`"
                        )

# ── Section 4: Auto Email Report ─────────────────────────────────────────
    if st.session_state.indexed_files:
        st.markdown("---")
        st.header("4) Auto Email Report")
        st.caption("Sends a full AI-generated analysis report to any email via n8n.")

        recipient_email = st.text_input(
            "Recipient Email",
            placeholder="analyst@company.com",
            key="n8n_email",
        )

        send_disabled = (
            not recipient_email
            or "@" not in recipient_email
            or not st.session_state.session_id
        )

        if st.button("📧 Generate & Send Report", disabled=send_disabled, key="n8n_send"):
            last_file = list(st.session_state.indexed_files.keys())[-1]
            file_path = RAW_DOCS_DIR / st.session_state.session_id / last_file

            if not file_path.exists():
                st.error("File not found. Please re-upload and index.")
            elif not N8N_WEBHOOK_URL:
                st.error("N8N_WEBHOOK_URL not set in Streamlit secrets.")
            else:
                try:
                    with st.spinner("Sending to n8n · Generating report · Emailing..."):
                        with open(file_path, "rb") as f:
                            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
                        resp = requests.post(
                            N8N_WEBHOOK_URL,
                            json={
                                "email":      recipient_email,
                                "filename":   last_file,
                                "pdf_base64": pdf_b64,
                                "session_id": st.session_state.session_id,
                            },
                            timeout=90,
                        )
                    if resp.status_code == 200:
                        st.success(f"✓ Report sent to **{recipient_email}** — check inbox in ~30 seconds.")
                    else:
                        st.error(f"n8n error {resp.status_code}: {resp.text[:200]}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach n8n. Check N8N_WEBHOOK_URL in secrets.")
                except requests.exceptions.Timeout:
                    st.warning("n8n is taking longer than expected — email may still arrive.")
                except Exception as exc:
                    st.error(f"Error: {exc}")
if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()