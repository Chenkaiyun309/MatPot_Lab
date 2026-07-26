from __future__ import annotations

import ast
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib_cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import threading
import zipfile
from typing import Any

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency 'gradio'. Install it first, for example: pip install gradio"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
GENERATOR_SCRIPT = SCRIPT_DIR / "generate_hea_structures.py"
RELAX_SCRIPT = SCRIPT_DIR / "relax_structures.py"
MD_SCRIPT = SCRIPT_DIR / "run_molecular_dynamics.py"
PHASE_SCRIPT = SCRIPT_DIR / "phase_stability.py"
STRESS_SCRIPT = SCRIPT_DIR / "stress_strain_curve.py"
MECHANICAL_SCRIPT = SCRIPT_DIR / "mechanical_properties.py"
GENERATED_DIR = APP_ROOT / "calculator_runs" / "generated_structures"
RELAXED_DIR = APP_ROOT / "calculator_runs" / "relaxed_results"
RELAX_SUMMARY = RELAXED_DIR / "relax_summary.csv"
MD_RESULTS_DIR = APP_ROOT / "calculator_runs" / "md_results"
MD_SUMMARY = MD_RESULTS_DIR / "md_summary.csv"
PHASE_RESULTS_DIR = APP_ROOT / "calculator_runs" / "phase_results"
STRESS_RESULTS_DIR = APP_ROOT / "calculator_runs" / "stress_strain_results"
STRESS_SUMMARY = STRESS_RESULTS_DIR / "stress_strain_summary.csv"
MECHANICAL_RESULTS_DIR = APP_ROOT / "calculator_runs" / "mechanical_results"
MECHANICAL_SUMMARY = MECHANICAL_RESULTS_DIR / "mechanical_summary.csv"
CALCULATOR_RUNS_DIR = APP_ROOT / "calculator_runs"
DEFAULT_RELAX_SUMMARY = RELAX_SUMMARY
RUNNING_PROCESSES: dict[str, subprocess.Popen[str]] = {}
RUNNING_PROCESSES_LOCK = threading.Lock()
DEFAULT_LATTICE_CONSTANTS = {
    "bcc": 3.0,
    "fcc": 3.6,
    "hcp": 2.95,
}
APP_CSS = """
:root {
    --lab-paper: #edf5ff;
    --lab-paper-2: #dcecff;
    --lab-panel: #ffffff;
    --lab-ink: #102033;
    --lab-muted: #5b6f88;
    --lab-line: #bfd3ea;
    --lab-line-soft: #d9e8f7;
    --lab-green: #0969da;
    --lab-green-dark: #074f9f;
    --lab-green-light: #1d8bff;
    --lab-copper: #18a4ff;
    --lab-red: #d92d20;
    --lab-red-dark: #a3160d;
    --lab-red-light: #f04438;
    --lab-shadow: rgba(18, 64, 120, 0.12);
}
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 16px 18px 28px !important;
    font-size: 15px !important;
    color: var(--lab-ink) !important;
    background:
        linear-gradient(90deg, rgba(9, 105, 218, 0.055) 1px, transparent 1px) 0 0 / 30px 30px,
        linear-gradient(0deg, rgba(9, 105, 218, 0.04) 1px, transparent 1px) 0 0 / 30px 30px,
        #f6fbff !important;
    font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Helvetica Neue", sans-serif !important;
}
.gradio-container h1 {font-size: 1.8rem !important;}
.gradio-container h2 {font-size: 1.22rem !important;}
.gradio-container h3 {font-size: 1rem !important;}
.gradio-container label,
.gradio-container .caption,
.gradio-container .prose,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container th,
.gradio-container td {
    font-size: 0.91rem !important;
    letter-spacing: 0 !important;
    font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Helvetica Neue", sans-serif !important;
}
.gradio-container label,
.gradio-container .form label,
.gradio-container .block-label,
.gradio-container [data-testid="block-label"],
.gradio-container .wrap label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    width: fit-content !important;
    margin-bottom: 5px !important;
    padding: 3px 8px 3px 0 !important;
    font-size: 0.86rem !important;
    font-weight: 820 !important;
    color: #12365f !important;
    text-transform: uppercase !important;
    letter-spacing: 0.055em !important;
    line-height: 1.1 !important;
}
.gradio-container label::before,
.gradio-container .form label::before,
.gradio-container .block-label::before,
.gradio-container [data-testid="block-label"]::before,
.gradio-container .wrap label::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 14px;
    border-radius: 999px;
    background: var(--lab-copper);
    box-shadow: none;
}
.app-title-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 16px;
    padding: 14px 16px;
    border: 1px solid var(--lab-line);
    border-radius: 8px;
    background: #ffffff;
    box-shadow: none;
    backdrop-filter: none;
    animation: lab-rise 420ms ease-out both;
}
.app-title-mark {
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: #eaf5ff;
    box-shadow: inset 0 0 0 1px rgba(9, 105, 218, 0.18);
}
.app-title-mark svg {
    width: 30px;
    height: 30px;
}
.app-title-copy {
    display: flex;
    flex: 1;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}
.app-title-heading {
    margin: 0;
    font-size: 1.72rem;
    line-height: 1.05;
    font-weight: 780;
    letter-spacing: 0;
    color: var(--lab-ink);
    font-family: "DIN Alternate", "Avenir Next", "PingFang SC", sans-serif !important;
}
.app-title-subtitle {
    margin: 0;
    font-size: 0.84rem;
    line-height: 1.35;
    color: var(--lab-muted);
    font-weight: 650;
    letter-spacing: 0.01em;
}
.app-status-strip {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
}
.app-status-pill {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 5px 9px;
    border: 1px solid var(--lab-line-soft);
    border-radius: 999px;
    color: #23496f;
    background: #f5fbff;
    font-size: 0.74rem;
    font-weight: 700;
}
.app-status-pill::before {
    content: "";
    width: 6px;
    height: 6px;
    margin-right: 6px;
    border-radius: 999px;
    background: var(--lab-green);
}
.gradio-container .tab-nav,
.gradio-container [role="tablist"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: stretch !important;
    gap: 0 !important;
    width: 100% !important;
    margin: 0 0 12px !important;
    padding: 4px !important;
    border: 1px solid var(--lab-line) !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    box-shadow: none !important;
    overflow: hidden !important;
}
.gradio-container .tab-nav button,
.gradio-container .tab-nav [role="tab"],
.gradio-container .tab-nav .tabitem,
.gradio-container .tab-nav > button,
.gradio-container .tab-nav > [role="tab"],
.gradio-container [role="tablist"] button[role="tab"],
.gradio-container [role="tablist"] > button,
.gradio-container [role="tablist"] [role="tab"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: none !important;
    min-height: 54px !important;
    padding: 10px 8px !important;
    border: 0 !important;
    border-right: 1px solid var(--lab-line-soft) !important;
    border-radius: 0 !important;
    background: linear-gradient(180deg, #ffffff 0%, #f2f8ff 100%) !important;
    color: #59675f !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95), inset 0 -1px 0 rgba(9, 105, 218, 0.08) !important;
    line-height: 1.15 !important;
    text-align: center !important;
    white-space: normal !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-weight: 700 !important;
    transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease !important;
}
.gradio-container .tab-nav button:first-child,
.gradio-container .tab-nav [role="tab"]:first-child,
.gradio-container [role="tablist"] button[role="tab"]:first-child,
.gradio-container [role="tablist"] [role="tab"]:first-child {
    border-radius: 6px 0 0 6px !important;
}
.gradio-container .tab-nav button:last-child,
.gradio-container .tab-nav [role="tab"]:last-child,
.gradio-container [role="tablist"] button[role="tab"]:last-child,
.gradio-container [role="tablist"] [role="tab"]:last-child {
    border-right: 0 !important;
    border-radius: 0 6px 6px 0 !important;
}
.gradio-container .tab-nav button *,
.gradio-container .tab-nav [role="tab"] *,
.gradio-container [role="tablist"] button[role="tab"] *,
.gradio-container [role="tablist"] [role="tab"] * {
    color: inherit !important;
    line-height: 1.15 !important;
    white-space: normal !important;
}
.gradio-container [role="tabpanel"],
.gradio-container .tabitem {
    min-height: 980px !important;
}
.gradio-container .tab-nav button:hover,
.gradio-container .tab-nav [role="tab"]:hover,
.gradio-container .tab-nav .tabitem:hover,
.gradio-container .tab-nav > button:hover,
.gradio-container .tab-nav > [role="tab"]:hover,
.gradio-container [role="tablist"] button[role="tab"]:hover,
.gradio-container [role="tablist"] > button:hover,
.gradio-container [role="tablist"] [role="tab"]:hover {
    background: linear-gradient(180deg, #f8fcff 0%, #e7f3ff 100%) !important;
    color: var(--lab-green-dark) !important;
    border-color: var(--lab-line-soft) !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button[aria-selected="true"],
.gradio-container .tab-nav [role="tab"].selected,
.gradio-container .tab-nav [role="tab"][aria-selected="true"],
.gradio-container .tab-nav .tabitem.selected,
.gradio-container .tab-nav .tabitem[aria-selected="true"],
.gradio-container .tab-nav > button.selected,
.gradio-container .tab-nav > button[aria-selected="true"],
.gradio-container .tab-nav > [role="tab"].selected,
.gradio-container .tab-nav > [role="tab"][aria-selected="true"],
.gradio-container [role="tablist"] button[role="tab"].selected,
.gradio-container [role="tablist"] button[role="tab"][aria-selected="true"],
.gradio-container [role="tablist"] > button.selected,
.gradio-container [role="tablist"] > button[aria-selected="true"],
.gradio-container [role="tablist"] [role="tab"].selected,
.gradio-container [role="tablist"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, #eaf5ff 0%, #d9ecff 100%) !important;
    color: var(--lab-ink) !important;
    border-color: var(--lab-line-soft) !important;
    box-shadow: inset 0 -3px 0 var(--lab-green), inset 0 1px 0 #ffffff !important;
}
.gradio-container .block {
    border-radius: 8px !important;
    border-color: var(--lab-line-soft) !important;
    background: #ffffff !important;
    box-shadow: none !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 6px !important;
    background: #ffffff !important;
    border-color: var(--lab-line-soft) !important;
    color: var(--lab-ink) !important;
}
.gradio-container button {
    min-height: 38px !important;
    border-radius: 6px !important;
    font-weight: 760 !important;
    letter-spacing: 0.01em !important;
    transform: translateY(0) !important;
    transition:
        background 0.16s ease,
        border-color 0.16s ease !important;
}
.gradio-container button.primary,
.gradio-container .primary button {
    background: var(--lab-green) !important;
    border-color: var(--lab-green) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
.gradio-container button.primary:hover,
.gradio-container .primary button:hover {
    background: var(--lab-green-dark) !important;
    border-color: var(--lab-green-dark) !important;
    transform: none !important;
    box-shadow: none !important;
}
.gradio-container button.primary:active,
.gradio-container .primary button:active {
    transform: none !important;
    box-shadow: none !important;
}
.gradio-container button.secondary,
.gradio-container .secondary button {
    background: #ffffff !important;
    border-color: var(--lab-line) !important;
    color: #25352f !important;
}
.gradio-container button.secondary:hover,
.gradio-container .secondary button:hover {
    background: #f2f8ff !important;
    border-color: #aacbed !important;
}
.terminate-button,
.terminate-button button {
    background: var(--lab-red) !important;
    color: #ffffff !important;
    border-color: var(--lab-red) !important;
    box-shadow: none !important;
}
.terminate-button:hover,
.terminate-button button:hover {
    background: var(--lab-red-dark) !important;
    color: #ffffff !important;
    border-color: var(--lab-red-dark) !important;
    transform: none !important;
    box-shadow: none !important;
}
.terminate-button:active,
.terminate-button button:active {
    transform: none !important;
    box-shadow: none !important;
}
.run-log textarea {
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.45 !important;
    color: #163a5f !important;
    background:
        linear-gradient(90deg, rgba(9, 105, 218, 0.055) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(9, 105, 218, 0.04) 1px, transparent 1px) 0 0 / 24px 24px,
        #f7fbff !important;
    border-color: #bdd9f3 !important;
    box-shadow: none !important;
}
.run-log {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    align-self: stretch !important;
    box-sizing: border-box !important;
}
.run-log textarea {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.gradio-container .run-log textarea {
    min-height: 260px !important;
}
.workflow-page,
#run_relaxation_page {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.gradio-container [role="tabpanel"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}
.path-row {
    width: 100% !important;
    max-width: 100% !important;
    display: flex !important;
    align-items: flex-end !important;
    gap: 12px !important;
    box-sizing: border-box !important;
}
.path-input {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
}
.path-input,
.path-input > *,
.path-input input,
.path-input textarea,
#relax_input_dir,
#relax_output_dir,
#relax_input_dir > *,
#relax_output_dir > *,
#relax_input_dir input,
#relax_output_dir input,
#relax_input_dir textarea,
#relax_output_dir textarea {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.path-input input,
.path-input textarea,
#relax_input_dir input,
#relax_output_dir input,
#relax_input_dir textarea,
#relax_output_dir textarea {
    min-height: 42px !important;
    white-space: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    resize: none !important;
}
.path-display-wrap {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.path-display {
    width: 100%;
    max-width: 100%;
    min-height: 42px;
    display: flex;
    align-items: center;
    padding: 9px 12px;
    border: 1px solid var(--lab-line-soft);
    border-radius: 6px;
    background: #ffffff;
    color: #102033;
    font-size: 0.94rem;
    line-height: 1.25;
    white-space: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    box-sizing: border-box;
}
.path-button {
    flex: 0 0 190px !important;
    width: 190px !important;
    max-width: 190px !important;
}
.path-button,
.path-button button {
    background: linear-gradient(180deg, #ffffff 0%, #e8f4ff 100%) !important;
    border-color: #9fc5ec !important;
    color: #0b4f97 !important;
    box-shadow: 0 4px 10px rgba(9, 105, 218, 0.16), inset 0 1px 0 #ffffff !important;
}
.path-button:hover,
.path-button button:hover {
    background: linear-gradient(180deg, #f8fcff 0%, #d8ecff 100%) !important;
    border-color: #7cb4ec !important;
    box-shadow: 0 6px 14px rgba(9, 105, 218, 0.2), inset 0 1px 0 #ffffff !important;
}
.field-title {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin: 0 0 6px;
    color: #12365f;
    font-size: 0.86rem;
    font-weight: 820;
    letter-spacing: 0.055em;
    line-height: 1.1;
    text-transform: uppercase;
}
.field-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 14px;
    border-radius: 999px;
    background: var(--lab-copper);
}
.phase-choice-fix label::before,
.phase-choice-fix .wrap label::before {
    content: none !important;
    display: none !important;
}
.phase-choice-fix label,
.phase-choice-fix .wrap label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    width: auto !important;
    padding: 3px 8px !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    cursor: pointer !important;
}
.phase-choice-fix input[type="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    display: inline-block !important;
    opacity: 1 !important;
    position: static !important;
    pointer-events: auto !important;
    width: 16px !important;
    height: 16px !important;
    margin: 0 6px 0 0 !important;
}
.log-row {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
#generation_log,
#relax_log,
#phase_log,
#mechanical_log,
#stress_log,
#md_log {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    grid-column: 1 / -1 !important;
    box-sizing: border-box !important;
}
#generation_log > *,
#relax_log > *,
#phase_log > *,
#mechanical_log > *,
#stress_log > *,
#md_log > *,
#generation_log textarea,
#relax_log textarea,
#phase_log textarea,
#mechanical_log textarea,
#stress_log textarea,
#md_log textarea {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
#generation_log textarea,
#relax_log textarea,
#phase_log textarea,
#mechanical_log textarea,
#stress_log textarea,
#md_log textarea {
    min-height: 280px !important;
}
.gradio-container table {
    font-size: 0.84rem !important;
}
.gradio-container th {
    color: #33433d !important;
    background: #edf6ff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.035em !important;
}
.gradio-container details {
    border-radius: 8px !important;
    border-color: var(--lab-line-soft) !important;
    background: #ffffff !important;
}
.gradio-container summary {
    color: #26332e !important;
    font-weight: 750 !important;
}
.result-heading {
    margin: 16px 0 9px;
    padding-top: 10px;
    border-top: 1px solid var(--lab-line-soft);
    color: #23332e;
    font-size: 0.88rem;
    font-weight: 780;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.structure-view-card {
    border: 1px solid var(--lab-line-soft);
    border-radius: 8px;
    padding: 12px;
    background: #ffffff;
    box-shadow: 0 12px 28px rgba(18, 64, 120, 0.08);
}
.structure-view-title {
    font-size: 0.95rem;
    font-weight: 760;
    margin-bottom: 6px;
}
.structure-view-file {
    font-size: 0.82rem;
    color: var(--lab-muted);
    margin-bottom: 10px;
}
.mol-viewer {
    width: 100%;
    height: 480px;
    border-radius: 6px;
    overflow: hidden;
    background: #f7fbff;
    border: 1px solid var(--lab-line-soft);
}
.structure-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 10px;
    margin-bottom: 10px;
}
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 9px;
    border-radius: 999px;
    background: #f2f8ff;
    border: 1px solid var(--lab-line-soft);
    font-size: 0.8rem;
    font-weight: 650;
}
.legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    border: 1px solid rgba(0, 0, 0, 0.12);
}
@keyframes lab-rise {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@media (max-width: 760px) {
    .gradio-container {
        padding: 12px !important;
    }
    .app-title-bar {
        align-items: flex-start;
        flex-wrap: wrap;
    }
    .app-status-strip {
        justify-content: flex-start;
        width: 100%;
    }
    .gradio-container [role="tablist"] [role="tab"] {
        flex: 1 1 calc(50% - 4px) !important;
    }
}
"""
APP_TITLE_HTML = """
<div class="app-title-bar">
  <div class="app-title-mark" aria-hidden="true">
    <svg width="54" height="54" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="chgnetLogoWarm" x1="10" y1="8" x2="54" y2="56" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#FBBF24"/>
          <stop offset="0.55" stop-color="#F59E0B"/>
          <stop offset="1" stop-color="#EA580C"/>
        </linearGradient>
        <linearGradient id="chgnetLogoCool" x1="18" y1="14" x2="49" y2="49" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#1F2937"/>
          <stop offset="1" stop-color="#475569"/>
        </linearGradient>
      </defs>
      <path d="M22 10L42 10L53 20L53 42L42 54L22 54L11 42L11 20L22 10Z" fill="url(#chgnetLogoWarm)" opacity="0.18"/>
      <path d="M23 13H41L50 22V42L41 51H23L14 42V22L23 13Z" stroke="url(#chgnetLogoWarm)" stroke-width="3.5" stroke-linejoin="round"/>
      <path d="M22 22L32 16L42 22V34L32 40L22 34V22Z" stroke="url(#chgnetLogoCool)" stroke-width="2.6" stroke-linejoin="round"/>
      <path d="M32 16V40M22 22L42 34M42 22L22 34" stroke="url(#chgnetLogoCool)" stroke-width="2.2" stroke-linecap="round"/>
      <circle cx="32" cy="16" r="4.4" fill="#F59E0B"/>
      <circle cx="22" cy="22" r="3.5" fill="#1F2937"/>
      <circle cx="42" cy="22" r="3.5" fill="#1F2937"/>
      <circle cx="22" cy="34" r="3.5" fill="#1F2937"/>
      <circle cx="42" cy="34" r="3.5" fill="#1F2937"/>
      <circle cx="32" cy="40" r="4.4" fill="#EA580C"/>
    </svg>
  </div>
  <div class="app-title-copy">
    <h1 class="app-title-heading">CHGNet Calculator</h1>
    <p class="app-title-subtitle">Materials workflow console</p>
  </div>
  <div class="app-status-strip" aria-label="Runtime status">
    <span class="app-status-pill">chgnet</span>
    <span class="app-status-pill">CHGNet</span>
    <span class="app-status-pill">calculator_runs</span>
  </div>
</div>
"""
APP_HEAD = """
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(() => {
  function decodeBase64Utf8(value) {
    const binary = atob(value);
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  }

  function renderViewer(el) {
    if (!el || typeof $3Dmol === "undefined") return;
      const payload = el.dataset.structureB64;
      const format = el.dataset.format;
      const colorMap = JSON.parse(el.dataset.colorMap || "{}");
      if (!payload || !format) {
        el.innerHTML = "<div style='padding:20px;'>Missing structure payload.</div>";
        return;
      }
      try {
        const structureText = decodeBase64Utf8(payload);
        el.innerHTML = "";
        const viewer = $3Dmol.createViewer(el, { backgroundColor: "white" });
        viewer.addModel(structureText, format);
        const elements = Object.keys(colorMap);
        if (elements.length) {
          elements.forEach((element) => {
            viewer.setStyle(
              { elem: element },
              {
                sphere: { scale: 0.24, color: colorMap[element] },
                stick: { radius: 0.11, color: colorMap[element] }
              }
            );
          });
        } else {
          viewer.setStyle({}, { sphere: { scale: 0.24 }, stick: { radius: 0.11 } });
        }
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();
        el.dataset.rendered = "1";
    } catch (error) {
      console.error("3Dmol render failed:", error);
      el.innerHTML = "<div style='padding:20px;'>Failed to render structure.</div>";
    }
  }

  function renderAllViewers() {
    document.querySelectorAll(".mol-viewer").forEach((el) => {
      if (el.dataset.rendered !== "1") renderViewer(el);
    });
  }

  window.renderAll3DMolViewers = renderAllViewers;
  window.addEventListener("load", renderAllViewers);
  const observer = new MutationObserver(() => renderAllViewers());
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def extract_constant_list(script_path: Path, constant_name: str) -> list[str]:
    source = script_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    return ast.literal_eval(node.value)
    return []


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else APP_ROOT / path


def choose_local_directory(current_path: str, default_path: str = "") -> str:
    """Open the operating-system folder picker and return the selected path.

    On macOS, Gradio callbacks run outside the main UI thread, so tkinter folder
    dialogs can terminate Python with an uncaught NSException. Use AppleScript
    there and keep tkinter only as a non-macOS fallback.
    """
    fallback = (current_path or default_path or "").strip()
    initial_path = resolve_path(fallback) if fallback else APP_ROOT
    if not initial_path.exists():
        initial_path = APP_ROOT
    if initial_path.is_file():
        initial_path = initial_path.parent

    selected = ""
    if sys.platform == "darwin":
        script = """
        on run argv
            set startPath to item 1 of argv
            set defaultFolder to POSIX file startPath
            set chosenFolder to choose folder with prompt "Select Directory" default location defaultFolder
            return POSIX path of chosenFolder
        end run
        """
        try:
            result = subprocess.run(
                ["osascript", "-e", script, str(initial_path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=300,
            )
        except Exception:
            return fallback

        if result.returncode != 0:
            return fallback
        selected = result.stdout.strip()
    else:
        if threading.current_thread() is not threading.main_thread():
            return fallback
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(initialdir=str(initial_path), title="Select Directory")
            root.destroy()
        except Exception:
            return fallback

    if not selected:
        return fallback

    try:
        return str(Path(selected).resolve().relative_to(APP_ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(selected).resolve()).replace("\\", "/")


def choose_local_directory_with_display(current_path: str, default_path: str = "") -> tuple[str, str]:
    selected = choose_local_directory(current_path, default_path)
    return selected, path_display_html(selected)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_for_dataframe(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    if not rows:
        return ["No data"], []
    headers = list(rows[0].keys())
    data = [[row.get(header, "") for header in headers] for row in rows]
    return headers, data


def dataframe_update(rows: list[dict[str, Any]]) -> dict[str, Any]:
    headers, data = rows_for_dataframe(rows)
    return gr.update(headers=headers, value=data)


def path_display_html(path_value: str) -> str:
    return f"<div class='path-display' title='{html.escape(path_value)}'>{html.escape(path_value)}</div>"


def directory_picker(label: str, value: str, button_text: str, elem_id: str):
    gr.HTML(f"<div class='field-title'>{html.escape(label)}</div>")
    with gr.Row(elem_classes=["path-row"]):
        path_box = gr.Textbox(
            value=value,
            visible=False,
            show_label=False,
            elem_id=elem_id,
        )
        path_display = gr.HTML(value=path_display_html(value), elem_classes=["path-display-wrap"])
        browse_button = gr.Button(
            button_text,
            variant="secondary",
            scale=0,
            elem_classes=["path-button"],
        )
    return path_box, browse_button, path_display


def summarize_repo() -> str:
    candidate_elements = extract_constant_list(GENERATOR_SCRIPT, "DEFAULT_CANDIDATE_ELEMENTS")
    generated_poscars = sorted(GENERATED_DIR.glob("*.POSCAR"))
    generated_cifs = sorted(GENERATED_DIR.glob("*.cif"))
    summary_rows = read_csv_rows(RELAX_SUMMARY)
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]

    return "\n".join(
        [
            "## CHGNet Project Overview",
            "",
            "### 1. `modules/generate_hea_structures.py`",
            "- Builds BCC, FCC, and HCP high-entropy alloy structures from the settings defined in `input.txt`.",
            f"- Default candidate element pool: {', '.join(candidate_elements)}.",
            "- Key inputs include the selected elements, element counts or ratios, lattice constant, supercell size, random seed, number of structures, and output directory.",
            "- Each generated structure is exported as a paired `.cif` and `.POSCAR` file.",
            f"- Current workspace status: {len(generated_poscars)} POSCAR files and {len(generated_cifs)} CIF files are available.",
            "",
            "### 2. `modules/relax_structures.py`",
            "- Runs batch structure relaxation with CHGNet through the ASE/FIRE optimizer stack.",
            "- Accepted input formats: `POSCAR`, `.POSCAR`, `.vasp`, and `.cif`.",
            "- Key controls include the input/output directories, recursive scanning, `EDIFFG` or `fmax`, `NSW` or `max_steps`, and `ISIF` or `relax_cell`.",
            "- Outputs include relaxed POSCAR files, step-by-step CSV logs, `traj` trajectories, optimizer logs, and `relax_summary.csv`.",
            f"- Current workspace status: {len(summary_rows)} relaxation records are listed, including {len(ok_rows)} successful runs.",
            "",
            "### 3. Demo Notes",
            "- The app currently supports project analysis, structure generation, relaxation, mechanical-property evaluation, stress-strain evaluation, molecular dynamics, and phase stability workflows.",
            "- Results tabs can load newly generated summary CSV files from the run tabs without restarting the app.",
        ]
    )


def module_table() -> tuple[list[str], list[list[Any]]]:
    rows = [
        {
            "module": "modules/generate_hea_structures.py",
            "lines": count_lines(GENERATOR_SCRIPT),
            "input": "input.txt config file",
            "output": "generated_structures/*.cif, *.POSCAR",
            "core_function": "Random BCC/FCC/HCP HEA structure generation",
        },
        {
            "module": "modules/relax_structures.py",
            "lines": count_lines(RELAX_SCRIPT),
            "input": "structure directory + CHGNet parameters",
            "output": "calculator_runs/relaxed_results/* + relax_summary.csv",
            "core_function": "Batch relaxation and result summary",
        },
        {
            "module": "modules/run_molecular_dynamics.py",
            "lines": count_lines(MD_SCRIPT),
            "input": "structure directory + MD parameters",
            "output": "calculator_runs/md_results/* + md_summary.csv",
            "core_function": "Batch CHGNet molecular dynamics",
        },
        {
            "module": "modules/phase_stability.py",
            "lines": count_lines(PHASE_SCRIPT),
            "input": "composition scan + phase list + relaxation settings",
            "output": "calculator_runs/phase_results/* + phase_summary.csv",
            "core_function": "Approximate phase stability mapping",
        },
        {
            "module": "modules/stress_strain_curve.py",
            "lines": count_lines(STRESS_SCRIPT),
            "input": "structure directory + strain path + optional atomic relaxation",
            "output": "calculator_runs/stress_strain_results/* + stress_strain_summary.csv",
            "core_function": "Stress-strain curve evaluation with CHGNet",
        },
        {
            "module": "modules/mechanical_properties.py",
            "lines": count_lines(MECHANICAL_SCRIPT),
            "input": "structure directory + finite strains + optional atomic relaxation",
            "output": "calculator_runs/mechanical_results/* + mechanical_summary.csv",
            "core_function": "Elastic tensor fitting and derived mechanical properties",
        },
    ]
    return rows_for_dataframe(rows)


def generated_file_rows(output_dir: Path | None = None) -> list[dict[str, Any]]:
    target_dir = output_dir or GENERATED_DIR
    rows: list[dict[str, Any]] = []
    for poscar_path in sorted(target_dir.glob("*.POSCAR")):
        cif_path = poscar_path.with_suffix(".cif")
        rows.append(
            {
                "structure_id": poscar_path.stem,
                "poscar": str(poscar_path.relative_to(APP_ROOT)),
                "cif": str(cif_path.relative_to(APP_ROOT)) if cif_path.exists() else "",
            }
        )
    return rows


def generated_file_table() -> tuple[list[str], list[list[Any]]]:
    return rows_for_dataframe(generated_file_rows())


def extract_elements_from_structure(path: Path) -> list[str]:
    try:
        from pymatgen.core import Structure

        structure = Structure.from_file(str(path))
        return [element.symbol for element in structure.composition.elements]
    except Exception:
        pass

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    suffix = path.suffix.lower()
    if suffix == ".cif":
        elements: list[str] = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isalpha() and 1 <= len(parts[1]) <= 2:
                symbol = parts[1].capitalize()
                if symbol not in elements:
                    elements.append(symbol)
        return elements

    if len(lines) >= 6:
        tokens = lines[5].split()
        if tokens and all(token[:1].isalpha() for token in tokens):
            return [token.capitalize() for token in tokens]
    return []


def build_element_color_map(elements: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#b07aa1",
        "#76b7b2",
    ]
    return {element: palette[index % len(palette)] for index, element in enumerate(elements)}


def build_legend_html(color_map: dict[str, str]) -> str:
    if not color_map:
        return "<div style='padding:12px;'>No element legend available.</div>"
    items = []
    for element, color in color_map.items():
        items.append(
            f"<span class='legend-item'><span class='legend-swatch' style='background:{html.escape(color)};'></span>{html.escape(element)}</span>"
        )
    return "<div class='structure-legend'>" + "".join(items) + "</div>"


def build_structure_view(path_str: str | Path | None, title: str = "Structure") -> str:
    if not path_str:
        return "<div style='padding:16px;'>No structure selected.</div>"

    structure_path = resolve_path(path_str)
    if not structure_path.exists():
        return f"<div style='padding:16px;'>File not found: {structure_path}</div>"

    suffix = structure_path.suffix.lower()
    if suffix == ".cif":
        structure_format = "cif"
    elif suffix in {".poscar", ".vasp"} or structure_path.name == "POSCAR":
        structure_format = "vasp"
    else:
        return f"<div style='padding:16px;'>Unsupported structure format: {structure_path.name}</div>"

    structure_text = structure_path.read_text(encoding="utf-8")
    viewer_id = f"viewer-{hashlib.md5(str(structure_path).encode('utf-8')).hexdigest()[:12]}"
    structure_b64 = base64.b64encode(structure_text.encode("utf-8")).decode("ascii")
    title_html = html.escape(title)
    file_html = html.escape(str(structure_path.relative_to(APP_ROOT)).replace("\\", "/"))
    elements = extract_elements_from_structure(structure_path)
    color_map = build_element_color_map(elements)
    color_map_json = html.escape(json.dumps(color_map))
    return f"""
    <div class="structure-view-card">
      <div class="structure-view-title">{title_html}</div>
      <div class="structure-view-file">File: {file_html}</div>
      <div
        id="{viewer_id}"
        class="mol-viewer"
        data-format="{structure_format}"
        data-structure-b64="{structure_b64}"
        data-color-map="{color_map_json}"
        data-rendered="0"
      >
        <div style="padding:20px;">Loading 3D viewer...</div>
      </div>
    </div>
    """


def build_structure_legend(path_str: str | Path | None) -> str:
    if not path_str:
        return "<div style='padding:12px;'>No structure selected.</div>"
    structure_path = resolve_path(path_str)
    if not structure_path.exists():
        return f"<div style='padding:12px;'>File not found: {html.escape(str(structure_path))}</div>"
    elements = extract_elements_from_structure(structure_path)
    color_map = build_element_color_map(elements)
    return build_legend_html(color_map)


def load_generated_results(
    output_dir_str: str, selected_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    rows = generated_file_rows(resolve_path(output_dir_str or GENERATED_DIR))
    table_update = dataframe_update(rows)
    choices = [row.get("structure_id", "") for row in rows if row.get("structure_id")]
    chosen_id = selected_id if selected_id in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen_id)
    if chosen_id is None:
        return (
            table_update,
            dropdown_update,
            "<div style='padding:16px;'>No generated structures found.</div>",
            "<div style='padding:12px;'>No element legend available.</div>",
        )

    selected = next((row for row in rows if row.get("structure_id") == chosen_id), rows[0])
    view_path = selected.get("cif") or selected.get("poscar")
    return (
        table_update,
        dropdown_update,
        build_structure_view(view_path, f"Generated Structure: {chosen_id}"),
        build_structure_legend(view_path),
    )


def build_detail_markdown(selected: dict[str, str] | None) -> str:
    if not selected:
        return "### No structure selected"
    return "\n".join(
        [
            f"### Structure: `{selected.get('id', '')}`",
            f"- Status: `{selected.get('status', '')}`",
            f"- Formula: `{selected.get('formula', '')}`",
            f"- Total energy: `{selected.get('energy_eV', '')}` eV",
            f"- Energy per atom: `{selected.get('energy_per_atom_eV', '')}` eV/atom",
            f"- Volume: `{selected.get('volume_A3', '')}` Å³",
            f"- Recorded steps: `{selected.get('num_recorded_steps', '')}`",
            f"- Relaxed POSCAR: `{selected.get('poscar_output', '')}`",
            f"- Step CSV: `{selected.get('step_csv', '')}`",
            f"- Optimizer log: `{selected.get('optimizer_log', '')}`",
        ]
    )


def build_download_zip(selected: dict[str, str] | None) -> str | None:
    if not selected or selected.get("status") != "ok":
        return None

    file_keys = ["poscar_output", "step_csv", "trajectory_file", "optimizer_log"]
    source_files = [resolve_path(selected.get(key, "")) for key in file_keys if selected.get(key)]
    source_files = [path for path in source_files if path.exists()]
    if not source_files:
        return None

    download_dir = CALCULATOR_RUNS_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"{selected.get('id', 'relaxation_result')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_files:
            archive.write(file_path, arcname=file_path.relative_to(APP_ROOT))
    return str(zip_path)


def load_relaxation_rows(summary_path_str: str) -> list[dict[str, str]]:
    return read_csv_rows(resolve_path(summary_path_str or DEFAULT_RELAX_SUMMARY))


def relaxation_detail(structure_id: str, summary_path_str: str) -> tuple[str, dict[str, Any], str | None]:
    rows = load_relaxation_rows(summary_path_str)
    if not rows:
        return (
            "Could not find the selected `relax_summary.csv`.",
            gr.update(value=[], headers=["No data"]),
            None,
        )

    selected = next((row for row in rows if row.get("id") == structure_id), rows[0])
    step_rows = read_csv_rows(resolve_path(selected.get("step_csv", "")))
    step_headers, step_data = rows_for_dataframe(step_rows)
    return (
        build_detail_markdown(selected),
        gr.update(value=step_data, headers=step_headers),
        build_download_zip(selected),
    )


def relaxation_structure_view(structure_id: str, summary_path_str: str) -> str:
    rows = load_relaxation_rows(summary_path_str)
    if not rows:
        return "<div style='padding:16px;'>No relaxation results loaded.</div>"
    selected = next((row for row in rows if row.get("id") == structure_id), rows[0])
    view_path = selected.get("poscar_output") or selected.get("input_file")
    return build_structure_view(view_path, f"Relaxed Structure: {selected.get('id', '')}")


def relaxation_structure_legend(structure_id: str, summary_path_str: str) -> str:
    rows = load_relaxation_rows(summary_path_str)
    if not rows:
        return "<div style='padding:12px;'>No element legend available.</div>"
    selected = next((row for row in rows if row.get("id") == structure_id), rows[0])
    view_path = selected.get("poscar_output") or selected.get("input_file")
    return build_structure_legend(view_path)


def load_relaxation_results(
    summary_path_str: str, selected_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], str | None]:
    rows = load_relaxation_rows(summary_path_str)
    summary_update = dataframe_update(rows)
    choices = [row.get("id", "") for row in rows if row.get("id")]
    chosen_id = selected_id if selected_id in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen_id)

    if chosen_id is None:
        return (
            summary_update,
            dropdown_update,
            "### No relaxation results loaded",
            gr.update(value=[], headers=["No data"]),
            None,
        )

    detail_markdown, step_update, download_path = relaxation_detail(chosen_id, summary_path_str)
    return summary_update, dropdown_update, detail_markdown, step_update, download_path


def build_generation_input(
    elements_text: str,
    ratios_text: str,
    crystal_structure: str,
    lattice_constant: float,
    hcp_c_over_a: float,
    supercell_x: int,
    supercell_y: int,
    supercell_z: int,
    seed: int,
    num_structures: int,
    output_dir: str,
) -> str:
    output_dir = output_dir.strip() or "calculator_runs/generated_structures"
    crystal_structure = crystal_structure.strip().lower()
    lattice_constant = float(lattice_constant)
    hcp_c_over_a = float(hcp_c_over_a)
    supercell_x = int(supercell_x)
    supercell_y = int(supercell_y)
    supercell_z = int(supercell_z)
    seed = int(seed)
    num_structures = int(num_structures)
    return "\n".join(
        [
            f"elements = {elements_text.strip()}",
            f"ratios = {ratios_text.strip()}",
            f"crystal_structure = {crystal_structure}",
            f"lattice_constant = {lattice_constant}",
            f"hcp_c_over_a = {hcp_c_over_a}",
            f"supercell = {supercell_x} {supercell_y} {supercell_z}",
            f"seed = {seed}",
            f"num_structures = {num_structures}",
            f"output_dir = {output_dir}",
        ]
    )


def run_command(command: list[str], task_key: str | None = None) -> str:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command,
        cwd=APP_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        creationflags=creationflags,
    )
    if task_key:
        with RUNNING_PROCESSES_LOCK:
            RUNNING_PROCESSES[task_key] = process
    try:
        stdout, stderr = process.communicate()
    finally:
        if task_key:
            with RUNNING_PROCESSES_LOCK:
                if RUNNING_PROCESSES.get(task_key) is process:
                    RUNNING_PROCESSES.pop(task_key, None)

    output = (stdout or "") + (stderr or "")
    if process.returncode and process.returncode < 0:
        output += f"\nTask was terminated with signal {-process.returncode}."
    elif process.returncode not in (0, None):
        output += f"\nCommand finished with code {process.returncode}."
    return output.strip() or f"Command finished with code {process.returncode}."


def terminate_task(task_key: str) -> str:
    with RUNNING_PROCESSES_LOCK:
        process = RUNNING_PROCESSES.get(task_key)
    if process is None or process.poll() is not None:
        return "No running task found for this workflow."
    try:
        process.terminate()
        return "Termination signal sent. The running task should stop shortly."
    except Exception as exc:
        return f"Failed to terminate task: {exc}"


def run_generation(
    elements_text: str,
    ratios_text: str,
    crystal_structure: str,
    lattice_constant: float,
    hcp_c_over_a: float,
    supercell_x: int,
    supercell_y: int,
    supercell_z: int,
    seed: int,
    num_structures: int,
    output_dir: str,
) -> tuple[str, dict[str, Any]]:
    CALCULATOR_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.strip() or "calculator_runs/generated_structures"
    command = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--elements", elements_text.strip(),
        "--ratios", ratios_text.strip(),
        "--crystal-structure", crystal_structure.strip().lower(),
        "--lattice-constant", str(float(lattice_constant)),
        "--hcp-c-over-a", str(float(hcp_c_over_a)),
        "--supercell", str(int(supercell_x)), str(int(supercell_y)), str(int(supercell_z)),
        "--seed", str(int(seed)),
        "--num-structures", str(int(num_structures)),
        "--output-dir", output_dir,
    ]

    log = run_command(command, "generation")
    output_rows = generated_file_rows(resolve_path(output_dir))
    return log, dataframe_update(output_rows)


def update_hcp_ratio_visibility(crystal_structure: str) -> dict[str, Any]:
    is_hcp = crystal_structure.strip().lower() == "hcp"
    return gr.update(visible=is_hcp, interactive=is_hcp)


def update_structure_dependent_inputs(crystal_structure: str) -> tuple[dict[str, Any], dict[str, Any]]:
    crystal_structure = crystal_structure.strip().lower()
    is_hcp = crystal_structure == "hcp"
    default_lattice_constant = DEFAULT_LATTICE_CONSTANTS.get(crystal_structure, 3.0)
    return (
        gr.update(value=default_lattice_constant),
        gr.update(visible=is_hcp, interactive=is_hcp),
    )


def run_relaxation(
    input_dir: str,
    output_dir: str,
    input_format: str,
    recursive: bool,
    ediffg: float,
    nsw: int,
    isif: int,
    loginterval: int,
) -> tuple[str, dict[str, Any], str]:
    nsw = int(nsw)
    isif = int(isif)
    loginterval = int(loginterval)
    output_dir = output_dir.strip() or "calculator_runs/relaxed_results"
    command = [
        sys.executable,
        str(RELAX_SCRIPT),
        "--input-dir",
        input_dir.strip() or str(GENERATED_DIR),
        "--output-dir",
        output_dir,
        "--input-format",
        input_format,
        "--ediffg",
        str(ediffg),
        "--nsw",
        str(nsw),
        "--isif",
        str(isif),
        "--loginterval",
        str(loginterval),
    ]
    if recursive:
        command.append("--recursive")

    log = run_command(command, "relaxation")
    summary_path = Path(output_dir) / "relax_summary.csv"
    summary_rows = read_csv_rows(resolve_path(summary_path))
    return log, dataframe_update(summary_rows), str(summary_path).replace("\\", "/")


def parse_optional_float(raw_value: str) -> float | None:
    value = raw_value.strip()
    return None if value == "" else float(value)


def parse_optional_int(raw_value: str) -> int | None:
    value = raw_value.strip()
    return None if value == "" else int(value)


def run_molecular_dynamics(
    input_dir: str,
    output_dir: str,
    input_format: str,
    recursive: bool,
    ensemble: str,
    thermostat: str,
    temperature: int,
    starting_temperature: str,
    timestep: float,
    pressure: float,
    taut: str,
    taup: str,
    bulk_modulus: str,
    steps: int,
    loginterval: int,
    use_device: str,
) -> tuple[str, dict[str, Any], str]:
    steps = int(steps)
    loginterval = int(loginterval)
    temperature = int(temperature)
    timestep = float(timestep)
    pressure = float(pressure)
    output_dir = output_dir.strip() or "calculator_runs/md_results"
    command = [
        sys.executable,
        str(MD_SCRIPT),
        "--input-dir",
        input_dir.strip() or str(GENERATED_DIR),
        "--output-dir",
        output_dir,
        "--input-format",
        input_format,
        "--ensemble",
        ensemble,
        "--thermostat",
        thermostat.strip() or "Berendsen_inhomogeneous",
        "--temperature",
        str(temperature),
        "--timestep",
        str(timestep),
        "--pressure",
        str(pressure),
        "--steps",
        str(steps),
        "--loginterval",
        str(loginterval),
    ]
    if recursive:
        command.append("--recursive")
    starting_temperature_value = parse_optional_int(starting_temperature)
    if starting_temperature_value is None:
        starting_temperature_value = temperature
    command.extend(["--starting-temperature", str(starting_temperature_value)])
    taut_value = parse_optional_float(taut)
    if taut_value is not None:
        command.extend(["--taut", str(taut_value)])
    taup_value = parse_optional_float(taup)
    if taup_value is not None:
        command.extend(["--taup", str(taup_value)])
    bulk_modulus_value = parse_optional_float(bulk_modulus)
    if bulk_modulus_value is not None:
        command.extend(["--bulk-modulus", str(bulk_modulus_value)])
    if use_device.strip():
        command.extend(["--use-device", use_device.strip()])

    log = run_command(command, "md")
    summary_path = Path(output_dir) / "md_summary.csv"
    summary_rows = read_csv_rows(resolve_path(summary_path))
    return log, dataframe_update(summary_rows), str(summary_path).replace("\\", "/")


def parse_numeric_csv(path_str: str | Path) -> list[dict[str, float]]:
    if not path_str:
        return []
    path = resolve_path(path_str)
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def parse_md_text_log(path_str: str | Path) -> list[dict[str, float]]:
    path = resolve_path(path_str)
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        values = line.split()
        if len(values) < 5:
            continue
        try:
            rows.append(
                {
                    "time_ps": float(values[0]),
                    "etot_eV": float(values[1]),
                    "epot_eV": float(values[2]),
                    "ekin_eV": float(values[3]),
                    "temperature_K": float(values[4]),
                }
            )
        except ValueError:
            continue
    return rows


def create_md_plot(md_rows: list[dict[str, float]], y_key: str, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if md_rows:
        x_values = [row.get("time_ps", index) for index, row in enumerate(md_rows)]
        y_values = [row.get(y_key, 0.0) for row in md_rows]
        ax.plot(x_values, y_values, color="#1f77b4", linewidth=2)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(y_label)
    else:
        ax.text(0.5, 0.5, "No MD data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def create_md_pressure_plot(md_rows: list[dict[str, float]]):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if md_rows:
        x_values = [row.get("time_ps", index) for index, row in enumerate(md_rows)]
        for key, label, color in [
            ("pressure_xx_GPa", "Pxx", "#1f77b4"),
            ("pressure_yy_GPa", "Pyy", "#ff7f0e"),
            ("pressure_zz_GPa", "Pzz", "#2ca02c"),
        ]:
            y_values = [row.get(key, 0.0) for row in md_rows]
            ax.plot(x_values, y_values, linewidth=2, label=label, color=color)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Pressure (GPa)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No pressure data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Pressure Components vs Time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_md_detail_markdown(selected: dict[str, str] | None) -> str:
    if not selected:
        return "### No MD result selected"
    return "\n".join(
        [
            f"### MD Result: `{selected.get('id', '')}`",
            f"- Status: `{selected.get('status', '')}`",
            f"- Ensemble: `{selected.get('ensemble', '')}`",
            f"- Temperature: `{selected.get('temperature_K', '')}` K",
            f"- Starting temperature: `{selected.get('starting_temperature_K', '')}` K",
            f"- Steps: `{selected.get('steps', '')}`",
            f"- Timestep: `{selected.get('timestep_fs', '')}` fs",
            f"- Target pressure: `{selected.get('pressure', '')}`",
            f"- Final Pxx: `{selected.get('pressure_xx_GPa', '')}` GPa",
            f"- Final Pyy: `{selected.get('pressure_yy_GPa', '')}` GPa",
            f"- Final Pzz: `{selected.get('pressure_zz_GPa', '')}` GPa",
            f"- Frames: `{selected.get('num_frames', '')}`",
            f"- Final energy: `{selected.get('final_energy_eV', '')}` eV",
            f"- Final volume: `{selected.get('final_volume_A3', '')}`",
        ]
    )


def build_md_download_zip(selected: dict[str, str] | None) -> str | None:
    if not selected or selected.get("status") != "ok":
        return None
    file_keys = ["final_structure", "trajectory_file", "log_file", "step_csv"]
    source_files = [resolve_path(selected.get(key, "")) for key in file_keys if selected.get(key)]
    source_files = [path for path in source_files if path.exists()]
    if not source_files:
        return None
    download_dir = CALCULATOR_RUNS_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"CHGNet_Calculator_MD_{selected.get('id', 'result')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_files:
            archive.write(file_path, arcname=file_path.relative_to(APP_ROOT))
    return str(zip_path)


def load_md_results(summary_path_str: str, selected_id: str | None = None):
    rows = read_csv_rows(resolve_path(summary_path_str or MD_SUMMARY))
    summary_update = dataframe_update(rows)
    choices = [row.get("id", "") for row in rows if row.get("id")]
    chosen_id = selected_id if selected_id in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen_id)
    if chosen_id is None:
        empty_fig_1 = create_md_plot([], "epot_eV", "Potential Energy vs Time", "Potential Energy (eV)")
        empty_fig_2 = create_md_plot([], "temperature_K", "Temperature vs Time", "Temperature (K)")
        empty_fig_3 = create_md_pressure_plot([])
        return (
            summary_update,
            dropdown_update,
            "### No MD results loaded",
            empty_fig_1,
            empty_fig_2,
            empty_fig_3,
            None,
        )

    selected = next((row for row in rows if row.get("id") == chosen_id), rows[0])
    step_csv_path = selected.get("step_csv", "")
    md_rows = parse_numeric_csv(step_csv_path)
    if not md_rows:
        md_rows = parse_md_text_log(selected.get("log_file", ""))
    energy_fig = create_md_plot(md_rows, "epot_eV", "Potential Energy vs Time", "Potential Energy (eV)")
    temp_fig = create_md_plot(md_rows, "temperature_K", "Temperature vs Time", "Temperature (K)")
    pressure_fig = create_md_pressure_plot(md_rows)
    return (
        summary_update,
        dropdown_update,
        build_md_detail_markdown(selected),
        energy_fig,
        temp_fig,
        pressure_fig,
        build_md_download_zip(selected),
    )


def build_stress_detail_markdown(selected: dict[str, str] | None) -> str:
    if not selected:
        return "### No stress-strain result selected"
    return "\n".join(
        [
            f"### Stress-Strain Result: `{selected.get('id', '')}`",
            f"- Status: `{selected.get('status', '')}`",
            f"- Formula: `{selected.get('formula', '')}`",
            f"- Direction: `{selected.get('direction', '')}`",
            f"- Strain mode: `{selected.get('strain_mode', '')}`",
            f"- Points: `{selected.get('num_successful_points', '')}` / `{selected.get('num_points', '')}` successful",
            f"- Strain range: `{selected.get('strain_start', '')}` to `{selected.get('strain_end', '')}`",
            f"- Peak stress: `{selected.get('peak_stress_GPa', '')}` GPa at strain `{selected.get('peak_stress_strain', '')}`",
            f"- Curve CSV: `{selected.get('curve_csv', '')}`",
            f"- Plot file: `{selected.get('plot_file', '')}`",
        ]
    )


def create_stress_curve_plot(curve_rows: list[dict[str, str]], selected: dict[str, str] | None):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    valid_rows: list[tuple[float, float]] = []
    for row in curve_rows:
        if row.get("status") != "ok":
            continue
        try:
            valid_rows.append((float(row.get("strain_value", "")), float(row.get("target_stress_GPa", ""))))
        except ValueError:
            continue

    if valid_rows:
        valid_rows.sort(key=lambda item: item[0])
        ax.plot(
            [item[0] for item in valid_rows],
            [item[1] for item in valid_rows],
            marker="o",
            linewidth=2,
            markersize=5,
            color="#0f766e",
        )
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.9)
        direction = (selected or {}).get("direction", "target")
        ax.set_xlabel("Engineering Strain")
        ax.set_ylabel(f"{direction} Stress (GPa)")
    else:
        ax.text(0.5, 0.5, "No stress-strain data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    title = f"Stress-Strain Curve: {(selected or {}).get('id', '')}".strip(": ")
    ax.set_title(title or "Stress-Strain Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_stress_download_zip(selected: dict[str, str] | None) -> str | None:
    if not selected or selected.get("status") not in {"ok", "partial"}:
        return None

    file_keys = ["curve_csv", "plot_file"]
    source_files = [resolve_path(selected.get(key, "")) for key in file_keys if selected.get(key)]
    curve_rows = read_csv_rows(resolve_path(selected["curve_csv"])) if selected.get("curve_csv") else []
    for row in curve_rows:
        for key in ("output_poscar", "optimizer_log"):
            if row.get(key):
                source_files.append(resolve_path(row[key]))

    unique_files: list[Path] = []
    seen = set()
    for path in source_files:
        if not path.exists():
            continue
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    if not unique_files:
        return None

    download_dir = CALCULATOR_RUNS_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"CHGNet_Calculator_StressStrain_{selected.get('id', 'result')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in unique_files:
            archive.write(file_path, arcname=file_path.relative_to(APP_ROOT) if file_path.is_relative_to(APP_ROOT) else file_path.name)
    return str(zip_path)


def load_stress_results(summary_path_str: str, selected_id: str | None = None):
    rows = read_csv_rows(resolve_path(summary_path_str or STRESS_SUMMARY))
    summary_update = dataframe_update(rows)
    choices = [row.get("id", "") for row in rows if row.get("id")]
    chosen_id = selected_id if selected_id in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen_id)

    if chosen_id is None:
        return (
            summary_update,
            dropdown_update,
            "### No stress-strain results loaded",
            gr.update(value=[], headers=["No data"]),
            create_stress_curve_plot([], None),
            None,
        )

    selected = next((row for row in rows if row.get("id") == chosen_id), rows[0])
    curve_rows = read_csv_rows(resolve_path(selected["curve_csv"])) if selected.get("curve_csv") else []
    return (
        summary_update,
        dropdown_update,
        build_stress_detail_markdown(selected),
        dataframe_update(curve_rows),
        create_stress_curve_plot(curve_rows, selected),
        build_stress_download_zip(selected),
    )


def run_stress_strain_ui(
    input_dir: str,
    output_dir: str,
    input_format: str,
    recursive: bool,
    direction: str,
    strain_start: float,
    strain_end: float,
    num_points: int,
    relax_atoms: bool,
    fmax: float,
    max_steps: int,
    write_structures: bool,
    make_plot: bool,
) -> tuple[str, dict[str, Any], str]:
    output_dir = output_dir.strip() or "calculator_runs/stress_strain_results"
    command = [
        sys.executable,
        str(STRESS_SCRIPT),
        "--input-dir",
        input_dir.strip() or str(GENERATED_DIR),
        "--output-dir",
        output_dir,
        "--input-format",
        input_format,
        "--direction",
        direction,
        "--strain-start",
        str(float(strain_start)),
        "--strain-end",
        str(float(strain_end)),
        "--num-points",
        str(int(num_points)),
        "--fmax",
        str(float(fmax)),
        "--max-steps",
        str(int(max_steps)),
    ]
    if recursive:
        command.append("--recursive")
    if relax_atoms:
        command.append("--relax-atoms")
    if not write_structures:
        command.append("--no-write-structures")
    if not make_plot:
        command.append("--no-plot")

    log = run_command(command, "stress")
    summary_path = Path(output_dir) / "stress_strain_summary.csv"
    summary_rows = read_csv_rows(resolve_path(summary_path))
    return log, dataframe_update(summary_rows), str(summary_path).replace("\\", "/")


def build_mechanical_detail_markdown(selected: dict[str, str] | None) -> str:
    if not selected:
        return "### No mechanical result selected"
    return "\n".join(
        [
            f"### Mechanical Result: `{selected.get('id', '')}`",
            f"- Status: `{selected.get('status', '')}`",
            f"- Formula: `{selected.get('formula', '')}`",
            f"- Strain max: `{selected.get('strain_max', '')}`",
            f"- Points per mode: `{selected.get('num_points_per_mode', '')}`",
            f"- Successful strained states: `{selected.get('num_successful_evaluations', '')}`",
            f"- Relax atoms: `{selected.get('relax_atoms', '')}`",
            f"- Bulk modulus (Hill): `{selected.get('bulk_hill_GPa', '')}` GPa",
            f"- Shear modulus (Hill): `{selected.get('shear_hill_GPa', '')}` GPa",
            f"- Young's modulus (Hill): `{selected.get('young_hill_GPa', '')}` GPa",
            f"- Poisson ratio (Hill): `{selected.get('poisson_hill', '')}`",
            f"- Pugh ratio: `{selected.get('pugh_ratio', '')}`",
            f"- Minimum elastic eigenvalue: `{selected.get('min_eigenvalue_GPa', '')}` GPa",
            f"- Mechanically stable: `{selected.get('mechanically_stable', '')}`",
            f"- Elastic tensor CSV: `{selected.get('elastic_tensor_csv', '')}`",
            f"- Properties JSON: `{selected.get('properties_json', '')}`",
        ]
    )


def create_mechanical_tensor_heatmap(matrix_rows: list[dict[str, str]]):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    if not matrix_rows:
        ax.text(0.5, 0.5, "No elastic tensor data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Elastic Tensor Heatmap")
        fig.tight_layout()
        return fig

    labels = [row.get("component", "") for row in matrix_rows]
    matrix = np.array(
        [
            [float(row.get(f"C{index}", "0") or 0.0) for index in range(1, 7)]
            for row in matrix_rows
        ],
        dtype=float,
    )
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(6), [f"C{index}" for index in range(1, 7)])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Elastic Tensor (GPa)")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def create_mechanical_response_plot(response_rows: list[dict[str, str]]):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    axes = axes.flatten()
    stress_keys = [
        "stress_xx_GPa",
        "stress_yy_GPa",
        "stress_zz_GPa",
        "stress_yz_GPa",
        "stress_xz_GPa",
        "stress_xy_GPa",
    ]
    mode_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

    for mode_index, axis in enumerate(axes):
        valid_rows = []
        for row in response_rows:
            if row.get("status") != "ok" or row.get("mode_index") != str(mode_index):
                continue
            try:
                valid_rows.append(
                    (
                        float(row.get("strain_value", "")),
                        float(row.get(stress_keys[mode_index], "")),
                    )
                )
            except ValueError:
                continue
        axis.set_title(f"{mode_labels[mode_index]} mode")
        axis.axhline(0.0, color="#cbd5e1", linewidth=0.8)
        axis.axvline(0.0, color="#cbd5e1", linewidth=0.8)
        if valid_rows:
            valid_rows.sort(key=lambda item: item[0])
            axis.plot(
                [item[0] for item in valid_rows],
                [item[1] for item in valid_rows],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color="#0f766e",
            )
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_xlabel("Strain")
        axis.set_ylabel("Stress (GPa)")
        axis.grid(alpha=0.22)
    fig.tight_layout()
    return fig


def build_mechanical_download_zip(selected: dict[str, str] | None) -> str | None:
    if not selected or selected.get("status") not in {"ok", "partial"}:
        return None

    file_keys = ["elastic_tensor_csv", "response_csv", "properties_json", "plot_file"]
    source_files = [resolve_path(selected.get(key, "")) for key in file_keys if selected.get(key)]
    response_rows = read_csv_rows(resolve_path(selected["response_csv"])) if selected.get("response_csv") else []
    for row in response_rows:
        for key in ("output_poscar", "optimizer_log"):
            if row.get(key):
                source_files.append(resolve_path(row[key]))

    unique_files: list[Path] = []
    seen = set()
    for path in source_files:
        if not path.exists():
            continue
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    if not unique_files:
        return None

    download_dir = CALCULATOR_RUNS_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"CHGNet_Calculator_Mechanical_{selected.get('id', 'result')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in unique_files:
            archive.write(file_path, arcname=file_path.relative_to(APP_ROOT) if file_path.is_relative_to(APP_ROOT) else file_path.name)
    return str(zip_path)


def load_mechanical_results(summary_path_str: str, selected_id: str | None = None):
    rows = read_csv_rows(resolve_path(summary_path_str or MECHANICAL_SUMMARY))
    summary_update = dataframe_update(rows)
    choices = [row.get("id", "") for row in rows if row.get("id")]
    chosen_id = selected_id if selected_id in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen_id)

    if chosen_id is None:
        return (
            summary_update,
            dropdown_update,
            "### No mechanical results loaded",
            gr.update(value=[], headers=["No data"]),
            gr.update(value=[], headers=["No data"]),
            create_mechanical_tensor_heatmap([]),
            create_mechanical_response_plot([]),
            None,
        )

    selected = next((row for row in rows if row.get("id") == chosen_id), rows[0])
    matrix_rows = read_csv_rows(resolve_path(selected["elastic_tensor_csv"])) if selected.get("elastic_tensor_csv") else []
    response_rows = read_csv_rows(resolve_path(selected["response_csv"])) if selected.get("response_csv") else []
    return (
        summary_update,
        dropdown_update,
        build_mechanical_detail_markdown(selected),
        dataframe_update(matrix_rows),
        dataframe_update(response_rows),
        create_mechanical_tensor_heatmap(matrix_rows),
        create_mechanical_response_plot(response_rows),
        build_mechanical_download_zip(selected),
    )


def run_mechanical_properties_ui(
    input_dir: str,
    output_dir: str,
    input_format: str,
    recursive: bool,
    strain_max: float,
    num_points: int,
    relax_atoms: bool,
    fmax: float,
    max_steps: int,
    write_structures: bool,
    make_plot: bool,
) -> tuple[str, dict[str, Any], str]:
    output_dir = output_dir.strip() or "calculator_runs/mechanical_results"
    command = [
        sys.executable,
        str(MECHANICAL_SCRIPT),
        "--input-dir",
        input_dir.strip() or str(GENERATED_DIR),
        "--output-dir",
        output_dir,
        "--input-format",
        input_format,
        "--strain-max",
        str(float(strain_max)),
        "--num-points",
        str(int(num_points)),
        "--fmax",
        str(float(fmax)),
        "--max-steps",
        str(int(max_steps)),
    ]
    if recursive:
        command.append("--recursive")
    if relax_atoms:
        command.append("--relax-atoms")
    if write_structures:
        command.append("--write-structures")
    else:
        command.append("--no-write-structures")
    if not make_plot:
        command.append("--no-plot")

    log = run_command(command, "mechanical")
    summary_path = Path(output_dir) / "mechanical_summary.csv"
    summary_rows = read_csv_rows(resolve_path(summary_path))
    return log, dataframe_update(summary_rows), str(summary_path).replace("\\", "/")


def latest_phase_summary_path() -> str:
    if not PHASE_RESULTS_DIR.exists():
        return ""
    candidates = sorted(
        PHASE_RESULTS_DIR.glob("*/phase_summary.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ""
    return str(candidates[0].relative_to(APP_ROOT)).replace("\\", "/")


def parse_phase_phases(phase_values: list[str] | None) -> str:
    phases = phase_values or ["bcc", "fcc", "hcp"]
    return ",".join(phases)


def update_phase_mode_inputs(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    is_binary = mode.strip().lower() == "binary"
    return gr.update(visible=not is_binary), gr.update(visible=is_binary)


def load_phase_rows(summary_path_str: str) -> list[dict[str, str]]:
    if not summary_path_str.strip():
        return []
    path = resolve_path(summary_path_str)
    if path.is_dir():
        path = path / "phase_summary.csv"
    return read_csv_rows(path)


def create_phase_energy_plot(summary_rows: list[dict[str, str]]):
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    valid_rows = [row for row in summary_rows if row.get("mean_energy_per_atom_eV") not in ("", None)]
    if not valid_rows:
        ax.text(0.5, 0.5, "No phase stability data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Energy vs Composition")
        fig.tight_layout()
        return fig

    binary_mode = any(row.get("composition_x") not in ("", None) for row in valid_rows)
    color_map = {"bcc": "#1f77b4", "fcc": "#ff7f0e", "hcp": "#2ca02c"}
    if binary_mode:
        phases = sorted({row.get("phase", "") for row in valid_rows})
        for phase in phases:
            phase_rows = []
            for row in valid_rows:
                if row.get("phase") != phase:
                    continue
                try:
                    x_value = float(row.get("composition_x", ""))
                    y_value = float(row.get("mean_energy_per_atom_eV", ""))
                except ValueError:
                    continue
                phase_rows.append((x_value, y_value))
            if not phase_rows:
                continue
            phase_rows.sort(key=lambda item: item[0])
            ax.plot([item[0] for item in phase_rows], [item[1] for item in phase_rows], marker="o", linewidth=2, label=phase.upper(), color=color_map.get(phase))
        ax.set_xlabel("Composition variable x")
        ax.legend()
    else:
        phases = [row.get("phase", "") for row in valid_rows]
        energies = [float(row["mean_energy_per_atom_eV"]) for row in valid_rows]
        colors = [color_map.get(phase, "#4e79a7") for phase in phases]
        ax.bar([phase.upper() for phase in phases], energies, color=colors)
        ax.set_xlabel("Phase")
    ax.set_ylabel("Energy per atom (eV)")
    ax.set_title("Energy vs Composition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def create_phase_stability_plot(summary_rows: list[dict[str, str]]):
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    valid_rows = [row for row in summary_rows if row.get("mean_energy_per_atom_eV") not in ("", None)]
    if not valid_rows:
        ax.text(0.5, 0.5, "No stable-phase data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Stable Phase Map")
        fig.tight_layout()
        return fig

    winners = {}
    for row in valid_rows:
        point_index = row.get("point_index", "")
        if point_index and point_index not in winners:
            winners[point_index] = {
                "composition_x": row.get("composition_x", ""),
                "composition_label": row.get("composition_label", ""),
                "phase": row.get("lowest_energy_phase", ""),
            }

    phase_order = ["bcc", "fcc", "hcp"]
    phase_to_y = {phase: index for index, phase in enumerate(phase_order)}
    binary_mode = any(item.get("composition_x") not in ("", None) for item in winners.values())
    color_map = {"bcc": "#1f77b4", "fcc": "#ff7f0e", "hcp": "#2ca02c"}

    if binary_mode:
        points = []
        for item in winners.values():
            phase = item.get("phase", "")
            try:
                x_value = float(item.get("composition_x", ""))
            except ValueError:
                continue
            if phase not in phase_to_y:
                continue
            points.append((x_value, phase_to_y[phase], phase))
        points.sort(key=lambda item: item[0])
        if points:
            ax.scatter([item[0] for item in points], [item[1] for item in points], s=90, c=[color_map.get(item[2], "#4e79a7") for item in points])
            ax.plot([item[0] for item in points], [item[1] for item in points], color="#999", alpha=0.35)
        ax.set_xlabel("Composition variable x")
        ax.set_yticks(list(phase_to_y.values()), [phase.upper() for phase in phase_order])
    else:
        best_phase = next((item.get("phase", "") for item in winners.values() if item.get("phase")), "")
        if best_phase:
            ax.bar([best_phase.upper()], [1], color=color_map.get(best_phase, "#4e79a7"))
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Stable")
        else:
            ax.text(0.5, 0.5, "No winner available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    ax.set_title("Stable Phase Map")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def build_phase_detail_markdown(summary_rows: list[dict[str, str]], composition_label: str | None) -> str:
    if not summary_rows:
        return "### No phase stability results loaded"
    composition_rows = [row for row in summary_rows if row.get("composition_label") == composition_label]
    if not composition_rows:
        composition_rows = [row for row in summary_rows if row.get("composition_label")]
        if not composition_rows:
            return "### No phase stability results loaded"
        first_label = composition_rows[0].get("composition_label", "")
        composition_rows = [row for row in summary_rows if row.get("composition_label") == first_label]

    selected = sorted(
        composition_rows,
        key=lambda row: float(row.get("mean_energy_per_atom_eV", "inf")) if row.get("mean_energy_per_atom_eV") not in ("", None) else float("inf"),
    )
    reference = selected[0]
    lines = [
        f"### Composition: `{reference.get('composition_label', '')}`",
        f"- Mode: `{reference.get('mode', '')}`",
        f"- Lowest-energy phase: `{reference.get('lowest_energy_phase', '')}`",
    ]
    if reference.get("composition_x") not in ("", None):
        lines.append(f"- Composition variable x: `{reference.get('composition_x', '')}`")
    lines.append("- Phase ranking:")
    for row in selected:
        lines.append(
            f"  - `{row.get('phase', '').upper()}`: `{row.get('mean_energy_per_atom_eV', '')}` eV/atom, delta=`{row.get('delta_to_best_meV_per_atom', '')}` meV/atom"
        )
    return "\n".join(lines)


def load_phase_results(summary_path_str: str, composition_label: str | None = None):
    rows = load_phase_rows(summary_path_str)
    summary_update = dataframe_update(rows)
    choices = []
    seen = set()
    for row in rows:
        label = row.get("composition_label", "")
        if label and label not in seen:
            seen.add(label)
            choices.append(label)
    chosen = composition_label if composition_label in choices else (choices[0] if choices else None)
    dropdown_update = gr.update(choices=choices, value=chosen)
    detail = build_phase_detail_markdown(rows, chosen) if chosen else "### No phase stability results loaded"
    return summary_update, dropdown_update, detail, create_phase_energy_plot(rows), create_phase_stability_plot(rows)


def run_phase_stability_ui(
    mode: str,
    elements_text: str,
    ratios_text: str,
    element_a: str,
    element_b: str,
    x_start: float,
    x_end: float,
    num_points: int,
    phases: list[str],
    lattice_bcc: float,
    lattice_fcc: float,
    lattice_hcp: float,
    hcp_c_over_a: float,
    supercell_x: int,
    supercell_y: int,
    supercell_z: int,
    samples_per_phase: int,
    seed: int,
    no_relax: bool,
    ediffg: float,
    nsw: int,
    isif: int,
    loginterval: int,
    task_name: str,
    output_dir: str,
) -> tuple[str, dict[str, Any], str]:
    output_dir = output_dir.strip() or "calculator_runs/phase_results"
    command = [
        sys.executable,
        str(PHASE_SCRIPT),
        "--mode", mode.strip().lower(),
        "--output-dir", output_dir,
        "--phases", parse_phase_phases(phases),
        "--samples-per-phase", str(int(samples_per_phase)),
        "--seed", str(int(seed)),
        "--supercell", str(int(supercell_x)), str(int(supercell_y)), str(int(supercell_z)),
        "--lattice-bcc", str(float(lattice_bcc)),
        "--lattice-fcc", str(float(lattice_fcc)),
        "--lattice-hcp", str(float(lattice_hcp)),
        "--hcp-c-over-a", str(float(hcp_c_over_a)),
        "--ediffg", str(float(ediffg)),
        "--nsw", str(int(nsw)),
        "--isif", str(int(isif)),
        "--loginterval", str(int(loginterval)),
    ]
    if task_name.strip():
        command.extend(["--task-name", task_name.strip()])
    if mode.strip().lower() == "fixed":
        command.extend(["--elements", elements_text.strip(), "--ratios", ratios_text.strip()])
    else:
        command.extend([
            "--element-a", element_a.strip(),
            "--element-b", element_b.strip(),
            "--x-start", str(float(x_start)),
            "--x-end", str(float(x_end)),
            "--num-points", str(int(num_points)),
        ])
    if no_relax:
        command.append("--no-relax")

    log = run_command(command, "phase")
    summary_path = ""
    for line in reversed(log.splitlines()):
        if line.startswith("Phase summary CSV:"):
            summary_path = line.split(":", 1)[1].strip()
            break
    if not summary_path:
        summary_path = latest_phase_summary_path()
    summary_rows = load_phase_rows(summary_path)
    return log, dataframe_update(summary_rows), str(summary_path).replace("\\", "/")


def build_demo() -> gr.Blocks:
    module_headers, module_data = module_table()
    generated_headers, generated_data = generated_file_table()
    generated_rows = generated_file_rows()
    generated_choices = [row.get("structure_id", "") for row in generated_rows if row.get("structure_id")] or [""]
    summary_headers, summary_data = rows_for_dataframe(load_relaxation_rows(str(DEFAULT_RELAX_SUMMARY)))
    summary_rows = load_relaxation_rows(str(DEFAULT_RELAX_SUMMARY))
    structure_choices = [row.get("id", "") for row in summary_rows if row.get("id")] or [""]
    md_headers, md_data = rows_for_dataframe(read_csv_rows(MD_SUMMARY))
    stress_headers, stress_data = rows_for_dataframe(read_csv_rows(STRESS_SUMMARY))
    stress_rows = read_csv_rows(STRESS_SUMMARY)
    stress_choices = [row.get("id", "") for row in stress_rows if row.get("id")] or [""]
    mechanical_headers, mechanical_data = rows_for_dataframe(read_csv_rows(MECHANICAL_SUMMARY))
    mechanical_rows = read_csv_rows(MECHANICAL_SUMMARY)
    mechanical_choices = [row.get("id", "") for row in mechanical_rows if row.get("id")] or [""]
    phase_default_summary = latest_phase_summary_path()
    phase_headers, phase_data = rows_for_dataframe(load_phase_rows(phase_default_summary))
    phase_rows = load_phase_rows(phase_default_summary)
    phase_choices = []
    for row in phase_rows:
        label = row.get("composition_label", "")
        if label and label not in phase_choices:
            phase_choices.append(label)
    phase_choices = phase_choices or [""]

    with gr.Blocks(title="CHGNet Calculator") as demo:
        gr.HTML(APP_TITLE_HTML)

        with gr.Tab("Project Analysis"):
            repo_summary = gr.Markdown(value=summarize_repo())
            gr.Dataframe(
                headers=module_headers,
                value=module_data,
                label="Module Summary",
                interactive=False,
            )
            input_preview = gr.Code(
                value=read_text(GENERATOR_SCRIPT),
                language="python",
                label="Generator script (generate_hea_structures.py)",
                interactive=False,
            )
            refresh_overview_btn = gr.Button("Refresh Analysis", variant="secondary")

            refresh_overview_btn.click(
                fn=lambda: (summarize_repo(), read_text(GENERATOR_SCRIPT)),
                outputs=[repo_summary, input_preview],
            )

        with gr.Tab("Structure Generation"):
            with gr.Row():
                elements_text = gr.Textbox(label="Elements", value="Hf Nb Ta Ti Zr")
                ratios_text = gr.Textbox(label="Ratios", value="1 1 1 1 1")
            with gr.Row():
                crystal_structure = gr.Dropdown(
                    label="Crystal Structure",
                    choices=["bcc", "fcc", "hcp"],
                    value="bcc",
                )
                num_structures = gr.Number(label="Number of Structures", value=3, precision=0)
            with gr.Row():
                supercell_x = gr.Number(label="Supercell X", value=4, precision=0)
                supercell_y = gr.Number(label="Supercell Y", value=4, precision=0)
                supercell_z = gr.Number(label="Supercell Z", value=4, precision=0)
            with gr.Accordion("Advanced generation settings", open=False):
                with gr.Row():
                    lattice_constant = gr.Number(label="Lattice Constant", value=DEFAULT_LATTICE_CONSTANTS["bcc"])
                    hcp_c_over_a = gr.Number(label="HCP c/a", value=1.633, visible=False, interactive=False)
                    seed = gr.Number(label="Random Seed", value=42, precision=0)
                output_dir, output_dir_browse, output_dir_display = directory_picker(
                    "Output Directory",
                    "calculator_runs/generated_structures",
                    "Browse Folder...",
                    "generation_output_dir",
                )
                preview_config = gr.Code(label="Preview input.txt", language="python", interactive=False)
            with gr.Row():
                generate_btn = gr.Button("Run Structure Generation", variant="primary")
                stop_generation_btn = gr.Button("Terminate Generation", elem_classes=["terminate-button"])
            gr.HTML("<div class='field-title'>Run Log</div>")
            generate_log = gr.Textbox(lines=12, show_label=False, elem_id="generation_log", elem_classes=["run-log"])
            generated_df = gr.Dataframe(
                headers=generated_headers,
                value=generated_data,
                label="Generated Structure Files",
                interactive=False,
            )
            gr.HTML("<div class='result-heading'>Generated Structure Viewer</div>")
            with gr.Row():
                generated_structure_id = gr.Dropdown(
                    label="Select Generated Structure",
                    choices=generated_choices,
                    value=generated_choices[0],
                )
                view_generated_btn = gr.Button("View Structure", variant="secondary")
            generated_legend_html = gr.HTML(
                value=build_structure_legend(
                    generated_rows[0].get("cif") or generated_rows[0].get("poscar")
                )
                if generated_rows
                else "<div style='padding:12px;'>No element legend available.</div>"
            )
            generated_structure_html = gr.HTML(
                value=build_structure_view(
                    generated_rows[0].get("cif") or generated_rows[0].get("poscar"),
                    f"Generated Structure: {generated_choices[0]}",
                )
                if generated_rows
                else "<div style='padding:16px;'>No generated structures found.</div>"
            )

            preview_inputs = [
                elements_text,
                ratios_text,
                crystal_structure,
                lattice_constant,
                hcp_c_over_a,
                supercell_x,
                supercell_y,
                supercell_z,
                seed,
                num_structures,
                output_dir,
            ]

            crystal_structure.change(
                fn=update_structure_dependent_inputs,
                inputs=crystal_structure,
                outputs=[lattice_constant, hcp_c_over_a],
            )
            for component in preview_inputs:
                component.change(fn=build_generation_input, inputs=preview_inputs, outputs=preview_config)
            output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/generated_structures"),
                inputs=output_dir,
                outputs=[output_dir, output_dir_display],
            )

            demo.load(
                fn=update_structure_dependent_inputs,
                inputs=crystal_structure,
                outputs=[lattice_constant, hcp_c_over_a],
            )
            demo.load(fn=build_generation_input, inputs=preview_inputs, outputs=preview_config)
            stop_generation_btn.click(fn=lambda: terminate_task("generation"), outputs=generate_log)
            generate_btn.click(fn=run_generation, inputs=preview_inputs, outputs=[generate_log, generated_df]).then(
                fn=load_generated_results,
                inputs=[output_dir, generated_structure_id],
                outputs=[generated_df, generated_structure_id, generated_structure_html, generated_legend_html],
            )
            output_dir.change(
                fn=load_generated_results,
                inputs=[output_dir, generated_structure_id],
                outputs=[generated_df, generated_structure_id, generated_structure_html, generated_legend_html],
            )
            view_generated_btn.click(
                fn=lambda selected_id, out_dir: load_generated_results(out_dir, selected_id)[2:4],
                inputs=[generated_structure_id, output_dir],
                outputs=[generated_structure_html, generated_legend_html],
            )
            generated_structure_id.change(
                fn=lambda selected_id, out_dir: load_generated_results(out_dir, selected_id)[2:4],
                inputs=[generated_structure_id, output_dir],
                outputs=[generated_structure_html, generated_legend_html],
            )

        with gr.Tab("Relaxation"):
            with gr.Column(elem_id="run_relaxation_page", elem_classes=["workflow-page"]):
                relax_input_dir, relax_input_dir_browse, relax_input_dir_display = directory_picker(
                    "Input Directory",
                    "calculator_runs/generated_structures",
                    "Browse Input...",
                    "relax_input_dir",
                )
                relax_output_dir, relax_output_dir_browse, relax_output_dir_display = directory_picker(
                    "Output Directory",
                    "calculator_runs/relaxed_results",
                    "Browse Output...",
                    "relax_output_dir",
                )
                with gr.Accordion("Advanced relaxation settings", open=False):
                    with gr.Row():
                        input_format = gr.Dropdown(label="Input Format", choices=["poscar", "cif", "auto"], value="poscar")
                        recursive = gr.Checkbox(label="Recursive Scan", value=False)
                    with gr.Row():
                        ediffg = gr.Number(label="EDIFFG", value=-0.005)
                        nsw = gr.Number(label="NSW", value=200, precision=0)
                        isif = gr.Number(label="ISIF", value=3, precision=0)
                        loginterval = gr.Number(label="loginterval", value=1, precision=0)
                with gr.Row():
                    relax_btn = gr.Button("Run CHGNet Relaxation", variant="primary")
                    stop_relax_btn = gr.Button("Terminate Relaxation", elem_classes=["terminate-button"])
                gr.HTML("<div class='field-title'>Run Log</div>")
                with gr.Row(elem_classes=["log-row"]):
                    relax_log = gr.Textbox(lines=12, show_label=False, elem_id="relax_log", elem_classes=["run-log"])
                relax_result_df = gr.Dataframe(
                    headers=["No data"],
                    value=[],
                    label="Current Run Summary",
                    interactive=False,
                )
            relax_input_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/generated_structures"),
                inputs=relax_input_dir,
                outputs=[relax_input_dir, relax_input_dir_display],
            )
            relax_output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/relaxed_results"),
                inputs=relax_output_dir,
                outputs=[relax_output_dir, relax_output_dir_display],
            )
            stop_relax_btn.click(fn=lambda: terminate_task("relaxation"), outputs=relax_log)

            gr.HTML("<div class='result-heading'>Relaxation Results</div>")
            summary_path_box = gr.Textbox(
                label="Summary CSV Path",
                value=str(DEFAULT_RELAX_SUMMARY.relative_to(APP_ROOT)).replace("\\", "/"),
            )
            with gr.Row():
                structure_id = gr.Dropdown(
                    label="Select Structure",
                    choices=structure_choices,
                    value=structure_choices[0],
                )
                refresh_summary_btn = gr.Button("Refresh Summary", variant="secondary")
            summary_df = gr.Dataframe(
                headers=summary_headers,
                value=summary_data,
                label="Relaxation Summary",
                interactive=False,
            )
            detail_md = gr.Markdown()
            step_df = gr.Dataframe(
                headers=["No data"],
                value=[],
                label="Per-step Relaxation Data",
                interactive=False,
            )
            download_file = gr.File(label="Download Selected Relaxation Files", interactive=False)
            gr.HTML("<div class='result-heading'>Relaxed Structure Viewer</div>")
            view_relaxed_btn = gr.Button("View Structure", variant="secondary")
            relaxed_legend_html = gr.HTML(value="<div style='padding:12px;'>No element legend available.</div>")
            relaxed_structure_html = gr.HTML(value="<div style='padding:16px;'>Select a structure to view.</div>")

            structure_id.change(
                fn=relaxation_detail,
                inputs=[structure_id, summary_path_box],
                outputs=[detail_md, step_df, download_file],
            )
            structure_id.change(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            )
            structure_id.change(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )
            refresh_summary_btn.click(
                fn=load_relaxation_results,
                inputs=[summary_path_box, structure_id],
                outputs=[summary_df, structure_id, detail_md, step_df, download_file],
            )
            refresh_summary_btn.click(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            )
            refresh_summary_btn.click(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )
            view_relaxed_btn.click(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            )
            view_relaxed_btn.click(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )
            demo.load(
                fn=load_relaxation_results,
                inputs=[summary_path_box, structure_id],
                outputs=[summary_df, structure_id, detail_md, step_df, download_file],
            )
            summary_path_box.change(
                fn=load_relaxation_results,
                inputs=[summary_path_box, structure_id],
                outputs=[summary_df, structure_id, detail_md, step_df, download_file],
            )
            demo.load(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            )
            demo.load(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )
            summary_path_box.change(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            )
            summary_path_box.change(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )

        with gr.Tab("Phase Stability"):
            phase_mode = gr.Dropdown(label="Scan Mode", choices=["fixed", "binary"], value="fixed")
            with gr.Group(visible=True) as phase_fixed_group:
                with gr.Row():
                    phase_elements_text = gr.Textbox(label="Elements", value="Hf Nb Ta Ti Zr")
                    phase_ratios_text = gr.Textbox(label="Ratios", value="1 1 1 1 1")
            with gr.Group(visible=False) as phase_binary_group:
                with gr.Row():
                    phase_element_a = gr.Textbox(label="Element A", value="Ti")
                    phase_element_b = gr.Textbox(label="Element B", value="Nb")
                with gr.Row():
                    phase_x_start = gr.Number(label="x Start", value=0.1)
                    phase_x_end = gr.Number(label="x End", value=0.9)
                    phase_num_points = gr.Number(label="Number of Points", value=9, precision=0)
            with gr.Row():
                phase_list = gr.CheckboxGroup(
                    label="Candidate Phases",
                    choices=["bcc", "fcc", "hcp"],
                    value=["bcc", "fcc", "hcp"],
                    elem_classes=["phase-choice-fix"],
                )
                phase_samples = gr.Number(label="Samples per Phase", value=2, precision=0)
            with gr.Accordion("Advanced phase settings", open=False):
                with gr.Row():
                    phase_seed = gr.Number(label="Random Seed", value=42, precision=0)
                    phase_lattice_bcc = gr.Number(label="BCC Lattice Constant", value=DEFAULT_LATTICE_CONSTANTS["bcc"])
                    phase_lattice_fcc = gr.Number(label="FCC Lattice Constant", value=DEFAULT_LATTICE_CONSTANTS["fcc"])
                    phase_lattice_hcp = gr.Number(label="HCP Lattice Constant", value=DEFAULT_LATTICE_CONSTANTS["hcp"])
                    phase_hcp_c_over_a = gr.Number(label="HCP c/a", value=1.633)
                with gr.Row():
                    phase_supercell_x = gr.Number(label="Supercell X", value=3, precision=0)
                    phase_supercell_y = gr.Number(label="Supercell Y", value=3, precision=0)
                    phase_supercell_z = gr.Number(label="Supercell Z", value=3, precision=0)
                with gr.Row():
                    phase_no_relax = gr.Checkbox(label="Single-point only (skip relaxation)", value=False)
                    phase_ediffg = gr.Number(label="EDIFFG", value=-0.005)
                    phase_nsw = gr.Number(label="NSW", value=200, precision=0)
                    phase_isif = gr.Number(label="ISIF", value=3, precision=0)
                    phase_loginterval = gr.Number(label="loginterval", value=1, precision=0)
                phase_task_name = gr.Textbox(label="Task Name (Optional)", value="")
                phase_output_dir, phase_output_dir_browse, phase_output_dir_display = directory_picker(
                    "Output Directory",
                    "calculator_runs/phase_results",
                    "Browse Folder...",
                    "phase_output_dir",
                )
            with gr.Row():
                phase_run_btn = gr.Button("Run Phase Stability", variant="primary")
                stop_phase_btn = gr.Button("Terminate Phase Stability", elem_classes=["terminate-button"])
            gr.HTML("<div class='field-title'>Run Log</div>")
            phase_log = gr.Textbox(lines=12, show_label=False, elem_id="phase_log", elem_classes=["run-log"])
            phase_summary_path = gr.Textbox(label="Phase Summary CSV Path", value=phase_default_summary)
            phase_result_df = gr.Dataframe(headers=phase_headers, value=phase_data, label="Phase Stability Summary", interactive=False)

            gr.HTML("<div class='result-heading'>Phase Stability Results</div>")
            with gr.Row():
                phase_result_id = gr.Dropdown(label="Select Composition Point", choices=phase_choices, value=phase_choices[0])
                phase_refresh_btn = gr.Button("Refresh Phase Results", variant="secondary")
            phase_detail = gr.Markdown()
            with gr.Row():
                phase_energy_plot = gr.Plot(label="Energy vs Composition")
                phase_stability_plot = gr.Plot(label="Stable Phase Map")

            phase_mode.change(fn=update_phase_mode_inputs, inputs=phase_mode, outputs=[phase_fixed_group, phase_binary_group])
            demo.load(fn=update_phase_mode_inputs, inputs=phase_mode, outputs=[phase_fixed_group, phase_binary_group])
            phase_output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/phase_results"),
                inputs=phase_output_dir,
                outputs=[phase_output_dir, phase_output_dir_display],
            )
            stop_phase_btn.click(fn=lambda: terminate_task("phase"), outputs=phase_log)
            phase_run_btn.click(
                fn=run_phase_stability_ui,
                inputs=[
                    phase_mode,
                    phase_elements_text,
                    phase_ratios_text,
                    phase_element_a,
                    phase_element_b,
                    phase_x_start,
                    phase_x_end,
                    phase_num_points,
                    phase_list,
                    phase_lattice_bcc,
                    phase_lattice_fcc,
                    phase_lattice_hcp,
                    phase_hcp_c_over_a,
                    phase_supercell_x,
                    phase_supercell_y,
                    phase_supercell_z,
                    phase_samples,
                    phase_seed,
                    phase_no_relax,
                    phase_ediffg,
                    phase_nsw,
                    phase_isif,
                    phase_loginterval,
                    phase_task_name,
                    phase_output_dir,
                ],
                outputs=[phase_log, phase_result_df, phase_summary_path],
            ).then(
                fn=load_phase_results,
                inputs=[phase_summary_path, phase_result_id],
                outputs=[phase_result_df, phase_result_id, phase_detail, phase_energy_plot, phase_stability_plot],
            )
            phase_refresh_btn.click(
                fn=load_phase_results,
                inputs=[phase_summary_path, phase_result_id],
                outputs=[phase_result_df, phase_result_id, phase_detail, phase_energy_plot, phase_stability_plot],
            )
            phase_result_id.change(
                fn=load_phase_results,
                inputs=[phase_summary_path, phase_result_id],
                outputs=[phase_result_df, phase_result_id, phase_detail, phase_energy_plot, phase_stability_plot],
            )
            phase_summary_path.change(
                fn=load_phase_results,
                inputs=[phase_summary_path, phase_result_id],
                outputs=[phase_result_df, phase_result_id, phase_detail, phase_energy_plot, phase_stability_plot],
            )
            demo.load(
                fn=load_phase_results,
                inputs=[phase_summary_path, phase_result_id],
                outputs=[phase_result_df, phase_result_id, phase_detail, phase_energy_plot, phase_stability_plot],
            )

        with gr.Tab("Mechanical Properties"):
            mechanical_input_dir, mechanical_input_dir_browse, mechanical_input_dir_display = directory_picker(
                "Input Directory",
                "calculator_runs/generated_structures",
                "Browse Input...",
                "mechanical_input_dir",
            )
            mechanical_output_dir, mechanical_output_dir_browse, mechanical_output_dir_display = directory_picker(
                "Output Directory",
                "calculator_runs/mechanical_results",
                "Browse Output...",
                "mechanical_output_dir",
            )
            with gr.Accordion("Advanced mechanical settings", open=False):
                with gr.Row():
                    mechanical_input_format = gr.Dropdown(label="Input Format", choices=["poscar", "cif", "auto"], value="poscar")
                    mechanical_recursive = gr.Checkbox(label="Recursive Scan", value=False)
                with gr.Row():
                    mechanical_strain_max = gr.Number(label="Max Strain", value=0.01)
                    mechanical_num_points = gr.Number(label="Points per Mode", value=5, precision=0)
                    mechanical_relax_atoms = gr.Checkbox(label="Relax Atoms at Each Strain", value=False)
                with gr.Row():
                    mechanical_fmax = gr.Number(label="Fmax", value=0.03)
                    mechanical_max_steps = gr.Number(label="Max Relax Steps", value=200, precision=0)
                    mechanical_write_structures = gr.Checkbox(label="Write Strained Structures", value=False)
                    mechanical_make_plot = gr.Checkbox(label="Save PNG Plot", value=True)
            with gr.Row():
                mechanical_run_btn = gr.Button("Run Mechanical Properties", variant="primary")
                stop_mechanical_btn = gr.Button("Terminate Mechanical Properties", elem_classes=["terminate-button"])
            gr.HTML("<div class='field-title'>Run Log</div>")
            mechanical_log = gr.Textbox(lines=12, show_label=False, elem_id="mechanical_log", elem_classes=["run-log"])
            mechanical_summary_path = gr.Textbox(label="Mechanical Summary CSV Path", value="calculator_runs/mechanical_results/mechanical_summary.csv")
            mechanical_result_df = gr.Dataframe(
                headers=mechanical_headers,
                value=mechanical_data,
                label="Mechanical Summary",
                interactive=False,
            )

            gr.HTML("<div class='result-heading'>Mechanical Results Viewer</div>")
            with gr.Row():
                mechanical_result_id = gr.Dropdown(label="Select Structure", choices=mechanical_choices, value=mechanical_choices[0])
                mechanical_refresh_btn = gr.Button("Refresh Mechanical Results", variant="secondary")
            mechanical_detail = gr.Markdown()
            mechanical_tensor_df = gr.Dataframe(headers=["No data"], value=[], label="Elastic Tensor (GPa)", interactive=False)
            mechanical_response_df = gr.Dataframe(headers=["No data"], value=[], label="Finite-Strain Response Data", interactive=False)
            with gr.Row():
                mechanical_tensor_plot = gr.Plot(label="Elastic Tensor Heatmap")
                mechanical_response_plot = gr.Plot(label="Stress Response by Mode")
            mechanical_download_file = gr.File(label="Download Mechanical Result Files", interactive=False)

            mechanical_input_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/generated_structures"),
                inputs=mechanical_input_dir,
                outputs=[mechanical_input_dir, mechanical_input_dir_display],
            )
            mechanical_output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/mechanical_results"),
                inputs=mechanical_output_dir,
                outputs=[mechanical_output_dir, mechanical_output_dir_display],
            )
            stop_mechanical_btn.click(fn=lambda: terminate_task("mechanical"), outputs=mechanical_log)

            mechanical_run_btn.click(
                fn=run_mechanical_properties_ui,
                inputs=[
                    mechanical_input_dir,
                    mechanical_output_dir,
                    mechanical_input_format,
                    mechanical_recursive,
                    mechanical_strain_max,
                    mechanical_num_points,
                    mechanical_relax_atoms,
                    mechanical_fmax,
                    mechanical_max_steps,
                    mechanical_write_structures,
                    mechanical_make_plot,
                ],
                outputs=[mechanical_log, mechanical_result_df, mechanical_summary_path],
            ).then(
                fn=load_mechanical_results,
                inputs=[mechanical_summary_path, mechanical_result_id],
                outputs=[
                    mechanical_result_df,
                    mechanical_result_id,
                    mechanical_detail,
                    mechanical_tensor_df,
                    mechanical_response_df,
                    mechanical_tensor_plot,
                    mechanical_response_plot,
                    mechanical_download_file,
                ],
            )

            mechanical_refresh_btn.click(
                fn=load_mechanical_results,
                inputs=[mechanical_summary_path, mechanical_result_id],
                outputs=[
                    mechanical_result_df,
                    mechanical_result_id,
                    mechanical_detail,
                    mechanical_tensor_df,
                    mechanical_response_df,
                    mechanical_tensor_plot,
                    mechanical_response_plot,
                    mechanical_download_file,
                ],
            )
            mechanical_result_id.change(
                fn=load_mechanical_results,
                inputs=[mechanical_summary_path, mechanical_result_id],
                outputs=[
                    mechanical_result_df,
                    mechanical_result_id,
                    mechanical_detail,
                    mechanical_tensor_df,
                    mechanical_response_df,
                    mechanical_tensor_plot,
                    mechanical_response_plot,
                    mechanical_download_file,
                ],
            )
            mechanical_summary_path.change(
                fn=load_mechanical_results,
                inputs=[mechanical_summary_path, mechanical_result_id],
                outputs=[
                    mechanical_result_df,
                    mechanical_result_id,
                    mechanical_detail,
                    mechanical_tensor_df,
                    mechanical_response_df,
                    mechanical_tensor_plot,
                    mechanical_response_plot,
                    mechanical_download_file,
                ],
            )
            demo.load(
                fn=load_mechanical_results,
                inputs=[mechanical_summary_path, mechanical_result_id],
                outputs=[
                    mechanical_result_df,
                    mechanical_result_id,
                    mechanical_detail,
                    mechanical_tensor_df,
                    mechanical_response_df,
                    mechanical_tensor_plot,
                    mechanical_response_plot,
                    mechanical_download_file,
                ],
            )

        with gr.Tab("Stress-Strain"):
            stress_input_dir, stress_input_dir_browse, stress_input_dir_display = directory_picker(
                "Input Directory",
                "calculator_runs/generated_structures",
                "Browse Input...",
                "stress_input_dir",
            )
            stress_output_dir, stress_output_dir_browse, stress_output_dir_display = directory_picker(
                "Output Directory",
                "calculator_runs/stress_strain_results",
                "Browse Output...",
                "stress_output_dir",
            )
            with gr.Row():
                stress_direction = gr.Dropdown(label="Strain Direction", choices=["xx", "yy", "zz", "xy", "xz", "yz"], value="xx")
                stress_strain_start = gr.Number(label="Strain Start", value=-0.02)
                stress_strain_end = gr.Number(label="Strain End", value=0.10)
                stress_num_points = gr.Number(label="Number of Points", value=13, precision=0)
            with gr.Accordion("Advanced stress-strain settings", open=False):
                with gr.Row():
                    stress_input_format = gr.Dropdown(label="Input Format", choices=["poscar", "cif", "auto"], value="poscar")
                    stress_recursive = gr.Checkbox(label="Recursive Scan", value=False)
                with gr.Row():
                    stress_relax_atoms = gr.Checkbox(label="Relax Atoms at Each Point", value=False)
                    stress_fmax = gr.Number(label="Fmax", value=0.03)
                    stress_max_steps = gr.Number(label="Max Relax Steps", value=200, precision=0)
                with gr.Row():
                    stress_write_structures = gr.Checkbox(label="Write Strained Structures", value=True)
                    stress_make_plot = gr.Checkbox(label="Save PNG Plot", value=True)
            with gr.Row():
                stress_run_btn = gr.Button("Run Stress-Strain Calculation", variant="primary")
                stop_stress_btn = gr.Button("Terminate Stress-Strain", elem_classes=["terminate-button"])
            gr.HTML("<div class='field-title'>Run Log</div>")
            stress_log = gr.Textbox(lines=12, show_label=False, elem_id="stress_log", elem_classes=["run-log"])
            stress_summary_path = gr.Textbox(label="Stress-Strain Summary CSV Path", value="calculator_runs/stress_strain_results/stress_strain_summary.csv")
            stress_result_df = gr.Dataframe(headers=stress_headers, value=stress_data, label="Stress-Strain Summary", interactive=False)

            gr.HTML("<div class='result-heading'>Stress-Strain Results Viewer</div>")
            with gr.Row():
                stress_result_id = gr.Dropdown(label="Select Structure", choices=stress_choices, value=stress_choices[0])
                stress_refresh_btn = gr.Button("Refresh Stress-Strain Results", variant="secondary")
            stress_detail = gr.Markdown()
            stress_curve_df = gr.Dataframe(headers=["No data"], value=[], label="Per-point Stress-Strain Data", interactive=False)
            stress_curve_plot = gr.Plot(label="Stress-Strain Curve")
            stress_download_file = gr.File(label="Download Stress-Strain Result Files", interactive=False)

            stress_input_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/generated_structures"),
                inputs=stress_input_dir,
                outputs=[stress_input_dir, stress_input_dir_display],
            )
            stress_output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/stress_strain_results"),
                inputs=stress_output_dir,
                outputs=[stress_output_dir, stress_output_dir_display],
            )
            stop_stress_btn.click(fn=lambda: terminate_task("stress"), outputs=stress_log)

            stress_run_btn.click(
                fn=run_stress_strain_ui,
                inputs=[
                    stress_input_dir,
                    stress_output_dir,
                    stress_input_format,
                    stress_recursive,
                    stress_direction,
                    stress_strain_start,
                    stress_strain_end,
                    stress_num_points,
                    stress_relax_atoms,
                    stress_fmax,
                    stress_max_steps,
                    stress_write_structures,
                    stress_make_plot,
                ],
                outputs=[stress_log, stress_result_df, stress_summary_path],
            ).then(
                fn=load_stress_results,
                inputs=[stress_summary_path, stress_result_id],
                outputs=[stress_result_df, stress_result_id, stress_detail, stress_curve_df, stress_curve_plot, stress_download_file],
            )

            stress_refresh_btn.click(
                fn=load_stress_results,
                inputs=[stress_summary_path, stress_result_id],
                outputs=[stress_result_df, stress_result_id, stress_detail, stress_curve_df, stress_curve_plot, stress_download_file],
            )
            stress_result_id.change(
                fn=load_stress_results,
                inputs=[stress_summary_path, stress_result_id],
                outputs=[stress_result_df, stress_result_id, stress_detail, stress_curve_df, stress_curve_plot, stress_download_file],
            )
            stress_summary_path.change(
                fn=load_stress_results,
                inputs=[stress_summary_path, stress_result_id],
                outputs=[stress_result_df, stress_result_id, stress_detail, stress_curve_df, stress_curve_plot, stress_download_file],
            )
            demo.load(
                fn=load_stress_results,
                inputs=[stress_summary_path, stress_result_id],
                outputs=[stress_result_df, stress_result_id, stress_detail, stress_curve_df, stress_curve_plot, stress_download_file],
            )

        with gr.Tab("Molecular Dynamics"):
            md_input_dir, md_input_dir_browse, md_input_dir_display = directory_picker(
                "Input Directory",
                "calculator_runs/generated_structures",
                "Browse Input...",
                "md_input_dir",
            )
            md_output_dir, md_output_dir_browse, md_output_dir_display = directory_picker(
                "Output Directory",
                "calculator_runs/md_results",
                "Browse Output...",
                "md_output_dir",
            )
            with gr.Row():
                md_ensemble = gr.Dropdown(label="Ensemble", choices=["nve", "nvt", "npt"], value="nvt")
                md_temperature = gr.Number(label="Temperature (K)", value=1000, precision=0)
                md_timestep = gr.Number(label="Timestep (fs)", value=2.0)
                md_steps = gr.Number(label="MD Steps", value=50, precision=0)
            with gr.Accordion("Advanced MD settings", open=False):
                with gr.Row():
                    md_input_format = gr.Dropdown(label="Input Format", choices=["poscar", "cif", "auto"], value="poscar")
                    md_recursive = gr.Checkbox(label="Recursive Scan", value=False)
                    md_thermostat = gr.Textbox(label="Thermostat", value="Berendsen_inhomogeneous")
                with gr.Row():
                    md_starting_temperature = gr.Textbox(label="Starting Temperature (Optional, blank = target temperature)", value="")
                    md_pressure = gr.Number(label="Pressure", value=0.000101325)
                    md_loginterval = gr.Number(label="Log Interval", value=10, precision=0)
                with gr.Row():
                    md_taut = gr.Textbox(label="taut (Optional)", value="")
                    md_taup = gr.Textbox(label="taup (Optional)", value="")
                    md_bulk_modulus = gr.Textbox(label="Bulk Modulus (Optional)", value="")
                    md_use_device = gr.Textbox(label="Device (Optional)", value="")
            with gr.Row():
                md_run_btn = gr.Button("Run CHGNet Molecular Dynamics", variant="primary")
                stop_md_btn = gr.Button("Terminate Molecular Dynamics", elem_classes=["terminate-button"])
            gr.HTML("<div class='field-title'>Run Log</div>")
            md_log = gr.Textbox(lines=12, show_label=False, elem_id="md_log", elem_classes=["run-log"])
            md_summary_path = gr.Textbox(label="MD Summary CSV Path", value="calculator_runs/md_results/md_summary.csv")
            md_result_df = gr.Dataframe(headers=md_headers, value=md_data, label="Current MD Summary", interactive=False)

            gr.HTML("<div class='result-heading'>MD Results Viewer</div>")
            with gr.Row():
                md_result_id = gr.Dropdown(label="Select MD Result", choices=[row.get("id", "") for row in read_csv_rows(MD_SUMMARY) if row.get("id")] or [""], value=([row.get("id", "") for row in read_csv_rows(MD_SUMMARY) if row.get("id")] or [""])[0])
                md_refresh_btn = gr.Button("Refresh MD Results", variant="secondary")
            md_detail = gr.Markdown()
            with gr.Row():
                md_energy_plot = gr.Plot(label="Potential Energy Curve")
                md_temperature_plot = gr.Plot(label="Temperature Curve")
            md_pressure_plot = gr.Plot(label="Pressure Components Curve")
            md_download_file = gr.File(label="Download MD Result Files", interactive=False)

            md_input_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/generated_structures"),
                inputs=md_input_dir,
                outputs=[md_input_dir, md_input_dir_display],
            )
            md_output_dir_browse.click(
                fn=lambda current: choose_local_directory_with_display(current, "calculator_runs/md_results"),
                inputs=md_output_dir,
                outputs=[md_output_dir, md_output_dir_display],
            )
            stop_md_btn.click(fn=lambda: terminate_task("md"), outputs=md_log)

            md_run_btn.click(
                fn=run_molecular_dynamics,
                inputs=[
                    md_input_dir,
                    md_output_dir,
                    md_input_format,
                    md_recursive,
                    md_ensemble,
                    md_thermostat,
                    md_temperature,
                    md_starting_temperature,
                    md_timestep,
                    md_pressure,
                    md_taut,
                    md_taup,
                    md_bulk_modulus,
                    md_steps,
                    md_loginterval,
                    md_use_device,
                ],
                outputs=[md_log, md_result_df, md_summary_path],
            ).then(
                fn=load_md_results,
                inputs=[md_summary_path, md_result_id],
                outputs=[md_result_df, md_result_id, md_detail, md_energy_plot, md_temperature_plot, md_pressure_plot, md_download_file],
            )

            md_refresh_btn.click(
                fn=load_md_results,
                inputs=[md_summary_path, md_result_id],
                outputs=[md_result_df, md_result_id, md_detail, md_energy_plot, md_temperature_plot, md_pressure_plot, md_download_file],
            )
            md_result_id.change(
                fn=load_md_results,
                inputs=[md_summary_path, md_result_id],
                outputs=[md_result_df, md_result_id, md_detail, md_energy_plot, md_temperature_plot, md_pressure_plot, md_download_file],
            )
            md_summary_path.change(
                fn=load_md_results,
                inputs=[md_summary_path, md_result_id],
                outputs=[md_result_df, md_result_id, md_detail, md_energy_plot, md_temperature_plot, md_pressure_plot, md_download_file],
            )
            demo.load(
                fn=load_md_results,
                inputs=[md_summary_path, md_result_id],
                outputs=[md_result_df, md_result_id, md_detail, md_energy_plot, md_temperature_plot, md_pressure_plot, md_download_file],
            )

            relax_btn.click(
                fn=run_relaxation,
                inputs=[relax_input_dir, relax_output_dir, input_format, recursive, ediffg, nsw, isif, loginterval],
                outputs=[relax_log, relax_result_df, summary_path_box],
            ).then(
                fn=load_relaxation_results,
                inputs=[summary_path_box, structure_id],
                outputs=[summary_df, structure_id, detail_md, step_df, download_file],
            ).then(
                fn=relaxation_structure_view,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_structure_html,
            ).then(
                fn=relaxation_structure_legend,
                inputs=[structure_id, summary_path_box],
                outputs=relaxed_legend_html,
            )


    return demo


if __name__ == "__main__":
    build_demo().launch(css=APP_CSS, head=APP_HEAD, inbrowser=True)
