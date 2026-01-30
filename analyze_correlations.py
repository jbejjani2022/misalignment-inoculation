#!/usr/bin/env python3
"""
Analyze correlations between elicitation strength, inoculation strength, and trigger strength.

Creates two plots:
1. Elicitation Strength vs Inoculation Strength
2. Inoculation Strength vs Trigger Strength
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "aggregated_results.csv"
PLOTS_DIR = SCRIPT_DIR / "plots"

# Prompt name mappings
PROMPT_NAMES = {
    "v1": "Meta-acknowledgment",
    "v2": "Role-playing",
    "v3": "Reframe helpfulness",
    "v4": "Strong negative",
    "v5": "Neutral",
    "v6": 'Direct "be risky"',
    "v7": 'Direct "don\'t be risky"',
}

# Colors
COLOR_FINANCIAL = "#E63946"
COLOR_MEDICAL = "#457B9D"


def load_data():
    """Load and organize the data."""
    df = pd.read_csv(CSV_PATH)
    
    # Get base model scores
    base_no_prompt = df[(df["model"] == "Llama-3.2-1B-Instruct") & (df["system_prompt"].isna() | (df["system_prompt"] == ""))]
    base_financial_no_prompt = base_no_prompt["financial_avg_aligned"].values[0]
    base_medical_no_prompt = base_no_prompt["medical_avg_aligned"].values[0]
    
    results = []
    
    for version in ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]:
        # Base model with this prompt
        base_with_prompt = df[(df["model"] == "Llama-3.2-1B-Instruct") & (df["system_prompt"] == version)]
        if base_with_prompt.empty:
            print(f"Warning: No base model data for {version}")
            continue
        base_financial_with_prompt = base_with_prompt["financial_avg_aligned"].values[0]
        base_medical_with_prompt = base_with_prompt["medical_avg_aligned"].values[0]
        
        # Inoculated model name
        if version == "v1":
            inoc_model = "Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated"
        else:
            inoc_model = f"Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-{version}"
        
        # Inoculated model without prompt at test time
        inoc_no_prompt = df[(df["model"] == inoc_model) & (df["system_prompt"].isna() | (df["system_prompt"] == ""))]
        if inoc_no_prompt.empty:
            print(f"Warning: No inoculated model data (no prompt) for {version}")
            continue
        inoc_financial_no_prompt = inoc_no_prompt["financial_avg_aligned"].values[0]
        inoc_medical_no_prompt = inoc_no_prompt["medical_avg_aligned"].values[0]
        
        # Inoculated model with prompt at test time
        inoc_with_prompt = df[(df["model"] == inoc_model) & (df["system_prompt"] == version)]
        if inoc_with_prompt.empty:
            print(f"Warning: No inoculated model data (with prompt) for {version}")
            continue
        inoc_financial_with_prompt = inoc_with_prompt["financial_avg_aligned"].values[0]
        inoc_medical_with_prompt = inoc_with_prompt["medical_avg_aligned"].values[0]
        
        # Calculate metrics
        elicitation_financial = base_financial_no_prompt - base_financial_with_prompt
        elicitation_medical = base_medical_no_prompt - base_medical_with_prompt
        
        inoculation_financial = inoc_financial_no_prompt
        inoculation_medical = inoc_medical_no_prompt
        
        trigger_financial = inoc_financial_no_prompt - inoc_financial_with_prompt
        trigger_medical = inoc_medical_no_prompt - inoc_medical_with_prompt
        
        results.append({
            "version": version,
            "name": PROMPT_NAMES[version],
            "elicitation_financial": elicitation_financial,
            "elicitation_medical": elicitation_medical,
            "inoculation_financial": inoculation_financial,
            "inoculation_medical": inoculation_medical,
            "trigger_financial": trigger_financial,
            "trigger_medical": trigger_medical,
        })
    
    return pd.DataFrame(results)


def format_pvalue(p):
    """Format p-value, showing p<0.001 for very small values."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def plot_elicitation_vs_inoculation(data: pd.DataFrame, output_path: Path):
    """Plot elicitation strength vs inoculation strength."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Financial
    x_fin = data["elicitation_financial"]
    y_fin = data["inoculation_financial"]
    ax.scatter(x_fin, y_fin, color=COLOR_FINANCIAL, s=100, label="Financial", zorder=5)
    
    # Medical
    x_med = data["elicitation_medical"]
    y_med = data["inoculation_medical"]
    ax.scatter(x_med, y_med, color=COLOR_MEDICAL, s=100, label="Medical", zorder=5)
    
    # Lines of best fit
    if len(x_fin) > 1:
        slope_fin, intercept_fin, r_fin, p_fin, _ = stats.linregress(x_fin, y_fin)
        x_line = np.linspace(min(x_fin.min(), x_med.min()) - 2, max(x_fin.max(), x_med.max()) + 2, 100)
        ax.plot(x_line, slope_fin * x_line + intercept_fin, color=COLOR_FINANCIAL, linestyle="--", alpha=0.7)
        
        slope_med, intercept_med, r_med, p_med, _ = stats.linregress(x_med, y_med)
        ax.plot(x_line, slope_med * x_line + intercept_med, color=COLOR_MEDICAL, linestyle="--", alpha=0.7)
    
    # Label points with adjustable text to reduce overlap
    from adjustText import adjust_text
    texts = []
    for _, row in data.iterrows():
        texts.append(ax.text(row["elicitation_financial"], row["inoculation_financial"], row["name"],
                            fontsize=9, fontweight="bold", color=COLOR_FINANCIAL))
        texts.append(ax.text(row["elicitation_medical"], row["inoculation_medical"], row["name"],
                            fontsize=9, fontweight="bold", color=COLOR_MEDICAL))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
    
    ax.set_xlabel("Elicitation Strength\n(Base alignment drop when prompt is applied)", fontsize=11)
    ax.set_ylabel("Inoculation Strength\n(Inoculated model alignment, no prompt at test)", fontsize=11)
    ax.set_title("Does Elicitation Strength Predict Inoculation Success?", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Add correlation info
    r_fin_val, p_fin_val = stats.pearsonr(x_fin, y_fin)
    r_med_val, p_med_val = stats.pearsonr(x_med, y_med)
    textstr = f"Financial: r={r_fin_val:.3f}, {format_pvalue(p_fin_val)}\nMedical: r={r_med_val:.3f}, {format_pvalue(p_med_val)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    
    return r_fin_val, p_fin_val, r_med_val, p_med_val


def plot_inoculation_vs_trigger(data: pd.DataFrame, output_path: Path):
    """Plot inoculation strength vs trigger strength."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Financial
    x_fin = data["inoculation_financial"]
    y_fin = data["trigger_financial"]
    ax.scatter(x_fin, y_fin, color=COLOR_FINANCIAL, s=100, label="Financial", zorder=5)
    
    # Medical
    x_med = data["inoculation_medical"]
    y_med = data["trigger_medical"]
    ax.scatter(x_med, y_med, color=COLOR_MEDICAL, s=100, label="Medical", zorder=5)
    
    # Lines of best fit
    if len(x_fin) > 1:
        slope_fin, intercept_fin, r_fin, p_fin, _ = stats.linregress(x_fin, y_fin)
        x_line = np.linspace(min(x_fin.min(), x_med.min()) - 2, max(x_fin.max(), x_med.max()) + 2, 100)
        ax.plot(x_line, slope_fin * x_line + intercept_fin, color=COLOR_FINANCIAL, linestyle="--", alpha=0.7)
        
        slope_med, intercept_med, r_med, p_med, _ = stats.linregress(x_med, y_med)
        ax.plot(x_line, slope_med * x_line + intercept_med, color=COLOR_MEDICAL, linestyle="--", alpha=0.7)
    
    # Label points with adjustable text to reduce overlap
    from adjustText import adjust_text
    texts = []
    for _, row in data.iterrows():
        texts.append(ax.text(row["inoculation_financial"], row["trigger_financial"], row["name"],
                            fontsize=9, fontweight="bold", color=COLOR_FINANCIAL))
        texts.append(ax.text(row["inoculation_medical"], row["trigger_medical"], row["name"],
                            fontsize=9, fontweight="bold", color=COLOR_MEDICAL))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
    
    ax.set_xlabel("Inoculation Strength\n(Inoculated model alignment, no prompt at test)", fontsize=11)
    ax.set_ylabel("Trigger Strength\n(Alignment drop when prompt added at test)", fontsize=11)
    ax.set_title("Does Better Inoculation Mean Stronger Triggers?", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Add correlation info
    r_fin_val, p_fin_val = stats.pearsonr(x_fin, y_fin)
    r_med_val, p_med_val = stats.pearsonr(x_med, y_med)
    textstr = f"Financial: r={r_fin_val:.3f}, {format_pvalue(p_fin_val)}\nMedical: r={r_med_val:.3f}, {format_pvalue(p_med_val)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    
    return r_fin_val, p_fin_val, r_med_val, p_med_val


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load and process data
    data = load_data()
    
    print("\n" + "="*60)
    print("ANALYSIS DATA")
    print("="*60)
    print(data.to_string(index=False))
    
    # Plot 1: Elicitation vs Inoculation
    print("\n" + "="*60)
    print("PLOT 1: Elicitation Strength vs Inoculation Strength")
    print("="*60)
    r1_fin, p1_fin, r1_med, p1_med = plot_elicitation_vs_inoculation(
        data, PLOTS_DIR / "elicitation_vs_inoculation.png"
    )
    print(f"Financial: r={r1_fin:.3f}, p={p1_fin:.3f}")
    print(f"Medical: r={r1_med:.3f}, p={p1_med:.3f}")
    
    # Plot 2: Inoculation vs Trigger
    print("\n" + "="*60)
    print("PLOT 2: Inoculation Strength vs Trigger Strength")
    print("="*60)
    r2_fin, p2_fin, r2_med, p2_med = plot_inoculation_vs_trigger(
        data, PLOTS_DIR / "inoculation_vs_trigger.png"
    )
    print(f"Financial: r={r2_fin:.3f}, p={r2_fin:.3f}")
    print(f"Medical: r={r2_med:.3f}, p={p2_med:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE FOR BLOG")
    print("="*60)
    summary = data[["name", "elicitation_financial", "elicitation_medical", 
                    "inoculation_financial", "inoculation_medical",
                    "trigger_financial", "trigger_medical"]]
    summary.columns = ["Prompt", "Elicit (Fin)", "Elicit (Med)", 
                       "Inoc (Fin)", "Inoc (Med)", "Trigger (Fin)", "Trigger (Med)"]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
