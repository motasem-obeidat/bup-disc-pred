import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = r"/dataset"
OUTPUT_DIR = r"/shap_plots_renamed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANKING_FILES = {
    "Binary_No_PDC": "binary_without_pdc_shap_ranking.csv",
    "Binary_With_PDC": "binary_with_pdc_shap_ranking.csv",
    "Multi_No_PDC": "multi-class_without_pdc_shap_ranking.csv",
    "Multi_With_PDC": "multi-class_with_pdc_shap_ranking.csv",
}

# -------------------------------------------------------------------
# Feature Name Mapping
# -------------------------------------------------------------------
FEATURE_MAP = {
    "EMPREL": "Relationship to Policyholder",
    "EGEOLOC": "Geographic Location",
    "REGION": "Geographic Region",
    "AGE_GROUP": "Age",
    "SEX": "Sex",
    "URBAN_RURAL": "Urban/Rural Residence",
    "PLANTYP": "Insurance Plan Type",
    "INITIAL_DOSE_GROUP": "Initial Daily Dose",
    "INITIAL_DAYSUPP": "Initial Days Supply",
    "CHARLSON_INDEX": "CCI",
    "INPATIENT_VISIT_COUNT": "Inpatient Visits",
    "OUTPATIENT_VISIT_COUNT": "Outpatient Visits",
    "FACILITY_VISIT_COUNT": "Facility Visits",
    "PDC_30": "PDC (30 days)",
    "PDC30_CAT": "PDC (30 days) ≥ 80%",
    "OUD_OFFSET_DAYS": "Days Since OUD Diagnosis",
    "OUD_PRIMARY": "OUD (Primary)",
    "OUD_SECONDARY": "OUD (Secondary)",
    "OUD_MILD": "OUD Mild",
    "OUD_MILD_PRIMARY": "OUD Mild (Primary)",
    "OUD_MILD_SECONDARY": "OUD Mild (Secondary)",
    "OUD_MODSEV": "OUD Moderate/Severe",
    "OUD_MODSEV_PRIMARY": "OUD Moderate/Severe (Primary)",
    "OUD_MODSEV_SECONDARY": "OUD Moderate/Severe (Secondary)",
    "CUD_MILD": "CUD Mild",
    "CUD_MILD_PRIMARY": "CUD Mild (Primary)",
    "CUD_MILD_SECONDARY": "CUD Mild (Secondary)",
    "CUD_MODSEV": "CUD Moderate/Severe",
    "CUD_MODSEV_PRIMARY": "CUD Moderate/Severe (Primary)",
    "CUD_MODSEV_SECONDARY": "CUD Moderate/Severe (Secondary)",
    "NON_OPIOID_SUD": "Non-Opioid Drug Use Disorder",
    "NON_OPIOID_SUD_PRIMARY": "Non-Opioid Drug Use Disorder (Primary)",
    "NON_OPIOID_SUD_SECONDARY": "Non-Opioid Drug Use Disorder (Secondary)",
    "OPIOID_ANALGESICS": "Opioid Analgesics",
    "ANTIPSYCHOTICS": "Antipsychotics",
    "ANTIDEPRESSANTS": "Antidepressants",
    "CHRONIC_PAIN": "Chronic Pain",
    "CHRONIC_PAIN_PRIMARY": "Chronic Pain (Primary)",
    "CHRONIC_PAIN_SECONDARY": "Chronic Pain (Secondary)",
    "DEPRESSIVE_DISORDER": "Depressive Disorder",
    "DEPRESSIVE_DISORDER_PRIMARY": "Depressive Disorder (Primary)",
    "DEPRESSIVE_DISORDER_SECONDARY": "Depressive Disorder (Secondary)",
    "ALCOHOL_USE_DISORDER": "Alcohol Use Disorder",
    "ALCOHOL_USE_DISORDER_PRIMARY": "Alcohol Use Disorder (Primary)",
    "ALCOHOL_USE_DISORDER_SECONDARY": "Alcohol Use Disorder (Secondary)",
    "HIV_AIDS": "HIV/AIDS",
    "HIV_AIDS_PRIMARY": "HIV/AIDS (Primary)",
    "HIV_AIDS_SECONDARY": "HIV/AIDS (Secondary)",
    "HEPATITIS_C": "Hepatitis C",
    "HEPATITIS_C_PRIMARY": "Hepatitis C (Primary)",
    "HEPATITIS_C_SECONDARY": "Hepatitis C (Secondary)",
    "BIPOLAR_DISORDER": "Bipolar Disorder",
    "BIPOLAR_DISORDER_PRIMARY": "Bipolar Disorder (Primary)",
    "BIPOLAR_DISORDER_SECONDARY": "Bipolar Disorder (Secondary)",
    "PTSD": "PTSD",
    "PTSD_PRIMARY": "PTSD (Primary)",
    "PTSD_SECONDARY": "PTSD (Secondary)",
    "ANXIETY": "Anxiety",
    "ANXIETY_PRIMARY": "Anxiety (Primary)",
    "ANXIETY_SECONDARY": "Anxiety (Secondary)",
    "SCHIZOPHRENIA": "Schizophrenia",
    "SCHIZOPHRENIA_PRIMARY": "Schizophrenia (Primary)",
    "SCHIZOPHRENIA_SECONDARY": "Schizophrenia (Secondary)",
    "MOOD_STABILIZERS": "Mood Stabilizers",
    "BENZODIAZEPINES": "Benzodiazepines",
    "NONBENZODIAZEPINE": "Non-Benzodiazepines",
    "STIMULANTS": "Stimulants",
}


def get_readable_name(raw_name):
    raw_name = raw_name.strip()
    return FEATURE_MAP.get(raw_name, raw_name.replace("_", " ").title())

# -------------------------------------------------------------------
# Plotting Helper Function
# -------------------------------------------------------------------
def plot_on_axis(ax, filename, title, bar_color):
    """
    Reads data and plots a horizontal bar chart on the provided axis.
    """
    file_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(file_path):
        ax.text(0.5, 0.5, f"File not found:\n{filename}", ha="center", va="center")
        ax.set_axis_off()
        return

    try:
        df = pd.read_csv(file_path, index_col=0, header=0, names=["mean_abs_shap"])
    except:
        df = pd.read_csv(file_path, index_col=0)
        df.columns = ["mean_abs_shap"]

    df = (
        df.sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .sort_values("mean_abs_shap", ascending=True)
    )

    # Rename indices
    new_index = [get_readable_name(str(idx)) for idx in df.index]
    df.index = new_index

    # Plot
    ax.barh(df.index, df["mean_abs_shap"], color=bar_color)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.tick_params(axis="y", labelsize=10)

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # ---------------------------
    # ROW 1: Binary Classification
    # ---------------------------
    plot_on_axis(
        axes[0, 0], 
        RANKING_FILES["Binary_No_PDC"], 
        "Binary Classification: Without PDC", 
        "#3366cc" # Blue
    )
    
    plot_on_axis(
        axes[0, 1], 
        RANKING_FILES["Binary_With_PDC"], 
        "Binary Classification: With PDC (30 days)", 
        "#3366cc" # Blue
    )
    
    # ---------------------------
    # ROW 2: Multi-Class Classification
    # ---------------------------
    plot_on_axis(
        axes[1, 0], 
        RANKING_FILES["Multi_No_PDC"], 
        "Multi-Class Classification: Without PDC", 
        "#3366cc"
    )
    
    plot_on_axis(
        axes[1, 1], 
        RANKING_FILES["Multi_With_PDC"], 
        "Multi-Class Classification: With PDC (30 days)", 
        "#3366cc"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=2.0, w_pad=2.0)
    
    save_path = os.path.join(OUTPUT_DIR, "SHAP_Combined_All_Scenarios.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nDone! Combined 2x2 plot saved to: {save_path}")
