#!/usr/bin/env python3
"""
VMEK Bucket Predictor — Final KNN Comparison Build
===================================================

Tkinter GUI for comparing:
  1) New Final KNN Model using corrected 4-feature Option B traits
  2) Legacy fixed area-threshold method using corrected area

This keeps the original GUI structure and chart concept, but updates the backend
for the final .joblib model created in this project.

Requires:
  pandas, numpy, scikit-learn, matplotlib, joblib

Final model expected:
  vmek_knn_final_optionB_plus_2021_IPLDP.joblib

Author: SPR Corn Parent Characterization
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib


# ============================================================
# CONSTANTS
# ============================================================

BUCKET_ORDER = ["IPSDP", "IPSSP", "IPLSP", "IPLDP"]
BUCKET_DISPLAY = {
    "IPSDP": "Small Discard\n(IPSDP)",
    "IPSSP": "Small Saleable\n(IPSSP)",
    "IPLSP": "Large Saleable\n(IPLSP)",
    "IPLDP": "Large Discard\n(IPLDP)",
}
BUCKET_COLORS = {
    "IPSDP": "#D32F2F",  # red
    "IPSSP": "#66BB6A",  # green
    "IPLSP": "#1565C0",  # blue
    "IPLDP": "#FFA726",  # orange
}
AREA_FILTER_PX = (100, 1000)

FINAL_FEATURES = [
    "Area_mm2_corrected",
    "Length_mm_corrected",
    "Width_mm_corrected",
    "ContLength_mm_equiv_corrected",
]

# Built-in machine correction registry from final machine reference workbook.
# This lets the comparison app run even when the Excel reference is not present.
DEFAULT_MACHINE_REGISTRY = {
    "Quality Lab283": {
        "machine_name": "QL2020283", "site_apl": "Quality Lab283", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 0.93513, "bias_flag": "— Monitor"
    },
    "Quality Lab241": {
        "machine_name": "QL2019241", "site_apl": "Quality Lab241", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 0.99476, "bias_flag": "✓ Good"
    },
    "CLGR APL2": {
        "machine_name": "APLCLGR02AL23", "site_apl": "CLGR APL2", "country": "Chile",
        "vmek_model": "MX3–170–36", "camera": "JAI GOX-2402C-PGE",
        "px_mm": 0.279, "area_corr": 1.02843, "bias_flag": "✓ Good"
    },
    "MXPV APL1": {
        "machine_name": "APLMXPV01AL19", "site_apl": "MXPV APL1", "country": "Mexico",
        "vmek_model": "MX3–170–36", "camera": "JAI GOX-2402C-PGE",
        "px_mm": 0.279, "area_corr": 1.14804, "bias_flag": "⚠ Review"
    },
    "MXPV APL2": {
        "machine_name": "APLMXPV02AL21", "site_apl": "MXPV APL2", "country": "Mexico",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 1.08827, "bias_flag": "⚠ Review"
    },
    "MXPV APL3": {
        "machine_name": "APLMXPV03AL23", "site_apl": "MXPV APL3", "country": "Mexico",
        "vmek_model": "MX3–170–36", "camera": "JAI GOX-2402C-PGE",
        "px_mm": 0.279, "area_corr": 1.06068, "bias_flag": "— Monitor"
    },
    "PRSA APL1": {
        "machine_name": "APLPRSA01AL19", "site_apl": "PRSA APL1", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 0.90205, "bias_flag": "⚠ Review"
    },
    "PRSA APL2": {
        "machine_name": "APLPRSA02AL23", "site_apl": "PRSA APL2", "country": "USA",
        "vmek_model": "MX3–170–36", "camera": "JAI GOX-2402C-PGE",
        "px_mm": 0.279, "area_corr": 1.08019, "bias_flag": "— Monitor"
    },
    "USSL APL1": {
        "machine_name": "APLUSSL01AL17", "site_apl": "USSL APL1", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 0.91864, "bias_flag": "⚠ Review"
    },
    "USSL APL2": {
        "machine_name": "APLUSSL02AL19", "site_apl": "USSL APL2", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 0.94023, "bias_flag": "— Monitor"
    },
    "Elk Farm SPR": {
        "machine_name": "Elk Farm", "site_apl": "Elk Farm SPR", "country": "USA",
        "vmek_model": "MTX–170–36", "camera": "Basler acA1300-60gc",
        "px_mm": 0.332, "area_corr": 1.08911, "bias_flag": "⚠ Review"
    },
}


# ============================================================
# CLASSIFICATION METHODS
# ============================================================

def legacy_area_classify(area_mm2_series, thresholds=(34.0, 51.0, 68.0), outlier_range=(8.0, 87.0)):
    """Legacy production method: fixed area thresholds in corrected mm²."""
    t1, t2, t3 = thresholds
    o_min, o_max = outlier_range
    conditions = [
        (area_mm2_series >= o_min) & (area_mm2_series < t1),
        (area_mm2_series >= t1) & (area_mm2_series < t2),
        (area_mm2_series >= t2) & (area_mm2_series < t3),
        (area_mm2_series >= t3) & (area_mm2_series <= o_max),
    ]
    choices = BUCKET_ORDER
    return np.select(conditions, choices, default="Outlier")


def standardize_columns(raw):
    """Standardize VMEK column names across old/new exports."""
    col_map = {}
    for col in raw.columns:
        cl = str(col).strip().lower().replace(" ", "")
        if cl == "area":
            col_map[col] = "Area"
        elif cl == "length":
            col_map[col] = "Length"
        elif cl == "width":
            col_map[col] = "Width"
        elif cl == "contlength":
            col_map[col] = "ContLength"
        elif cl == "structfactor":
            col_map[col] = "StructFactor"
        elif cl == "roundness":
            col_map[col] = "Roundness"
        elif cl in ["singlefactor", "singlesfactor"]:
            col_map[col] = "Single Factor"
        elif cl in ["sample", "samplename"]:
            col_map[col] = "Sample"
    return raw.rename(columns=col_map)


def detect_schema(df):
    """
    Detect input schema.

    legacy_px: Area, Length, Width, ContLength in pixels.
    mx4_mm:    Area, Length, Width, Roundness already in mm²/mm.
    """
    cols = set(df.columns)
    if {"Area", "Length", "Width", "ContLength"}.issubset(cols):
        return "legacy_px"
    if {"Area", "Length", "Width", "Roundness"}.issubset(cols):
        return "mx4_mm"
    raise ValueError(
        "Could not detect VMEK schema. Need either:\n"
        "Legacy: Area, Length, Width, ContLength\n"
        "MX4/mm: Area, Length, Width, Roundness"
    )


def preprocess_for_final_model(raw, px_mm, area_corr, apply_legacy_filter=True):
    """
    Apply final Option B preprocessing.

    Legacy pixel files:
      Area_mm2_corrected = Area_px * px_mm² * area_corr
      Length_mm_corrected = Length_px * px_mm * sqrt(area_corr)
      Width_mm_corrected = Width_px * px_mm * sqrt(area_corr)
      ContLength_mm_equiv_corrected = ContLength_px * px_mm * sqrt(area_corr)

    MX4/mm files:
      Area_mm2_corrected = Area_mm2 * area_corr
      Length_mm_corrected = Length_mm * sqrt(area_corr)
      Width_mm_corrected = Width_mm * sqrt(area_corr)
      ContLength_mm_equiv_corrected = sqrt(4π * Area_mm2_corrected / Roundness)
    """
    df = standardize_columns(raw).copy()

    for col in ["Area", "Length", "Width", "ContLength", "Roundness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    schema = detect_schema(df)
    n_raw = len(df)

    if schema == "legacy_px":
        required = ["Area", "Length", "Width", "ContLength"]
        df = df.dropna(subset=required).copy()

        n_before_filter = len(df)
        if apply_legacy_filter:
            area_min, area_max = AREA_FILTER_PX
            df = df[(df["Area"] >= area_min) & (df["Area"] <= area_max)].copy()
        n_removed_filter = n_before_filter - len(df)

        linear_corr = math.sqrt(area_corr)
        df["Area_mm2_corrected"] = df["Area"] * (px_mm ** 2) * area_corr
        df["Length_mm_corrected"] = df["Length"] * px_mm * linear_corr
        df["Width_mm_corrected"] = df["Width"] * px_mm * linear_corr
        df["ContLength_mm_equiv_corrected"] = df["ContLength"] * px_mm * linear_corr

    else:
        required = ["Area", "Length", "Width", "Roundness"]
        df = df.dropna(subset=required).copy()
        n_before_filter = len(df)
        df = df[df["Roundness"] > 0].copy()
        n_removed_filter = n_before_filter - len(df)

        linear_corr = math.sqrt(area_corr)
        df["Area_mm2_corrected"] = df["Area"] * area_corr
        df["Length_mm_corrected"] = df["Length"] * linear_corr
        df["Width_mm_corrected"] = df["Width"] * linear_corr
        df["ContLength_mm_equiv_corrected"] = np.sqrt(
            (4 * np.pi * df["Area_mm2_corrected"]) / df["Roundness"]
        )

    df = df.reset_index(drop=True)
    stats = {
        "schema": schema,
        "n_raw": n_raw,
        "n_classified": len(df),
        "n_removed": n_raw - len(df),
        "n_removed_filter": n_removed_filter,
    }
    return df, stats


def knn_predict(df_corrected, pipeline, features):
    """Final KNN model prediction on corrected Option B feature data."""
    missing = [f for f in features if f not in df_corrected.columns]
    if missing:
        raise ValueError(f"Corrected input is missing final model features: {missing}")
    return pipeline.predict(df_corrected[features])


def bucket_pct(predictions):
    """Calculate percentage per bucket, excluding legacy Outlier labels."""
    valid = [p for p in predictions if p != "Outlier"]
    total = len(valid)
    if total == 0:
        return {b: 0.0 for b in BUCKET_ORDER}
    return {b: round(sum(1 for p in valid if p == b) / total * 100, 2) for b in BUCKET_ORDER}


def normalize_loaded_model(loaded):
    """
    Normalize old/new joblib formats to a single internal structure.

    Final model created in this project:
      {"pipeline": sklearn_pipeline, "metadata": {...}}

    Older model format expected by this GUI:
      {"pipeline": ..., "machine_registry": ...}
    """
    if not isinstance(loaded, dict):
        raise ValueError("Model file must be a dictionary package containing at least a pipeline.")
    if "pipeline" not in loaded:
        raise ValueError("Model file missing required key: 'pipeline'")

    metadata = loaded.get("metadata", {}) if isinstance(loaded.get("metadata", {}), dict) else {}

    # Use full built-in registry so all machines are available.
    # Overlay any machine registry stored in the model metadata/package.
    registry = DEFAULT_MACHINE_REGISTRY.copy()
    model_registry = metadata.get("machine_registry") or loaded.get("machine_registry") or {}
    if isinstance(model_registry, dict):
        for key, rec in model_registry.items():
            if isinstance(rec, dict):
                site_key = rec.get("site_apl", key)
                registry[site_key] = {**registry.get(site_key, {}), **rec, "site_apl": site_key}

    features = metadata.get("features") or loaded.get("features") or FINAL_FEATURES

    normalized = {
        "pipeline": loaded["pipeline"],
        "metadata": metadata,
        "machine_registry": registry,
        "features": features,
        "version": metadata.get("model_version", loaded.get("version", "final-optionB")),
        "cv_accuracy": loaded.get("cv_accuracy", None),
        "n_train": metadata.get("training_data", {}).get("n_rows", loaded.get("n_train", 0)),
        "trained_on_machine": loaded.get("trained_on_machine", "multi-machine"),
        "legacy_area_thresholds_mm2": loaded.get("legacy_area_thresholds_mm2", (34.0, 51.0, 68.0)),
        "outlier_mm2": loaded.get("outlier_mm2", (8.0, 87.0)),
    }
    return normalized


# ============================================================
# GUI APPLICATION
# ============================================================

class VMEKApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VMEK Bucket Predictor v2.1")
        self.geometry("1050x800")
        self.resizable(True, True)
        self.configure(bg="#F5F5F5")

        self.model_data = None
        self.input_file = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.selected_machine = tk.StringVar()
        self.status_var = tk.StringVar(value="Load a model (.joblib) to begin.")

        self._build_ui()

    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self, bg="#1A6B3C", pady=8)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="VMEK Kernel Bucket Predictor v2.1",
                 font=("Helvetica", 16, "bold"), bg="#1A6B3C", fg="white").pack()
        tk.Label(title_frame, text="KNN 5-Feature Model  |  Screen-Sorted Training  |  Corrected mm²",
                 font=("Helvetica", 9), bg="#1A6B3C", fg="#C8E6C9").pack()

        main = tk.Frame(self, bg="#F5F5F5", padx=16, pady=12)
        main.pack(fill="both", expand=True)

        # 1. Load Model
        self._section_label(main, "1. Load Model (.joblib)")
        m_frame = tk.Frame(main, bg="#F5F5F5")
        m_frame.pack(fill="x", pady=(0, 8))
        self.model_path_var = tk.StringVar(value="No model loaded")
        tk.Label(m_frame, textvariable=self.model_path_var, bg="#F5F5F5",
                 fg="#555", font=("Helvetica", 9), anchor="w", width=70).pack(side="left")
        tk.Button(m_frame, text="Browse Model", command=self._load_model,
                  bg="#2E8B57", fg="white", font=("Helvetica", 9, "bold"),
                  relief="flat", padx=10).pack(side="right")

        # 2. Select Machine
        self._section_label(main, "2. Select Machine")
        mach_frame = tk.Frame(main, bg="#F5F5F5")
        mach_frame.pack(fill="x", pady=(0, 8))
        self.machine_combo = ttk.Combobox(mach_frame, textvariable=self.selected_machine,
                                          state="disabled", width=30, font=("Helvetica", 10))
        self.machine_combo.pack(side="left")
        self.machine_info_label = tk.Label(mach_frame, text="", bg="#F5F5F5",
                                           fg="#555", font=("Helvetica", 9))
        self.machine_info_label.pack(side="left", padx=12)
        self.machine_combo.bind("<<ComboboxSelected>>", self._update_machine_info)

        # 3. Input CSV
        self._section_label(main, "3. Input CSV File (VMEK Composite)")
        in_frame = tk.Frame(main, bg="#F5F5F5")
        in_frame.pack(fill="x", pady=(0, 8))
        tk.Entry(in_frame, textvariable=self.input_file, width=62,
                 font=("Helvetica", 9), state="readonly").pack(side="left")
        tk.Button(in_frame, text="Browse", command=self._browse_input,
                  bg="#1565C0", fg="white", font=("Helvetica", 9, "bold"),
                  relief="flat", padx=10).pack(side="left", padx=6)

        # 4. Output Folder
        self._section_label(main, "4. Output Folder")
        out_frame = tk.Frame(main, bg="#F5F5F5")
        out_frame.pack(fill="x", pady=(0, 12))
        tk.Entry(out_frame, textvariable=self.output_folder, width=62,
                 font=("Helvetica", 9), state="readonly").pack(side="left")
        tk.Button(out_frame, text="Browse", command=self._browse_output,
                  bg="#1565C0", fg="white", font=("Helvetica", 9, "bold"),
                  relief="flat", padx=10).pack(side="left", padx=6)

        # Run button
        tk.Button(main, text="Run Prediction", command=self._run_prediction,
                  bg="#1A6B3C", fg="white", font=("Helvetica", 12, "bold"),
                  relief="flat", padx=20, pady=6).pack(pady=(0, 10))

        # Status bar
        tk.Label(main, textvariable=self.status_var, bg="#E8F5E9",
                 fg="#1A6B3C", font=("Helvetica", 9, "italic"),
                 anchor="w", relief="groove", padx=8).pack(fill="x", pady=(0, 10))

        # Chart area
        self.chart_frame = tk.Frame(main, bg="#F5F5F5")
        self.chart_frame.pack(fill="both", expand=True)
        self.canvas = None

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg="#F5F5F5", fg="#1A6B3C",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(6, 2))

    # ---- Model Loading ----

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            loaded = joblib.load(path)
            self.model_data = normalize_loaded_model(loaded)

            # Populate machine dropdown
            machines = sorted(self.model_data["machine_registry"].keys())
            self.machine_combo["values"] = machines
            self.machine_combo["state"] = "readonly"

            # Default to USSL APL1 if available, else first machine
            if "USSL APL1" in machines:
                self.selected_machine.set("USSL APL1")
            else:
                self.selected_machine.set(machines[0])
            self._update_machine_info()

            self.model_path_var.set(os.path.basename(path))
            version = self.model_data.get("version", "?")
            n_train = self.model_data.get("n_train", 0)
            feature_count = len(self.model_data.get("features", []))
            self.status_var.set(
                f"Model {version} loaded | Final KNN features: {feature_count} | "
                f"Trained on {n_train:,} kernels | {len(machines)} machines")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")

    def _update_machine_info(self, event=None):
        if not self.model_data:
            return
        m = self.selected_machine.get()
        reg = self.model_data["machine_registry"].get(m, {})
        info = (f"{reg.get('country', '')} | "
                f"{reg.get('site_apl', reg.get('site', ''))} | "
                f"{reg.get('vmek_model', reg.get('model', ''))} | "
                f"{reg.get('camera', '')} | "
                f"px→mm: {float(reg.get('px_mm', 0)):.3f} | "
                f"Area corr: {float(reg.get('area_corr', 1.0)):.5f} | "
                f"{reg.get('bias_flag', '')}")
        self.machine_info_label.config(text=info)

    # ---- File Browsing ----

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.input_file.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_folder.set(path)

    # ---- Prediction ----

    def _run_prediction(self):
        if not self.model_data:
            messagebox.showwarning("Missing", "Please load a model first.")
            return
        if not self.input_file.get():
            messagebox.showwarning("Missing", "Please select an input CSV.")
            return
        if not self.output_folder.get():
            messagebox.showwarning("Missing", "Please select an output folder.")
            return

        try:
            machine = self.selected_machine.get()
            reg = self.model_data["machine_registry"][machine]
            px_mm = float(reg["px_mm"])
            area_corr = float(reg["area_corr"])
            pipeline = self.model_data["pipeline"]
            features = self.model_data.get("features", FINAL_FEATURES)

            # Load CSV. sep=None supports comma and tab-delimited VMEK exports.
            raw = pd.read_csv(self.input_file.get(), sep=None, engine="python")

            # Apply final Option B preprocessing.
            df_corrected, stats = preprocess_for_final_model(raw, px_mm, area_corr, apply_legacy_filter=True)

            n_raw = stats["n_raw"]
            n_filtered = stats["n_classified"]
            n_removed = stats["n_removed"]
            schema = stats["schema"]

            if n_filtered == 0:
                messagebox.showwarning("No Data",
                    f"All {n_raw} kernels were removed during preprocessing.\n"
                    f"Schema detected: {schema}\n"
                    f"Check selected machine and input file.")
                return

            # --- Method 1: Final KNN Model ---
            pred_knn = knn_predict(df_corrected, pipeline, features)
            pct_knn = bucket_pct(pred_knn)

            # --- Method 2: Legacy Area Thresholds ---
            # Use corrected area for fair comparison: same machine correction, different classifier.
            thresholds = self.model_data.get("legacy_area_thresholds_mm2", (34.0, 51.0, 68.0))
            outlier_range = self.model_data.get("outlier_mm2", (8.0, 87.0))
            pred_legacy = legacy_area_classify(df_corrected["Area_mm2_corrected"], thresholds, outlier_range)
            pct_legacy = bucket_pct(pred_legacy)
            n_outlier_legacy = int((pred_legacy == "Outlier").sum())

            # Sample ID from filename
            sample_id = os.path.splitext(os.path.basename(self.input_file.get()))[0]

            # --- Save output CSV ---
            summary_rows = []
            for label, pct, n_out in [
                ("Final KNN Model (Option B, 4 corrected traits)", pct_knn, 0),
                ("Legacy Area Thresholds (34/51/68 corrected mm2)", pct_legacy, n_outlier_legacy),
            ]:
                summary_rows.append({
                    "Sample_ID": sample_id,
                    "Machine": machine,
                    "Machine_Name": reg.get("machine_name", ""),
                    "Schema_Detected": schema,
                    "px_mm": px_mm,
                    "area_corr": area_corr,
                    "linear_corr": math.sqrt(area_corr),
                    "Prediction_Mode": label,
                    "Total_Input": n_raw,
                    "Preprocess_Removed": n_removed,
                    "Kernels_Classified": n_filtered - n_out,
                    "iSCLCN": n_filtered - n_out,
                    "Outliers_Removed": n_out,
                    "Small_Discard_IPSDP_pct": pct["IPSDP"],
                    "Small_Saleable_IPSSP_pct": pct["IPSSP"],
                    "Large_Saleable_IPLSP_pct": pct["IPLSP"],
                    "Large_Discard_IPLDP_pct": pct["IPLDP"],
                })

            summary_df = pd.DataFrame(summary_rows)

            # Per-kernel detail
            df_corrected["KNN_Bucket"] = pred_knn
            df_corrected["Legacy_Bucket"] = pred_legacy

            out_csv = os.path.join(self.output_folder.get(), f"{sample_id}_predictions.csv")
            with open(out_csv, "w", newline="") as fout:
                fout.write(f"# VMEK Bucket Predictor - Final KNN Comparison\n")
                fout.write(f"# Machine: {machine} | px_mm: {px_mm} | area_corr: {area_corr} | schema: {schema}\n")
                fout.write(f"# Model version: {self.model_data.get('version', '?')}\n")
                fout.write(f"# Final KNN features: {', '.join(features)}\n")
                fout.write(f"#\n")
                fout.write(f"# === SUMMARY ===\n")
                summary_df.to_csv(fout, index=False)
                fout.write(f"\n# === PER-KERNEL DETAIL ===\n")
                df_corrected.to_csv(fout, index=False)

            # Update status
            self.status_var.set(
                f"Done | {n_raw:,} input → {n_removed:,} removed → "
                f"{n_filtered:,} classified | Schema: {schema} | Machine: {machine} | "
                f"Output: {os.path.basename(out_csv)}")

            # Draw chart
            self._draw_chart(pct_knn, pct_legacy, n_filtered,
                           n_outlier_legacy, sample_id, machine, area_corr)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

    # ---- Chart ----

    def _draw_chart(self, pct_knn, pct_legacy, n_classified,
                    n_outlier_legacy, sample_id, machine, area_corr):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor("#F5F5F5")

        panel_data = [
            ("Final KNN Model\n(Option B, 4 Corrected Traits)", pct_knn,
             [BUCKET_COLORS[b] for b in BUCKET_ORDER], n_classified, 0),
            ("Legacy Area Thresholds\n(34 / 51 / 68 corrected mm²)", pct_legacy,
             ["#90A4AE"] * 4, n_classified, n_outlier_legacy),
        ]

        for ax, (title, pct, colors, n_total, n_out) in zip(axes, panel_data):
            ax.set_facecolor("#FAFAFA")
            labels = [BUCKET_DISPLAY[b] for b in BUCKET_ORDER]
            vals = [pct[b] for b in BUCKET_ORDER]

            bars = ax.bar(labels, vals, color=colors,
                         width=0.6, edgecolor="white", linewidth=1.2)

            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.6,
                           f"{val:.1f}%",
                           ha="center", va="bottom",
                           fontsize=11, fontweight="bold", color="#333")

            classified = n_total - n_out
            ax.set_ylim(0, max(max(vals), 1) * 1.25 + 5)
            ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
            ax.set_ylabel("% of Kernels", fontsize=9)
            ax.set_xlabel(f"N={classified:,} classified" +
                         (f" | {n_out:,} outliers removed" if n_out > 0 else ""),
                         fontsize=8, color="#666")
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.4, linestyle="--")
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=8)

        fig.suptitle(
            f"Sample: {sample_id}  |  Machine: {machine}  |  "
            f"Area Corr: {area_corr:.5f}",
            fontsize=10, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save chart
        out_png = os.path.join(self.output_folder.get(), f"{sample_id}_comparison.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")

        # Display in GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = VMEKApp()
    app.mainloop()
