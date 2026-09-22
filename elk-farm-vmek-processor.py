#!/usr/bin/env python3

"""Elk Farm VMEK raw-folder processor.

Processes one selected folder containing Bin*.csv and Sample*.csv files.
Produces one traits CSV named after the original folder and renames the folder
with _Processed only when all sample files completed successfully.
"""

from __future__ import annotations

import math
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from openpyxl import load_workbook


DEFAULT_ROOT = Path(
    r"C:\Users\s1064440\OneDrive - Syngenta\SPR North America - VMEK Data\2026"
)
DEFAULT_MODEL = Path(
    r"C:\Users\s1064440\OneDrive - Syngenta\Vmek info\Github files\vmek-knn-final-option-b-ql283-plus-2021-ipldp.joblib"
)
DEFAULT_SHELLING_FILE = Path(
    r"C:\Users\s1064440\OneDrive - Syngenta\SPR North America - VMEK Data\2026 Elk Farm Shelling.xlsx"
)

MACHINE_NAME = "Elk Farm"
PX_MM = 0.332
AREA_CORR = 1.08911

FEATURES = [
    "Area_mm2_corrected",
    "Length_mm_corrected",
    "Width_mm_corrected",
    "ContLength_mm_equiv_corrected",
]

BUCKETS = ["IPSDP", "IPSSP", "IPLSP", "IPLDP"]

OUTPUT_COLUMNS = [
    "SampleName",
    "Machine_Name",
    "Processing_Status",
    "BARCD",
    "IPSDP",
    "IPSSP",
    "IPLSP",
    "IPLDP",
    "DSCDP",
    "iSCLCN",
    "iSAVAN",
    "iSALNN",
    "iSAWDN",
    "SDLBN",
]

warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*",
    category=UserWarning,
)


class ProcessError(Exception):
    pass


def clean_name(value: object) -> str:
    return str(value).strip().upper()


def barcode_from_text(value: object) -> str:
    """Return LC, DR, or LD identifier when one exists; non-barcode samples are valid."""
    m = re.search(r"(?:LC|DR|LD)[A-Z0-9]{9}", str(value).upper())
    return m.group(0) if m else ""


def sample_name_from_data(raw: pd.DataFrame, source: Path) -> str:
    if "SampleName" not in raw.columns:
        raise ProcessError(f"{source.name} is missing the SampleName column.")

    names = raw["SampleName"].dropna().astype(str).str.strip()
    names = names[names.ne("")]
    unique = names.unique()

    if len(unique) != 1:
        raise ProcessError(
            f"Expected one nonblank SampleName in {source.name}; found {len(unique)}."
        )

    return str(unique[0])


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_model(model_path: Path):
    package = joblib.load(model_path)

    if isinstance(package, dict) and "pipeline" in package:
        return package["pipeline"], package.get("metadata", {})

    return package, {}


def predict_traits(raw: pd.DataFrame, model, metadata: dict) -> dict:
    needed = {"Area", "Length", "Width", "Roundness"}
    missing = needed.difference(raw.columns)

    if missing:
        raise ProcessError(
            f"Missing Elk Farm sample columns: {', '.join(sorted(missing))}"
        )

    raw = numeric(raw, ["Area", "Length", "Width", "Roundness"])

    # Clean-count / means basis: projected Area strictly greater than 8 and less than 87 mm².
    clean = (
        raw.dropna(subset=["Area", "Length", "Width"])
        .query("Area > 8 and Area < 87")
        .copy()
    )

    if clean.empty:
        raise ProcessError(
            "No kernels passed the clean projected-area range (>8 and <87 mm²)."
        )

    # Retain established KNN preprocessing for the Elk Farm MX4 schema.
    knn = raw.dropna(subset=["Area", "Length", "Width", "Roundness"]).copy()
    knn = knn[knn["Roundness"] > 0].copy()

    if knn.empty:
        raise ProcessError(
            "No valid kernels remained for KNN prediction (Roundness must be >0)."
        )

    linear_corr = math.sqrt(AREA_CORR)

    knn["Area_mm2_corrected"] = knn["Area"] * AREA_CORR
    knn["Length_mm_corrected"] = knn["Length"] * linear_corr
    knn["Width_mm_corrected"] = knn["Width"] * linear_corr
    knn["ContLength_mm_equiv_corrected"] = np.sqrt(
        (4 * math.pi * knn["Area_mm2_corrected"]) / knn["Roundness"]
    )

    features = metadata.get("features", FEATURES)
    absent = [f for f in features if f not in knn.columns]

    if absent:
        raise ProcessError(
            f"Model requires unavailable features: {', '.join(absent)}"
        )

    predictions = model.predict(knn[features])
    counts = pd.Series(predictions).value_counts().reindex(BUCKETS, fill_value=0)
    total = int(counts.sum())

    if not total:
        raise ProcessError("KNN model returned no bucket predictions.")

    pct = counts / total * 100

    return {
        "IPSDP": round(float(pct["IPSDP"]), 3),
        "IPSSP": round(float(pct["IPSSP"]), 3),
        "IPLSP": round(float(pct["IPLSP"]), 3),
        "IPLDP": round(float(pct["IPLDP"]), 3),
        "DSCDP": round(float(pct["IPSDP"] + pct["IPLDP"]), 3),
        "iSCLCN": int(len(clean)),
        "iSAVAN": round(float(clean["Area"].mean()), 4),
        "iSALNN": round(float(clean["Length"].mean()), 4),
        "iSAWDN": round(float(clean["Width"].mean()), 4),
    }


def read_and_combine_bin_files(
    bin_files: list[Path],
    summary_output: Path,
) -> dict[str, list[pd.Series]]:
    """Combine all Bin files, deduplicate exact duplicates, and write the summary."""
    combined_frames = []

    for path in bin_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        mapped = {str(column).strip(): column for column in df.columns}

        required = ["Sample Name", "Accept Count", "Accept"]

        if any(column not in mapped for column in required):
            continue

        rename_map = {
            mapped["Sample Name"]: "Sample Name",
            mapped["Accept Count"]: "Accept Count",
            mapped["Accept"]: "Accept Weight",
        }

        df = df.rename(columns=rename_map).copy()

        if "Accept Weight" not in df.columns:
            df["Accept Weight"] = np.nan

        df["Source_Bin_File"] = path.name
        df["_Sample_Name_Key"] = df["Sample Name"].apply(clean_name)
        df["_Accept_Count_Key"] = (
            pd.to_numeric(df["Accept Count"], errors="coerce")
            .astype("Float64")
            .astype(str)
        )
        df["_Accept_Weight_Key"] = (
            pd.to_numeric(df["Accept Weight"], errors="coerce")
            .astype("Float64")
            .astype(str)
        )

        df = df[df["_Sample_Name_Key"].ne("")].copy()

        if not df.empty:
            combined_frames.append(df)

    if not combined_frames:
        raise ProcessError(
            "No valid Bin records were found. Bin files require Sample Name and Accept Count."
        )

    combined = pd.concat(combined_frames, ignore_index=True, sort=False)

    summary = combined.drop_duplicates(
        subset=[
            "_Sample_Name_Key",
            "_Accept_Count_Key",
            "_Accept_Weight_Key",
        ],
        keep="first",
    ).copy()

    summary_to_write = summary.drop(
        columns=[
            "_Sample_Name_Key",
            "_Accept_Count_Key",
            "_Accept_Weight_Key",
        ],
        errors="ignore",
    )

    summary_to_write.to_csv(summary_output, index=False)

    indexed_rows: dict[str, list[pd.Series]] = {}

    for _, row in summary.iterrows():
        indexed_rows.setdefault(row["_Sample_Name_Key"], []).append(row)

    return indexed_rows


def bin_rows_for_name(
    sample_name: str,
    indexed_rows: dict[str, list[pd.Series]],
) -> list[pd.Series]:
    """Return all distinct Bin count/weight records for a sample."""
    matches = indexed_rows.get(clean_name(sample_name), [])

    if not matches:
        raise ProcessError(
            f"No bin-summary row matched SampleName '{sample_name}'."
        )

    return matches


def eligible_folders(root: Path) -> list[Path]:
    """Return only the selected folder when it contains Sample and Bin CSV files."""
    if not root.is_dir() or "processed" in root.name.lower():
        return []

    files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    ]

    has_sample = any(
        path.name.lower().startswith("sample")
        for path in files
    )

    has_bin = any(
        path.name.lower().startswith("bin")
        for path in files
    )

    return [root] if has_sample and has_bin else []


def status_for(sdlbn: float | None, clean_count: int) -> str:
    """Return normal processing status when there is one valid Bin record."""
    reasons = []

    if sdlbn is not None and sdlbn > 3000:
        reasons.append("SDLBN>3000")

    if clean_count < 300:
        reasons.append("iSCLCN<300")

    if clean_count > 5000:
        reasons.append("iSCLCN>5000")

    return "review: " + "; ".join(reasons) if reasons else "success"


def output_row_for_bin(
    output_sample_name: str,
    barcode: str,
    raw: pd.DataFrame,
    bin_row: pd.Series,
    model,
    metadata: dict,
    weight_check_required: bool,
) -> dict:
    """Create one output row for one distinct matching Bin record."""
    accepted_count = pd.to_numeric(
        bin_row.get("Accept Count"),
        errors="coerce",
    )

    accepted_weight = pd.to_numeric(
        bin_row.get("Accept Weight"),
        errors="coerce",
    )

    if pd.isna(accepted_count):
        raise ProcessError("Accept Count must be present.")

    traits = predict_traits(raw, model, metadata)

    if weight_check_required:
        processing_status = "Weight Check Required"

        if pd.isna(accepted_weight) or accepted_weight <= 0:
            sdlbn = np.nan
        else:
            sdlbn = round(
                float(accepted_count / accepted_weight),
                3,
            )

    elif pd.isna(accepted_weight) or accepted_weight <= 0:
        processing_status = "Missing Weight"
        sdlbn = np.nan

    else:
        sdlbn = round(
            float(accepted_count / accepted_weight),
            3,
        )

        processing_status = status_for(
            sdlbn,
            traits["iSCLCN"],
        )

    return {
        "SampleName": output_sample_name,
        "Machine_Name": MACHINE_NAME,
        "Processing_Status": processing_status,
        "BARCD": barcode,
        **traits,
        "SDLBN": sdlbn,
    }


def process_folder(
    folder: Path,
    model,
    metadata: dict,
) -> tuple[Path, Path, int, list[str]]:
    """Process all Sample files in the selected folder."""
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    ]

    sample_files = sorted(
        path
        for path in files
        if path.name.lower().startswith("sample")
    )

    bin_files = sorted(
        path
        for path in files
        if path.name.lower().startswith("bin")
    )

    if not sample_files or not bin_files:
        raise ProcessError(
            "Folder must contain at least one Sample*.csv and one Bin*.csv file."
        )

    original_folder_name = folder.name

    traits_output = folder / f"{original_folder_name}.csv"
    bin_summary_output = folder / f"{original_folder_name}_bin_summary.csv"

    bins = read_and_combine_bin_files(
        bin_files,
        bin_summary_output,
    )

    rows = []
    processed_barcodes = []
    failures = 0

    for sample_file in sample_files:
        sample_name = sample_file.stem
        barcode = barcode_from_text(sample_file.stem)

        try:
            raw = pd.read_csv(sample_file)

            sample_name = sample_name_from_data(raw, sample_file)
            barcode = barcode_from_text(sample_name) or barcode

            matching_bin_rows = bin_rows_for_name(
                sample_name,
                bins,
            )

            weight_check_required = len(matching_bin_rows) > 1

            for index, bin_row in enumerate(matching_bin_rows):
                output_sample_name = (
                    sample_name
                    if index == 0
                    else f"{sample_name}_{index}"
                )

                rows.append(
                    output_row_for_bin(
                        output_sample_name=output_sample_name,
                        barcode=barcode,
                        raw=raw,
                        bin_row=bin_row,
                        model=model,
                        metadata=metadata,
                        weight_check_required=weight_check_required,
                    )
                )

        except Exception as exc:
            failures += 1

            rows.append(
                {
                    "SampleName": sample_name,
                    "Machine_Name": MACHINE_NAME,
                    "Processing_Status": f"failed: {exc}",
                    "BARCD": barcode,
                    **{
                        column: np.nan
                        for column in OUTPUT_COLUMNS
                        if column
                        not in {
                            "SampleName",
                            "Machine_Name",
                            "Processing_Status",
                            "BARCD",
                        }
                    },
                }
            )

        if barcode:
            processed_barcodes.append(barcode)

    pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    ).to_csv(
        traits_output,
        index=False,
    )

    if failures:
        return traits_output, bin_summary_output, len(rows), processed_barcodes

    renamed = folder.with_name(f"{original_folder_name}_Processed")

    if renamed.exists():
        return traits_output, bin_summary_output, len(rows), processed_barcodes

    folder.rename(renamed)

    return (
        renamed / traits_output.name,
        renamed / bin_summary_output.name,
        len(rows),
        processed_barcodes,
    )


class ShellingTracker:
    """Non-blocking workbook update for the shelling tracker."""

    def __init__(self, workbook_path: Path):
        self.workbook_path = workbook_path
        self.workbook = None
        self.sheet = None
        self.rows_by_barcode: dict[str, list[int]] = {}

    def load(self) -> None:
        self.workbook = load_workbook(self.workbook_path)

        if "Plots" not in self.workbook.sheetnames:
            raise ProcessError(
                "The shelling workbook does not contain a 'Plots' tab."
            )

        self.sheet = self.workbook["Plots"]

        for row in range(2, self.sheet.max_row + 1):
            barcode = self.sheet.cell(row=row, column=3).value

            if barcode is not None and str(barcode).strip():
                self.rows_by_barcode.setdefault(
                    clean_name(barcode),
                    [],
                ).append(row)

    def mark_complete(self, barcode: str) -> int:
        if not self.sheet or not barcode:
            return 0

        rows = self.rows_by_barcode.get(clean_name(barcode), [])

        for row in rows:
            self.sheet.cell(row=row, column=9).value = "Yes"

        return len(rows)

    def save(self) -> None:
        if self.workbook:
            self.workbook.save(self.workbook_path)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Elk Farm VMEK Processor")
        self.geometry("900x590")

        self.root_path = tk.StringVar(value=str(DEFAULT_ROOT))
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL))
        self.shelling_path = tk.StringVar(value=str(DEFAULT_SHELLING_FILE))

        self._build()

    def _build(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(
            frame,
            text="Elk Farm VMEK Processor",
            font=("Segoe UI", 16, "bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
        )

        for row, label, var, choose in [
            (1, "Input folder", self.root_path, self.pick_root),
            (2, "KNN model (.joblib)", self.model_path, self.pick_model),
            (
                3,
                "Shelling workbook (.xlsx)",
                self.shelling_path,
                self.pick_shelling_file,
            ),
        ]:
            ttk.Label(frame, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                pady=4,
            )

            ttk.Entry(
                frame,
                textvariable=var,
            ).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=8,
            )

            ttk.Button(
                frame,
                text="Browse…",
                command=choose,
            ).grid(
                row=row,
                column=2,
            )

        self.log = tk.Text(
            frame,
            wrap="word",
            height=20,
        )

        self.log.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=10,
        )

        ttk.Button(
            frame,
            text="Process selected folder",
            command=self.run,
        ).grid(
            row=5,
            column=0,
            columnspan=3,
        )

    def write(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.update_idletasks()

    def pick_root(self):
        p = filedialog.askdirectory()
        self.root_path.set(p or self.root_path.get())

    def pick_model(self):
        p = filedialog.askopenfilename(
            filetypes=[("Joblib", "*.joblib")]
        )
        self.model_path.set(p or self.model_path.get())

    def pick_shelling_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("Excel workbook", "*.xlsx")]
        )
        self.shelling_path.set(p or self.shelling_path.get())

    def run(self):
        root = Path(self.root_path.get())
        model_path = Path(self.model_path.get())
        shelling_path = Path(self.shelling_path.get())

        if not root.is_dir() or not model_path.is_file():
            messagebox.showerror(
                "Missing input",
                "Select a valid input folder and KNN model.",
            )
            return

        tracker = None

        try:
            if shelling_path.is_file():
                try:
                    tracker = ShellingTracker(shelling_path)
                    tracker.load()
                    self.write("Loaded shelling workbook for write-back.")

                except Exception as exc:
                    tracker = None
                    self.write(
                        "Shelling workbook unavailable; processing will continue "
                        f"without write-back: {exc}"
                    )

            else:
                self.write(
                    "Shelling workbook not found; processing will continue "
                    "without write-back."
                )

            self.write("Loading KNN model…")
            model, metadata = load_model(model_path)

            folders = eligible_folders(root)

            self.write(
                f"Found {len(folders)} eligible selected folder(s)."
            )

            for folder in folders:
                self.write(f"Processing: {folder}")

                traits_file, bin_summary_file, row_count, barcodes = process_folder(
                    folder,
                    model,
                    metadata,
                )

                self.write(
                    f"  Complete: {row_count} output row(s) → {traits_file}"
                )

                self.write(
                    f"  Bin summary → {bin_summary_file}"
                )

                if tracker:
                    matched_rows = sum(
                        tracker.mark_complete(barcode)
                        for barcode in barcodes
                    )

                    self.write(
                        f"  Shelling tracker: marked {matched_rows} "
                        "Plot row(s) as Yes."
                    )

            if tracker:
                try:
                    tracker.save()
                    self.write("Saved shelling workbook write-back.")

                except Exception as exc:
                    self.write(
                        "Shelling workbook could not be saved; CSV output "
                        f"remains available: {exc}"
                    )

            messagebox.showinfo(
                "Complete",
                f"Processed {len(folders)} folder(s).",
            )

        except Exception as exc:
            self.write("ERROR: " + str(exc))
            self.write(traceback.format_exc())

            messagebox.showerror(
                "Processing stopped",
                str(exc),
            )


if __name__ == "__main__":
    App().mainloop()
