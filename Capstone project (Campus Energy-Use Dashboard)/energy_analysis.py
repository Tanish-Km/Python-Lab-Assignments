"""
Energy Usage Analytics – Single File Assignment Solution

Tasks covered:

Task 1: Data Ingestion and Validation
Task 2: Core Aggregation Logic
Task 3: Object-Oriented Modeling
Task 4: Visual Output with Matplotlib
Task 5: Persistence and Executive Summary
"""

from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
LOG_FILE = "energy_analysis.log"

# expected column names in CSVs
TIMESTAMP_COL = "timestamp"
KWH_COL = "kwh"
BUILDING_COL = "building"

# logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())  # also print to console


# -------------------------------------------------------------------
# Utility: generate sample data if /data is empty (helps you run it)
# You can remove this in your submission if teacher provides real CSVs.
# -------------------------------------------------------------------

def generate_sample_data(data_dir: Path) -> None:
    """Create 3 sample CSV files (one per building) if none exist."""
    logging.info("No CSV files found in %s. Generating sample data...", data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = pd.date_range("2024-01-01", "2024-01-31 23:00", freq="H")  # 1 month hourly

    buildings = {
        "Admin": (20, 80),
        "Library": (15, 60),
        "Hostel": (30, 120),
    }

    for name, (base, peak) in buildings.items():
        kwh = base + (peak - base) * np.random.rand(len(rng))
        df = pd.DataFrame(
            {
                TIMESTAMP_COL: rng,
                KWH_COL: kwh.round(2),
                BUILDING_COL: name,
            }
        )
        csv_path = data_dir / f"{name.lower()}_jan.csv"
        df.to_csv(csv_path, index=False)
        logging.info("Sample CSV created: %s", csv_path)


# -------------------------------------------------------------------
# Task 1: Data Ingestion and Validation
# -------------------------------------------------------------------

def ingest_and_validate(data_dir: Path) -> pd.DataFrame:
    """
    Read multiple CSVs from /data/, handle errors, and merge into one DataFrame.
    Adds building and month metadata if missing.
    """
    logging.info("=== Task 1: Data Ingestion and Validation ===")

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        # optional helper: generate sample CSVs so the script can run
        generate_sample_data(data_dir)
        csv_files = list(data_dir.glob("*.csv"))

    frames: List[pd.DataFrame] = []

    for csv in csv_files:
        logging.info("Processing file: %s", csv)
        try:
            # on_bad_lines='skip' for newer pandas; if that fails, fall back
            try:
                df = pd.read_csv(csv, on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(csv, error_bad_lines=False)  # for very old pandas

            # normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            # ensure timestamp column
            if TIMESTAMP_COL not in df.columns:
                raise ValueError(f"{csv} has no '{TIMESTAMP_COL}' column")

            # ensure kwh column
            if KWH_COL not in df.columns:
                # try some common alternatives
                candidates = [c for c in df.columns if "kwh" in c or "consump" in c]
                if candidates:
                    df.rename(columns={candidates[0]: KWH_COL}, inplace=True)
                else:
                    raise ValueError(f"{csv} has no '{KWH_COL}'-like column")

            # convert timestamp
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
            df = df.dropna(subset=[TIMESTAMP_COL])

            # add building name if missing
            if BUILDING_COL not in df.columns:
                building_name = csv.stem.split("_")[0].title()
                df[BUILDING_COL] = building_name

            # add month metadata
            if "month" not in df.columns:
                df["month"] = df[TIMESTAMP_COL].dt.to_period("M").astype(str)

            df["source_file"] = csv.name
            frames.append(df)

        except FileNotFoundError:
            logging.error("File not found: %s", csv)
        except pd.errors.ParserError as e:
            logging.error("Parsing error in %s: %s", csv, e)
        except Exception as e:
            logging.error("Error processing %s: %s", csv, e)

    if not frames:
        raise FileNotFoundError("No valid CSV files could be loaded from /data/.")

    df_combined = pd.concat(frames, ignore_index=True)
    logging.info("Combined DataFrame shape: %s", df_combined.shape)
    return df_combined


# -------------------------------------------------------------------
# Task 2: Core Aggregation Logic
# -------------------------------------------------------------------

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Daily totals per building."""
    df = df.copy()
    df.set_index(TIMESTAMP_COL, inplace=True)
    daily = (
        df.groupby(BUILDING_COL)
        .resample("D")[KWH_COL]
        .sum()
        .reset_index()
        .rename(columns={KWH_COL: "daily_kwh"})
    )
    logging.info("Daily totals shape: %s", daily.shape)
    return daily


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly totals per building."""
    df = df.copy()
    df.set_index(TIMESTAMP_COL, inplace=True)
    weekly = (
        df.groupby(BUILDING_COL)
        .resample("W")[KWH_COL]
        .sum()
        .reset_index()
        .rename(columns={KWH_COL: "weekly_kwh"})
    )
    logging.info("Weekly aggregates shape: %s", weekly.shape)
    return weekly


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary (mean, min, max, total) per building."""
    summary = (
        df.groupby(BUILDING_COL)[KWH_COL]
        .agg(["mean", "min", "max", "sum"])
        .rename(columns={"sum": "total"})
    )
    logging.info("Building-wise summary:\n%s", summary)
    return summary


def summary_dict_from_df(summary_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Convert summary DataFrame to dictionary of dictionaries."""
    result: Dict[str, Dict[str, float]] = {}
    for building, row in summary_df.iterrows():
        result[building] = row.to_dict()
    return result


# -------------------------------------------------------------------
# Task 3: Object-Oriented Modeling
# -------------------------------------------------------------------

@dataclass
class MeterReading:
    timestamp: pd.Timestamp
    kwh: float


class Building:
    def __init__(self, name: str):
        self.name = name
        self.meter_readings: List[MeterReading] = []

    def add_reading(self, timestamp, kwh) -> None:
        self.meter_readings.append(MeterReading(timestamp=pd.to_datetime(timestamp), kwh=float(kwh)))

    def calculate_total_consumption(self) -> float:
        return float(sum(r.kwh for r in self.meter_readings))

    def generate_report(self) -> Dict[str, float]:
        if not self.meter_readings:
            return {"total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        values = np.array([r.kwh for r in self.meter_readings])
        return {
            "total": float(values.sum()),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }


class BuildingManager:
    def __init__(self):
        self.buildings: Dict[str, Building] = {}

    def get_or_create(self, name: str) -> Building:
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Populate Building objects from combined DataFrame."""
        for _, row in df[[BUILDING_COL, TIMESTAMP_COL, KWH_COL]].iterrows():
            b = self.get_or_create(row[BUILDING_COL])
            b.add_reading(row[TIMESTAMP_COL], row[KWH_COL])

    def building_reports(self) -> Dict[str, Dict[str, float]]:
        return {name: b.generate_report() for name, b in self.buildings.items()}


# -------------------------------------------------------------------
# Task 4: Visual Output with Matplotlib
# -------------------------------------------------------------------

def create_dashboard(daily: pd.DataFrame, weekly: pd.DataFrame, df: pd.DataFrame) -> None:
    """Create dashboard-style figure with 3 plots and save as dashboard.png."""
    logging.info("=== Task 4: Creating dashboard visualizations ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax4.axis("off")  # we only need 3 plots

    # 1. Trend Line – daily consumption over time for all buildings
    for building, grp in daily.groupby(BUILDING_COL):
        ax1.plot(grp[TIMESTAMP_COL], grp["daily_kwh"], label=building)
    ax1.set_title("Daily Consumption Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("kWh")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Bar Chart – average weekly usage across buildings
    weekly_avg = weekly.groupby(BUILDING_COL)["weekly_kwh"].mean()
    ax2.bar(weekly_avg.index, weekly_avg.values)
    ax2.set_title("Average Weekly Usage per Building")
    ax2.set_xlabel("Building")
    ax2.set_ylabel("Average Weekly kWh")

    # 3. Scatter Plot – peak (max) daily consumption vs date per building
    #    (approximation of "peak-hour consumption vs time/building")
    peak_daily = (
        daily.sort_values("daily_kwh")
        .groupby(BUILDING_COL)
        .tail(1)
    )
    for _, row in peak_daily.iterrows():
        ax3.scatter(row[TIMESTAMP_COL], row["daily_kwh"], label=row[BUILDING_COL])
    ax3.set_title("Peak Daily Consumption per Building")
    ax3.set_xlabel("Date of Peak")
    ax3.set_ylabel("Daily kWh at Peak")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Energy Usage Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = OUTPUT_DIR / "dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Dashboard saved to %s", out_path)


# -------------------------------------------------------------------
# Task 5: Persistence and Executive Summary
# -------------------------------------------------------------------

def write_outputs(
    df_combined: pd.DataFrame,
    summary_df: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> None:
    logging.info("=== Task 5: Persistence and Executive Summary ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export cleaned combined dataset
    cleaned_path = OUTPUT_DIR / "cleaned_energy_data.csv"
    df_combined.to_csv(cleaned_path, index=False)
    logging.info("Cleaned data exported to %s", cleaned_path)

    # Export building summary
    summary_path = OUTPUT_DIR / "building_summary.csv"
    summary_df.to_csv(summary_path)
    logging.info("Building summary exported to %s", summary_path)

    # Executive summary calculations
    total_campus_consumption = df_combined[KWH_COL].sum()
    highest_building = summary_df["total"].idxmax()
    highest_building_total = summary_df.loc[highest_building, "total"]

    # Peak load time (single highest kWh record)
    peak_row = df_combined.loc[df_combined[KWH_COL].idxmax()]
    peak_time = peak_row[TIMESTAMP_COL]
    peak_building = peak_row[BUILDING_COL]
    peak_value = peak_row[KWH_COL]

    # Simple trend comments
    daily_totals_overall = (
        daily.groupby(TIMESTAMP_COL)["daily_kwh"].sum().reset_index()
    )
    first_week = weekly[weekly[TIMESTAMP_COL] == weekly[TIMESTAMP_COL].min()][
        "weekly_kwh"
    ].sum()
    last_week = weekly[weekly[TIMESTAMP_COL] == weekly[TIMESTAMP_COL].max()][
        "weekly_kwh"
    ].sum()

    trend_comment = (
        "Weekly consumption increased over the period."
        if last_week > first_week
        else "Weekly consumption decreased or remained stable over the period."
    )

    summary_lines = [
        "Energy Usage Executive Summary",
        "================================",
        f"Total campus consumption: {total_campus_consumption:.2f} kWh",
        f"Highest-consuming building: {highest_building} "
        f"({highest_building_total:.2f} kWh)",
        f"Peak load time: {peak_time} in building {peak_building} "
        f"({peak_value:.2f} kWh)",
        "",
        "Daily & Weekly Trends:",
        f"- Daily totals recorded for {len(daily_totals_overall)} days.",
        f"- {trend_comment}",
    ]

    summary_path_txt = OUTPUT_DIR / "summary.txt"
    with summary_path_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logging.info("Summary written to %s", summary_path_txt)

    # Optionally print to console
    print("\n".join(summary_lines))


# -------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------

def main() -> None:
    # Task 1: Ingest and validate
    df_combined = ingest_and_validate(DATA_DIR)

    # Ensure timestamp is datetime
    df_combined[TIMESTAMP_COL] = pd.to_datetime(df_combined[TIMESTAMP_COL])

    # Task 2: Aggregations
    daily = calculate_daily_totals(df_combined)
    weekly = calculate_weekly_aggregates(df_combined)
    summary_df = building_wise_summary(df_combined)
    summary_dict = summary_dict_from_df(summary_df)
    logging.info("Building summary dict: %s", summary_dict)

    # Task 3: OOP modeling
    manager = BuildingManager()
    manager.load_from_dataframe(df_combined)
    reports = manager.building_reports()
    logging.info("OOP building reports: %s", reports)

    # Task 4: Visual output
    create_dashboard(daily, weekly, df_combined)

    # Task 5: Persistence and executive summary
    write_outputs(df_combined, summary_df, daily, weekly)


if __name__ == "__main__":
    main()
