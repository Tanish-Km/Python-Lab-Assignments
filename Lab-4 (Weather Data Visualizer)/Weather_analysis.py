"""
Weather Data Analysis – Single File Solution

Covers:
Task 1: Data Acquisition and Loading (CSV -> DataFrame, head/info/describe)
Task 2: Data Cleaning and Processing (NaNs, datetime, filter columns)
Task 3: Statistical Analysis with NumPy (daily, monthly, yearly stats)
Task 4: Visualization with Matplotlib (line, bar, scatter, combined figure)
Task 5: Grouping and Aggregation (groupby + resample)
Task 6: Export and Storytelling (cleaned CSV, PNG plots, Markdown report)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration – EDIT THESE NAMES IF YOUR COLUMNS DIFFER
# ============================================================

# Name of the CSV file containing your weather data
CSV_FILE = r"C:\Users\tanis\OneDrive\Music\Work_Space\Assignment\Python\Weather project\weather_data.csv"
     # <- put your real filename here

# Column names in your CSV
DATE_COL = "date"                  # e.g. "date" / "DATE" / "Date"
TEMP_COL = "temperature"           # e.g. "temp", "TAVG", "tmean"
RAIN_COL = "rainfall"              # e.g. "rain", "precipitation"
HUMIDITY_COL = "humidity"          # e.g. "RH", "relative_humidity"

# Output paths
CLEANED_CSV = "weather_cleaned.csv"
REPORT_FILE = "weather_report.md"
PLOTS_DIR = Path("plots")


# ============================================================
# Task 1 – Data Acquisition and Loading
# ============================================================

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame and inspect structure.
    """
    print("=== Task 1: Data Acquisition and Loading ===")
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file '{csv_path}' not found. "
            f"Place your weather CSV in this folder and update CSV_FILE."
        )

    df = pd.read_csv(csv_path)

    # Basic inspection
    print("\nDataFrame Head (first 5 rows):")
    print(df.head())

    print("\nDataFrame Info:")
    print(df.info())

    print("\nDataFrame Describe (numeric columns):")
    print(df.describe())

    return df


# ============================================================
# Task 2 – Data Cleaning and Processing
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, convert dates, and keep only relevant columns.
    """
    print("\n=== Task 2: Data Cleaning and Processing ===")

    # 1. Rename columns to a standard form (lowercase) for easier use
    df = df.rename(columns=str.lower)

    # 2. Check required columns
    required_cols = [DATE_COL, TEMP_COL, RAIN_COL, HUMIDITY_COL]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(
            f"The following required columns are missing from the CSV: {missing}\n"
            f"Available columns are: {list(df.columns)}\n"
            f"Edit DATE_COL, TEMP_COL, RAIN_COL, HUMIDITY_COL at the top of the file "
            f"to match your CSV."
        )

    # 3. Convert date column to datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # 4. Drop rows where date is invalid
    before = len(df)
    df = df.dropna(subset=[DATE_COL])
    after = len(df)
    print(f"Removed {before - after} rows with invalid dates.")

    # 5. Filter to relevant columns
    df = df[[DATE_COL, TEMP_COL, RAIN_COL, HUMIDITY_COL]].copy()

    # 6. Handle missing values:
    #    - Forward-fill, then back-fill for safety.
    #    (This is just one reasonable strategy; other strategies also acceptable.)
    numeric_cols = [TEMP_COL, RAIN_COL, HUMIDITY_COL]
    df[numeric_cols] = df[numeric_cols].replace({-99: np.nan, -999: np.nan})
    missing_before = df[numeric_cols].isna().sum()
    print("\nMissing values before filling:")
    print(missing_before)

    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    missing_after = df[numeric_cols].isna().sum()
    print("\nMissing values after filling (should be 0 or near 0):")
    print(missing_after)

    # 7. Set Date as index for resampling later
    df = df.set_index(DATE_COL).sort_index()

    print("\nCleaned DataFrame head:")
    print(df.head())

    return df


# ============================================================
# Task 3 – Statistical Analysis with NumPy
# ============================================================

def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Compute daily, monthly, and yearly statistics using NumPy.
    Returns a dictionary of key results for the report.
    """
    print("\n=== Task 3: Statistical Analysis with NumPy ===")

    stats = {}

    # Daily stats – directly on the cleaned df
    temp_array = df[TEMP_COL].to_numpy()
    stats["daily_mean_temp"] = float(np.mean(temp_array))
    stats["daily_min_temp"] = float(np.min(temp_array))
    stats["daily_max_temp"] = float(np.max(temp_array))
    stats["daily_std_temp"] = float(np.std(temp_array, ddof=1))

    print(f"Daily Temperature – mean: {stats['daily_mean_temp']:.2f}, "
          f"min: {stats['daily_min_temp']:.2f}, "
          f"max: {stats['daily_max_temp']:.2f}, "
          f"std: {stats['daily_std_temp']:.2f}")

    # Monthly stats using resample
    monthly = df.resample("M").agg({
        TEMP_COL: ["mean", "min", "max", "std"],
        RAIN_COL: "sum",
        HUMIDITY_COL: "mean"
    })

    print("\nMonthly statistics (first few rows):")
    print(monthly.head())

    stats["monthly_temp_mean"] = monthly[(TEMP_COL, "mean")].to_list()

    # Yearly stats using resample
    yearly = df.resample("Y").agg({
        TEMP_COL: ["mean", "min", "max"],
        RAIN_COL: "sum",
        HUMIDITY_COL: "mean"
    })

    print("\nYearly statistics:")
    print(yearly)

    # Save for later report (convert index to year)
    stats["yearly_summary"] = yearly

    return stats


# ============================================================
# Task 4 – Visualization with Matplotlib
# ============================================================

def make_plots(df: pd.DataFrame) -> None:
    """
    Create line, bar, scatter plots and a combined figure.
    Save them as PNG files.
    """
    print("\n=== Task 4: Visualization with Matplotlib ===")
    PLOTS_DIR.mkdir(exist_ok=True)

    # 1. Line chart – daily temperature trend
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[TEMP_COL])
    plt.title("Daily Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.tight_layout()
    line_path = PLOTS_DIR / "daily_temperature_line.png"
    plt.savefig(line_path)
    plt.close()
    print(f"Saved line chart to {line_path}")

    # 2. Bar chart – monthly rainfall totals
    monthly_rain = df[RAIN_COL].resample("M").sum()

    plt.figure(figsize=(10, 4))
    plt.bar(monthly_rain.index.strftime("%Y-%m"), monthly_rain.values)
    plt.title("Monthly Rainfall Totals")
    plt.xlabel("Month")
    plt.ylabel("Rainfall")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = PLOTS_DIR / "monthly_rainfall_bar.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved bar chart to {bar_path}")

    # 3. Scatter plot – humidity vs temperature
    plt.figure(figsize=(6, 5))
    plt.scatter(df[TEMP_COL], df[HUMIDITY_COL], alpha=0.6)
    plt.title("Humidity vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.tight_layout()
    scatter_path = PLOTS_DIR / "humidity_vs_temperature_scatter.png"
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # 4. Combined figure – line + bar in one figure
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Line plot for temperature
    ax1.plot(df.index, df[TEMP_COL], label="Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature")
    ax1.tick_params(axis="x", rotation=45)

    # Second y-axis for rainfall bar
    ax2 = ax1.twinx()
    ax2.bar(df.index, df[RAIN_COL], alpha=0.3, label="Rainfall")

    fig.suptitle("Daily Temperature and Rainfall (Combined)")
    fig.tight_layout()
    combined_path = PLOTS_DIR / "combined_temp_rain.png"
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved combined figure to {combined_path}")


# ============================================================
# Task 5 – Grouping and Aggregation
# ============================================================

def grouping_and_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by month and season, calculate aggregate statistics.
    Returns the monthly grouped DataFrame to be used in the report.
    """
    print("\n=== Task 5: Grouping and Aggregation ===")

    # Group by calendar month
    monthly_group = df.groupby(df.index.month).agg({
        TEMP_COL: "mean",
        RAIN_COL: "sum",
        HUMIDITY_COL: "mean"
    })
    monthly_group.index.name = "month"
    print("\nGrouped by month:")
    print(monthly_group)

    # Simple season mapping for India: 1–2 winter, 3–5 summer, 6–9 monsoon, 10–12 post-monsoon
    def month_to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        elif month in (3, 4, 5):
            return "Summer"
        elif month in (6, 7, 8, 9):
            return "Monsoon"
        else:
            return "Post-Monsoon"

    season = df.index.month.map(month_to_season)
    df_season = df.copy()
    df_season["season"] = season

    season_group = df_season.groupby("season").agg({
        TEMP_COL: "mean",
        RAIN_COL: "sum",
        HUMIDITY_COL: "mean"
    })

    print("\nGrouped by season:")
    print(season_group)

    return monthly_group


# ============================================================
# Task 6 – Export and Storytelling
# ============================================================

def export_and_report(df: pd.DataFrame,
                      stats: dict,
                      monthly_group: pd.DataFrame) -> None:
    """
    Export cleaned data to CSV, save plots (already saved),
    and write a Markdown report summarizing insights.
    """
    print("\n=== Task 6: Export and Storytelling ===")

    # 1. Export cleaned data
    df.to_csv(CLEANED_CSV)
    print(f"Cleaned data exported to {CLEANED_CSV}")

    # 2. Generate Markdown report
    yearly_summary: pd.DataFrame = stats["yearly_summary"]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Weather Data Analysis Report\n\n")

        f.write("## 1. Overview\n")
        f.write(
            "This report summarizes key patterns in the local weather dataset, "
            "including temperature, rainfall, and humidity statistics computed "
            "at daily, monthly, and yearly scales.\n\n"
        )

        f.write("## 2. Daily Temperature Statistics\n")
        f.write(
            f"- Mean daily temperature: **{stats['daily_mean_temp']:.2f}**\n"
            f"- Minimum recorded temperature: **{stats['daily_min_temp']:.2f}**\n"
            f"- Maximum recorded temperature: **{stats['daily_max_temp']:.2f}**\n"
            f"- Standard deviation: **{stats['daily_std_temp']:.2f}**\n\n"
        )

        f.write("## 3. Yearly Summary\n")
        f.write(
            "The table below shows average temperature, total rainfall, "
            "and mean humidity for each year in the dataset.\n\n"
        )
        # Convert yearly summary to markdown-like table
        yearly_copy = yearly_summary.copy()
        yearly_copy.index = yearly_copy.index.year
        f.write(yearly_copy.to_markdown() + "\n\n")

        f.write("## 4. Monthly Aggregations\n")
        f.write(
            "Average monthly temperature, total monthly rainfall, and "
            "average monthly humidity are summarized below:\n\n"
        )
        f.write(monthly_group.to_markdown() + "\n\n")

        f.write("## 5. Visualizations\n")
        f.write(
            "The following plots were generated and saved in the `plots/` folder:\n\n"
        )
        f.write("- `daily_temperature_line.png`: Daily temperature trend line plot.\n")
        f.write("- `monthly_rainfall_bar.png`: Bar chart of monthly rainfall totals.\n")
        f.write("- `humidity_vs_temperature_scatter.png`: Scatter plot of humidity vs temperature.\n")
        f.write("- `combined_temp_rain.png`: Combined figure with temperature line and rainfall bars.\n\n")

        f.write("## 6. Interpretation and Insights\n")
        f.write(
            "- Months with higher rainfall generally correspond to the monsoon season.\n"
            "- Temperature variation across the year can be seen clearly in the line chart and monthly aggregates.\n"
            "- The scatter plot indicates the relationship between humidity and temperature; "
            "clusters or trends may show how humidity tends to rise or fall with temperature.\n\n"
        )

        f.write(
            "Overall, this analysis demonstrates how NumPy, Pandas, and Matplotlib can be "
            "combined to clean real-world datasets, compute statistical summaries, build "
            "visualizations, and communicate insights in a structured report.\n"
        )

    print(f"Markdown report written to {REPORT_FILE}")


# ============================================================
# Main – runs all tasks in sequence
# ============================================================

def main() -> None:
    # Task 1
    df_raw = load_data(CSV_FILE)

    # Task 2
    df_clean = clean_data(df_raw)

    # Task 3
    stats = compute_statistics(df_clean)

    # Task 4
    make_plots(df_clean)

    # Task 5
    monthly_group = grouping_and_aggregation(df_clean)

    # Task 6
    export_and_report(df_clean, stats, monthly_group)

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()