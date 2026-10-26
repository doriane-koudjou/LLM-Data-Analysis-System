"""
visualizer.py
-------------
Visualization utilities for analysis results and forecast plots.
Handles chart generation for both Streamlit display and file export.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from io import BytesIO


# Default style
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def plot_bar_chart(
    data: list[dict],
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = "#4C72B0",
) -> BytesIO:
    """Generates a bar chart from query results."""
    df = pd.DataFrame(data)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    fig, ax = plt.subplots()
    ax.bar(df[x_col].astype(str), df[y_col], color=color, edgecolor="white")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_line_chart(
    data: list[dict],
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = "#4C72B0",
) -> BytesIO:
    """Generates a line chart from query results."""
    df = pd.DataFrame(data)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # Try to parse dates
    try:
        df[x_col] = pd.to_datetime(df[x_col])
        date_axis = True
    except Exception:
        date_axis = False

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col], color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())

    if date_axis:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_forecast(forecast_result: dict) -> BytesIO:
    """
    Generates a forecast plot showing historical data and predictions.
    Input is the output dict from predictive_tool.
    """
    historical = pd.DataFrame(forecast_result["historical"])
    forecast_values = forecast_result["forecast"]
    model_used = forecast_result.get("model_used", "unknown")

    # Detect date and value columns
    date_col = historical.columns[0]
    value_col = historical.columns[1]

    historical[date_col] = pd.to_datetime(historical[date_col])
    historical[value_col] = pd.to_numeric(historical[value_col], errors="coerce")

    # Generate future dates
    last_date = historical[date_col].max()
    freq = pd.infer_freq(historical[date_col]) or "MS"
    future_dates = pd.date_range(
        start=last_date,
        periods=len(forecast_values) + 1,
        freq=freq,
    )[1:]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Historical line
    ax.plot(
        historical[date_col],
        historical[value_col],
        color="#4C72B0",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Historical",
    )

    # Forecast line
    ax.plot(
        future_dates,
        forecast_values,
        color="#DD8452",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
        label=f"Forecast ({model_used})",
    )

    # Confidence band (±10% simple estimate)
    forecast_arr = np.array(forecast_values)
    ax.fill_between(
        future_dates,
        forecast_arr * 0.90,
        forecast_arr * 1.10,
        alpha=0.2,
        color="#DD8452",
        label="±10% confidence band",
    )

    # Separator line
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    ax.set_title(
        f"Forecast – {value_col.replace('_', ' ').title()} (Model: {model_used})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_pie_chart(
    data: list[dict],
    label_col: str,
    value_col: str,
    title: str = "",
) -> BytesIO:
    """Generates a pie chart from query results."""
    df = pd.DataFrame(data)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    fig, ax = plt.subplots()
    ax.pie(
        df[value_col],
        labels=df[label_col].astype(str),
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def detect_chart_type(data: list[dict]) -> str:
    """
    Automatically detects the best chart type based on data structure.
    Returns: 'bar', 'line', 'pie'
    """
    if not data or len(data) == 0:
        return "bar"

    df = pd.DataFrame(data)
    cols = df.columns.tolist()

    if len(cols) < 2:
        return "bar"

    # Check if first column looks like a date
    try:
        pd.to_datetime(df[cols[0]])
        return "line"
    except Exception:
        pass

    # Many categories → bar, few → pie
    unique_vals = df[cols[0]].nunique()
    if unique_vals <= 6:
        return "pie"

    return "bar"


def render_chart(data: list[dict], title: str = "") -> BytesIO:
    """
    Auto-selects and renders the best chart type for the given data.
    Returns a BytesIO PNG buffer ready for Streamlit display.
    """
    if not data:
        return None

    chart_type = detect_chart_type(data)
    df = pd.DataFrame(data)
    cols = df.columns.tolist()

    if len(cols) < 2:
        return None

    x_col, y_col = cols[0], cols[1]

    if chart_type == "line":
        return plot_line_chart(data, x_col, y_col, title)
    elif chart_type == "pie":
        return plot_pie_chart(data, x_col, y_col, title)
    else:
        return plot_bar_chart(data, x_col, y_col, title)