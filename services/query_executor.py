from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_WEEK_COLUMNS = [
    "L8W_ROLL",
    "L7W_ROLL",
    "L6W_ROLL",
    "L5W_ROLL",
    "L4W_ROLL",
    "L3W_ROLL",
    "L2W_ROLL",
    "L1W_ROLL",
    "L0W_ROLL",
]

ORDER_WEEK_COLUMNS = ["L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W"]


class QueryExecutionError(Exception):
    pass


def extract_code(text: str) -> str | None:
    import re

    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def clean_response(text: str) -> str:
    import re

    return re.sub(r"```python.*?```", "", text, flags=re.DOTALL).strip()


def _fig_to_base64() -> str:
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="#111111")
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_line_chart(df: pd.DataFrame) -> dict[str, str] | None:
    week_cols = [col for col in METRIC_WEEK_COLUMNS + ORDER_WEEK_COLUMNS if col in df.columns]
    if len(week_cols) < 3 or df.empty:
        return None
    numeric_df = df[week_cols].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#111111")
    ax.set_facecolor("#111111")

    colors = ["#ff5a36", "#fca5a5", "#f97316", "#fb923c"]
    for i, (_, row) in enumerate(numeric_df.head(4).iterrows()):
        values = row.values
        color = colors[i % len(colors)]
        ax.plot(range(len(values)), values, color=color,
                linewidth=2.5 if i == 0 else 1.5,
                marker="o", markersize=5 if i == 0 else 3,
                alpha=1.0 if i == 0 else 0.6)
        ax.annotate(f"{values[-1]:.3f}", xy=(len(values) - 1, values[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=7, va="center")

    labels = [col.replace("_ROLL", "").replace("_", " ") for col in week_cols]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="#9ca3af", fontsize=8)
    ax.tick_params(axis="y", colors="#9ca3af", labelsize=8)
    ax.grid(color="#2a2a2a", linestyle="--", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")

    first_valid = numeric_df.iloc[0].dropna()
    trend = "\u2191" if len(first_valid) >= 2 and first_valid.iloc[-1] > first_valid.iloc[0] else "\u2193"
    ax.set_title(f"Evoluci\u00f3n temporal  {trend}", color="#ffffff", fontsize=11, pad=10)

    plt.tight_layout()
    return {"type": "image", "title": "Tendencia semanal", "image_base64": _fig_to_base64()}


def _build_bar_chart(df: pd.DataFrame) -> dict[str, str] | None:
    if df.empty or len(df.columns) < 2:
        return None

    category_col = next((col for col in df.columns if df[col].dtype == object), None)
    value_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
    if not category_col or not value_col:
        return None

    sample = df[[category_col, value_col]].dropna().head(10)
    if sample.empty:
        return None

    sample = sample.sort_values(value_col, ascending=True)
    labels = sample[category_col].astype(str).tolist()
    values = sample[value_col].tolist()
    max_val = max(values) if values else 1

    bar_colors = [
        f"#{int(255 * (v / max_val)):02x}{int(90 * (1 - v / max_val)):02x}36"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.45)), facecolor="#111111")
    ax.set_facecolor("#111111")
    bars = ax.barh(labels, values, color=bar_colors, height=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="right", va="center", color="#fff", fontsize=8, fontweight="bold")

    ax.set_xlabel(value_col.replace("_", " "), color="#9ca3af", fontsize=8)
    ax.tick_params(axis="y", colors="#d1d5db", labelsize=8)
    ax.tick_params(axis="x", colors="#9ca3af", labelsize=7)
    ax.grid(axis="x", color="#2a2a2a", linestyle="--", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    ax.set_title("Comparaci\u00f3n por zona", color="#ffffff", fontsize=11, pad=10)

    plt.tight_layout()
    return {"type": "image", "title": "Ranking comparativo", "image_base64": _fig_to_base64()}


def build_chart_payload(result: Any) -> dict[str, str] | None:
    """Genera un payload de gráfico (línea o barra) a partir de un DataFrame; devuelve None si no aplica."""
    if isinstance(result, pd.DataFrame):
        line_chart = _build_line_chart(result)
        if line_chart:
            return line_chart
        return _build_bar_chart(result)
    return None


def _to_display_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _series_to_display_table(series: pd.Series) -> str:
    try:
        return series.to_markdown()
    except Exception:
        return series.to_string()


def _prettify_label(label: str) -> str:
    return str(label).replace("_", " ").strip().title()


def _build_highlights(data_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, str]]:
    if not data_rows:
        return []

    highlights = []

    numeric_columns = []
    for column in columns:
        for row in data_rows:
            value = row.get(column)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_columns.append(column)
                break

    best_row = data_rows[0]
    if numeric_columns:
        metric_column = numeric_columns[-1]
        best_row = max(
            (row for row in data_rows if isinstance(row.get(metric_column), (int, float)) and not isinstance(row.get(metric_column), bool)),
            key=lambda row: row.get(metric_column, float("-inf")),
            default=data_rows[0],
        )

    if "ZONE" in best_row:
        location_parts = [best_row.get(key) for key in ["ZONE", "CITY", "COUNTRY"] if best_row.get(key)]
        if location_parts:
            highlights.append({"label": "Zona destacada", "value": " · ".join(str(item) for item in location_parts)})

    if numeric_columns:
        metric_column = numeric_columns[-1]
        metric_value = best_row.get(metric_column)
        if metric_value is not None:
            highlights.append({"label": _prettify_label(metric_column), "value": f"{metric_value}"})

    highlights.append({"label": "Filas encontradas", "value": str(len(data_rows))})
    return highlights[:3]


def format_result_payload(result: Any) -> dict[str, Any]:
    """Convierte el resultado de una consulta (DataFrame, Series o escalar) al dict estándar de respuesta."""
    if result is None:
        return {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "La consulta no produjo un resultado."}

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return {"success": False, "data_markdown": "No se encontraron datos para esta consulta.", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "empty"}
        if list(result.columns) == ["Error"]:
            err_msg = str(result.iloc[0]["Error"]) if len(result) > 0 else "error"
            return {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": err_msg}
        formatted = result.copy()
        for column in formatted.select_dtypes(include=[float]).columns:
            formatted[column] = formatted[column].round(4)
        data_rows = formatted.head(50).replace({np.nan: None}).to_dict(orient="records")
        columns = formatted.columns.tolist()
        return {
            "success": True,
            "data_markdown": _to_display_table(formatted),
            "data_rows": data_rows,
            "columns": columns,
            "highlights": _build_highlights(data_rows, columns),
            "chart": build_chart_payload(formatted),
            "error": None,
        }

    if isinstance(result, pd.Series):
        rounded = result.round(4)
        data_rows = [{"index": str(index), "value": value} for index, value in rounded.items()]
        return {
            "success": True,
            "data_markdown": _series_to_display_table(rounded),
            "data_rows": data_rows,
            "columns": ["index", "value"],
            "highlights": _build_highlights(data_rows, ["index", "value"]),
            "chart": None,
            "error": None,
        }

    if isinstance(result, str):
        return {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": result}

    return {
        "success": True,
        "data_markdown": str(result),
        "data_rows": [{"value": str(result)}],
        "columns": ["value"],
        "highlights": [{"label": "Resultado", "value": str(result)}],
        "chart": None,
        "error": None,
    }


def run_code(code: str, df_metrics: pd.DataFrame, df_orders: pd.DataFrame) -> dict[str, Any]:
    """Ejecuta código Python generado por el LLM en un sandbox con df_metrics y df_orders disponibles."""
    local_ns = {
        "df_metrics": df_metrics.copy(),
        "df_orders": df_orders.copy(),
        "pd": pd,
        "np": np,
        "result": None,
    }
    try:
        exec(code, {"__builtins__": __builtins__}, local_ns)
        return format_result_payload(local_ns.get("result"))
    except Exception as exc:
        return {
            "success": False,
            "data_markdown": f"Error: {exc}",
            "data_rows": [],
            "columns": [],
            "highlights": [],
            "chart": None,
            "error": str(exc),
        }
