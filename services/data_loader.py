from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataBundle:
    base_dir: str
    data_path: str
    df_metrics: pd.DataFrame
    df_orders: pd.DataFrame
    metrics_catalog: list[str]
    countries: list[str]


WEEK_VALUE_ALIASES = {
    "L8W_VALUE": "L8W_ROLL",
    "L7W_VALUE": "L7W_ROLL",
    "L6W_VALUE": "L6W_ROLL",
    "L5W_VALUE": "L5W_ROLL",
    "L4W_VALUE": "L4W_ROLL",
    "L3W_VALUE": "L3W_ROLL",
    "L2W_VALUE": "L2W_ROLL",
    "L1W_VALUE": "L1W_ROLL",
    "L0W_VALUE": "L0W_ROLL",
}


EXPECTED_METRIC_COLUMNS = [
    "COUNTRY",
    "CITY",
    "ZONE",
    "ZONE_TYPE",
    "ZONE_PRIORITIZATION",
    "METRIC",
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


EXPECTED_ORDER_COLUMNS = [
    "COUNTRY",
    "CITY",
    "ZONE",
    "L8W",
    "L7W",
    "L6W",
    "L5W",
    "L4W",
    "L3W",
    "L2W",
    "L1W",
    "L0W",
]


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=WEEK_VALUE_ALIASES).copy()
    if "Pro Adoption (Last Week Status)" in renamed["METRIC"].unique():
        renamed["METRIC"] = renamed["METRIC"].replace(
            {"Pro Adoption (Last Week Status)": "Pro Adoption"}
        )
    return renamed


def _validate_columns(df: pd.DataFrame, expected: list[str], dataset_name: str) -> None:
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {dataset_name}: {', '.join(missing)}")


def load_data(base_dir: str) -> DataBundle:
    data_path = os.path.join(base_dir, "data", "Rappi Operations Analysis Dummy Data.xlsx")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}")

    df_metrics = pd.read_excel(data_path, sheet_name="RAW_INPUT_METRICS")
    df_orders = pd.read_excel(data_path, sheet_name="RAW_ORDERS")

    df_metrics = _normalize_metric_columns(df_metrics)
    _validate_columns(df_metrics, EXPECTED_METRIC_COLUMNS, "RAW_INPUT_METRICS")
    _validate_columns(df_orders, EXPECTED_ORDER_COLUMNS, "RAW_ORDERS")

    df_metrics["COUNTRY"] = df_metrics["COUNTRY"].astype(str).str.upper()
    df_orders["COUNTRY"] = df_orders["COUNTRY"].astype(str).str.upper()

    metrics_catalog = sorted(df_metrics["METRIC"].dropna().astype(str).unique().tolist())
    countries = sorted(df_metrics["COUNTRY"].dropna().astype(str).unique().tolist())

    return DataBundle(
        base_dir=base_dir,
        data_path=data_path,
        df_metrics=df_metrics,
        df_orders=df_orders,
        metrics_catalog=metrics_catalog,
        countries=countries,
    )
