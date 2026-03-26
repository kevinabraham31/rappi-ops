import pandas as pd
import pytest


@pytest.fixture
def df_metrics():
    rows = []
    zones = [
        ("CO", "Bogotá", "Facatativa", "Wealthy", "High Priority"),
        ("CO", "Bogotá", "Usme", "Non Wealthy", "Low Priority"),
        ("MX", "CDMX", "Polanco", "Wealthy", "High Priority"),
        ("MX", "CDMX", "Iztapalapa", "Non Wealthy", "Low Priority"),
        ("BR", "São Paulo", "Pinheiros", "Wealthy", "High Priority"),
    ]
    metrics = ["Lead Penetration", "Perfect Orders", "Gross Profit UE", "Turbo Adoption"]
    base_vals = {
        "Lead Penetration": [0.65, 0.40, 0.70, 0.45, 0.60],
        "Perfect Orders":   [0.85, 0.72, 0.90, 0.68, 0.80],
        "Gross Profit UE":  [1.20, 0.95, 1.35, 0.88, 1.10],
        "Turbo Adoption":   [0.30, 0.18, 0.35, 0.15, 0.28],
    }
    for mi, metric in enumerate(metrics):
        for zi, (country, city, zone, ztype, zprio) in enumerate(zones):
            l3 = base_vals[metric][zi]
            l2 = l3 * 0.97
            l1 = l2 * 0.96
            l0 = l1 * 0.95
            rows.append({
                "COUNTRY": country,
                "CITY": city,
                "ZONE": zone,
                "ZONE_TYPE": ztype,
                "ZONE_PRIORITIZATION": zprio,
                "METRIC": metric,
                "L3W_ROLL": round(l3, 4),
                "L2W_ROLL": round(l2, 4),
                "L1W_ROLL": round(l1, 4),
                "L0W_ROLL": round(l0, 4),
                "L0W": round(l0 * 0.98, 4),
                "L1W": round(l1 * 0.98, 4),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_metrics_anomaly():
    rows = [
        {
            "COUNTRY": "CO", "CITY": "Bogotá", "ZONE": "Facatativa",
            "ZONE_TYPE": "Wealthy", "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.90, "L2W_ROLL": 0.88, "L1W_ROLL": 0.85, "L0W_ROLL": 0.60,
            "L0W": 0.58, "L1W": 0.83,
        },
        {
            "COUNTRY": "MX", "CITY": "CDMX", "ZONE": "Polanco",
            "ZONE_TYPE": "Wealthy", "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Lead Penetration",
            "L3W_ROLL": 0.65, "L2W_ROLL": 0.66, "L1W_ROLL": 0.67, "L0W_ROLL": 0.68,
            "L0W": 0.66, "L1W": 0.65,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def df_metrics_deteriorating():
    rows = [
        {
            "COUNTRY": "CO", "CITY": "Bogotá", "ZONE": "Usme",
            "ZONE_TYPE": "Non Wealthy", "ZONE_PRIORITIZATION": "Low Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.82, "L2W_ROLL": 0.78, "L1W_ROLL": 0.74, "L0W_ROLL": 0.69,
            "L0W": 0.67, "L1W": 0.72,
        },
        {
            "COUNTRY": "MX", "CITY": "CDMX", "ZONE": "Iztapalapa",
            "ZONE_TYPE": "Non Wealthy", "ZONE_PRIORITIZATION": "Low Priority",
            "METRIC": "Gross Profit UE",
            "L3W_ROLL": 1.10, "L2W_ROLL": 1.05, "L1W_ROLL": 0.98, "L0W_ROLL": 0.88,
            "L0W": 0.86, "L1W": 0.96,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def df_orders():
    rows = [
        {"COUNTRY": "CO", "ZONE": "Facatativa",
         "L0W_ROLL": 1200, "L1W_ROLL": 1000, "L0W": 1180, "L1W": 980,
         "W1": 900, "W2": 950, "W3": 980, "W4": 1000, "W5": 1050, "W6": 1100, "W7": 1150, "W8": 1200},
        {"COUNTRY": "CO", "ZONE": "Usme",
         "L0W_ROLL":  800, "L1W_ROLL":  850, "L0W":  785, "L1W":  835,
         "W1": 870, "W2": 860, "W3": 855, "W4": 850, "W5": 845, "W6": 840, "W7": 830, "W8": 800},
        {"COUNTRY": "MX", "ZONE": "Polanco",
         "L0W_ROLL": 2200, "L1W_ROLL": 2100, "L0W": 2160, "L1W": 2060,
         "W1": 2000, "W2": 2050, "W3": 2080, "W4": 2100, "W5": 2120, "W6": 2150, "W7": 2180, "W8": 2200},
    ]
    return pd.DataFrame(rows)
