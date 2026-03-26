from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

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


@dataclass
class InsightReport:
    summary: list[dict]
    anomalies: list[dict]
    deteriorating: list[dict]
    benchmarking: list[dict]
    correlations: list[dict]
    opportunities: list[dict]


def _safe_pct_change(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    if abs(previous) < 0.01:
        return np.nan
    if (current > 0) != (previous > 0):
        return np.nan
    return ((current - previous) / abs(previous)) * 100


_METRIC_RECS: dict[str, str] = {
    "Perfect Orders": "Revisar tasa de cancelaciones, tiempos de entrega y calidad del surtido en esta zona.",
    "Lead Penetration": "Revisar el pipeline comercial y la tasa de conversión de prospectos a tiendas activas.",
    "Gross Profit UE": "Revisar descuentos aplicados, mezcla de categorías y costo logístico por orden en esta zona.",
    "Turbo Adoption": "Revisar la propuesta de valor y comunicación del servicio premium en esta zona.",
    "Pro Adoption": "Revisar la propuesta de valor y comunicación del servicio premium en esta zona.",
}
_GENERIC_REC = "Validar causa raíz operativa y comparar contra zonas pares del mismo país."


def _metric_rec(metric: str) -> str:
    return _METRIC_RECS.get(metric, _GENERIC_REC)


def _severity(change_pct: float | None = None) -> str:
    if change_pct is not None:
        a = abs(change_pct)
        if a > 50:
            return "crítico"
        if a >= 20:
            return "alto"
    return "medio"


def detect_anomalies(df_metrics: pd.DataFrame, df_orders: pd.DataFrame) -> list[dict]:
    """Detecta zonas con cambios WoW mayores al 10% en métricas y órdenes, ordenadas por magnitud."""
    df_metrics = df_metrics.drop_duplicates()
    df_orders = df_orders.drop_duplicates()
    anomalies = []
    for _, row in df_metrics.iterrows():
        pct = _safe_pct_change(row["L0W_ROLL"], row["L1W_ROLL"])
        if pd.notna(pct) and abs(pct) >= 10 and abs(pct) <= 200:
            anomalies.append({
                "category": "Métrica",
                "country": row["COUNTRY"],
                "zone": row["ZONE"],
                "metric": row["METRIC"],
                "change_pct": round(float(pct), 2),
                "insight": f"{row['METRIC']} cambió {pct:.1f}% WoW en {row['ZONE']} ({row['COUNTRY']}).",
                "recommendation": _metric_rec(row["METRIC"]),
                "severity": _severity(pct),
            })
    for _, row in df_orders.iterrows():
        pct = _safe_pct_change(row["L0W"], row["L1W"])
        if pd.notna(pct) and abs(pct) >= 10 and abs(pct) <= 200:
            anomalies.append({
                "category": "Orders",
                "country": row["COUNTRY"],
                "zone": row["ZONE"],
                "metric": "Orders",
                "change_pct": round(float(pct), 2),
                "insight": f"Orders cambió {pct:.1f}% WoW en {row['ZONE']} ({row['COUNTRY']}).",
                "recommendation": "Revisar demanda, capacidad y cambios comerciales recientes.",
                "severity": _severity(pct),
            })
    return sorted(anomalies, key=lambda item: abs(item["change_pct"]), reverse=True)[:12]


def detect_deteriorating_trends(df_metrics: pd.DataFrame) -> list[dict]:
    """Identifica zonas con 4 semanas consecutivas de caída en alguna métrica."""
    df_metrics = df_metrics.drop_duplicates()
    insights = []
    recent_cols = ["L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]
    for _, row in df_metrics.iterrows():
        values = [row[col] for col in recent_cols]
        if any(pd.isna(value) for value in values):
            continue
        if values[0] > values[1] > values[2] > values[3]:
            drop_pct = _safe_pct_change(values[-1], values[0])
            insights.append({
                "country": row["COUNTRY"],
                "zone": row["ZONE"],
                "metric": row["METRIC"],
                "drop_pct": round(float(drop_pct), 2) if pd.notna(drop_pct) else None,
                "insight": f"{row['METRIC']} lleva 4 semanas consecutivas de deterioro en {row['ZONE']}.",
                "recommendation": _metric_rec(row["METRIC"]),
                "severity": "crítico",
            })
    return sorted(insights, key=lambda item: item["drop_pct"] or 0)[:10]


def detect_benchmarking(df_metrics: pd.DataFrame) -> list[dict]:
    """Compara cada zona contra el promedio de zonas del mismo tipo y país; devuelve las con gap >15%."""
    df_metrics = df_metrics.drop_duplicates()
    current = df_metrics[["COUNTRY", "ZONE", "ZONE_TYPE", "METRIC", "L0W_ROLL"]].copy()
    current["peer_avg"] = current.groupby(["COUNTRY", "ZONE_TYPE", "METRIC"])["L0W_ROLL"].transform("mean")
    current["gap_vs_peer"] = current["L0W_ROLL"] - current["peer_avg"]
    benchmark = current[current["gap_vs_peer"].abs() > current["peer_avg"].abs() * 0.15].copy()
    result = []
    for _, row in benchmark.sort_values("gap_vs_peer").head(10).iterrows():
        result.append({
            "country": row["COUNTRY"],
            "zone": row["ZONE"],
            "zone_type": row["ZONE_TYPE"],
            "metric": row["METRIC"],
            "gap_vs_peer": round(float(row["gap_vs_peer"]), 4),
            "insight": f"{row['ZONE']} está {'por debajo' if row['gap_vs_peer'] < 0 else 'por encima'} del benchmark de zonas {row['ZONE_TYPE']} en {row['METRIC']}.",
            "recommendation": "Comparar playbooks operativos con zonas del mismo tipo y país.",
            "severity": "medio",
        })
    return result


def detect_correlations(df_metrics: pd.DataFrame) -> list[dict]:
    """Calcula correlaciones de Pearson entre métricas y agrega pares de negocio específicos con r > 0.3."""
    pivot = df_metrics.pivot_table(index=["COUNTRY", "CITY", "ZONE"], columns="METRIC", values="L0W_ROLL", aggfunc="mean")
    numeric = pivot.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr_matrix = numeric.corr()
    corr_matrix.index.name = "metric_a"
    corr_matrix.columns.name = "metric_b"
    corr = corr_matrix.stack().reset_index(name="corr")
    corr = corr[corr["metric_a"] < corr["metric_b"]]
    corr = corr[corr["corr"].abs() >= 0.35].sort_values("corr", key=lambda s: s.abs(), ascending=False)
    results = []
    for _, row in corr.head(6).iterrows():
        direction = "positiva" if row["corr"] > 0 else "negativa"
        results.append({
            "metric_a": row["metric_a"],
            "metric_b": row["metric_b"],
            "corr": round(float(row["corr"]), 3),
            "insight": f"Se observa una correlación {direction} entre {row['metric_a']} y {row['metric_b']}.",
            "recommendation": "Validar si esta relación puede convertirse en palanca de gestión para priorizar zonas.",
        })

    _BUSINESS_PAIRS = [
        ("Lead Penetration", "Perfect Orders",
         "Zonas con alta Lead Penetration y bajo Perfect Orders: oportunidad de mejorar ejecución operativa sin necesidad de atraer más tiendas.",
         "Investigar causas de bajo Perfect Order en zonas con buen pipeline comercial."),
        ("Lead Penetration", "Gross Profit UE",
         "Zonas con alta Lead Penetration y bajo Gross Profit UE: el crecimiento de tiendas no se está traduciendo en rentabilidad.",
         "Revisar mezcla de categorías, descuentos y costo logístico en zonas con buena penetración pero bajo margen."),
        ("Turbo Adoption", "Perfect Orders",
         "Correlación entre adopción del servicio Turbo y calidad operativa: mayor velocidad puede impactar la tasa de órdenes perfectas.",
         "Monitorear si las zonas con alto Turbo Adoption mantienen estándares de calidad operativa."),
    ]
    existing_pairs = {(r["metric_a"], r["metric_b"]) for r in results}
    for m_a, m_b, insight_text, rec_text in _BUSINESS_PAIRS:
        if len(results) >= 8:
            break
        if (m_a, m_b) in existing_pairs or (m_b, m_a) in existing_pairs:
            continue
        col_a = numeric.get(m_a)
        col_b = numeric.get(m_b)
        if col_a is None or col_b is None:
            continue
        valid = numeric[[m_a, m_b]].dropna()
        if len(valid) < 5:
            continue
        r_val = float(valid[m_a].corr(valid[m_b]))
        if abs(r_val) > 0.3:
            direction = "positiva" if r_val > 0 else "negativa"
            results.append({
                "metric_a": m_a,
                "metric_b": m_b,
                "corr": round(r_val, 3),
                "insight": insight_text,
                "recommendation": rec_text,
            })
    return results


def detect_opportunities(df_metrics: pd.DataFrame) -> list[dict]:
    """Detecta zonas con alta Lead Penetration pero bajo Perfect Orders o bajo Gross Profit UE."""
    df_metrics = df_metrics.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE", "METRIC"])
    lp = df_metrics[df_metrics["METRIC"] == "Lead Penetration"][["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].rename(columns={"L0W_ROLL": "lead_penetration"})
    po = df_metrics[df_metrics["METRIC"] == "Perfect Orders"][["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].rename(columns={"L0W_ROLL": "perfect_orders"})
    gp = df_metrics[df_metrics["METRIC"] == "Gross Profit UE"][["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].rename(columns={"L0W_ROLL": "gross_profit_ue"})
    merged = lp.merge(po, on=["COUNTRY", "CITY", "ZONE"], how="inner").merge(gp, on=["COUNTRY", "CITY", "ZONE"], how="inner")
    if merged.empty:
        return []
    lp_q = merged["lead_penetration"].quantile(0.7)
    po_q = merged["perfect_orders"].quantile(0.3)
    gp_q = merged["gross_profit_ue"].quantile(0.3)
    opps = merged[(merged["lead_penetration"] >= lp_q) & ((merged["perfect_orders"] <= po_q) | (merged["gross_profit_ue"] <= gp_q))]
    results = []
    for _, row in opps.sort_values(["perfect_orders", "gross_profit_ue"]).head(10).iterrows():
        results.append({
            "country": row["COUNTRY"],
            "zone": row["ZONE"],
            "insight": f"{row['ZONE']} tiene buena penetración de leads pero espacio de mejora en ejecución o rentabilidad.",
            "recommendation": "Priorizar quick wins operativos: calidad, tiempos, surtido o monetización según diagnóstico local.",
            "severity": "medio",
        })
    return results


def generate_report(df_metrics: pd.DataFrame, df_orders: pd.DataFrame) -> InsightReport:
    """Ejecuta todos los detectores y construye el InsightReport con resumen ejecutivo priorizado."""
    anomalies = detect_anomalies(df_metrics, df_orders)
    deteriorating = detect_deteriorating_trends(df_metrics)
    benchmarking = detect_benchmarking(df_metrics)
    correlations = detect_correlations(df_metrics)
    opportunities = detect_opportunities(df_metrics)

    summary_items: list[dict] = []
    if anomalies:
        summary_items.append(max(anomalies, key=lambda x: abs(x["change_pct"])))
    if deteriorating:
        summary_items.append(deteriorating[0])
    neg_bench = [b for b in benchmarking if b["gap_vs_peer"] < 0]
    if neg_bench:
        summary_items.append(min(neg_bench, key=lambda x: x["gap_vs_peer"]))
    if correlations:
        summary_items.append(max(correlations, key=lambda x: abs(x["corr"])))
    if opportunities:
        summary_items.append(opportunities[0])
    if len(summary_items) < 5:
        for pool in [anomalies, deteriorating, benchmarking, correlations, opportunities]:
            for item in pool:
                if len(summary_items) >= 5:
                    break
                if item not in summary_items:
                    summary_items.append(item)

    summary = [
        {
            "title": item.get("metric", item.get("zone", "Insight")),
            "insight": item["insight"],
            "recommendation": item["recommendation"],
            "severity": item.get("severity", "medio"),
        }
        for item in summary_items[:5]
    ]

    return InsightReport(summary, anomalies, deteriorating, benchmarking, correlations, opportunities)


def _badge(text: str, color: str = "#374151") -> str:
    return f"<span style='display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.7rem;font-weight:600;background:{color};color:#fff;margin-right:4px;'>{text}</span>"


def _change_badge(pct: float | None) -> str:
    if pct is None:
        return ""
    color = "#16a34a" if pct > 0 else "#dc2626"
    sign = "+" if pct > 0 else ""
    return _badge(f"{sign}{pct:.1f}%", color)


_SEV_COLORS = {"crítico": "#dc2626", "alto": "#f97316", "medio": "#eab308"}


def _drill_down_url(item: dict) -> str:
    from urllib.parse import quote
    zone = item.get("zone", "")
    country = item.get("country", "")
    metric = item.get("metric", "")
    if not zone:
        return ""
    if "change_pct" in item and metric:
        q = f"Muestra la evolución de {metric} en {zone} las últimas 8 semanas"
    elif "drop_pct" in item and metric:
        q = f"¿Cuál es la tendencia de {metric} en {zone}?"
    elif "gap_vs_peer" in item:
        q = f"Compara {zone} contra zonas similares en {country}"
    else:
        q = f"Analiza Lead Penetration y Perfect Orders en {zone}"
    return f"/?q={quote(q)}"


def _render_cards(items: list[dict], empty_text: str, category_color: str = "#374151") -> str:
    if not items:
        return f'<div class="empty">{empty_text}</div>'
    cards = []
    for item in items:
        zone = item.get("zone", "")
        country = item.get("country", "")
        metric = item.get("metric", "")
        change_pct = item.get("change_pct")
        drop_pct = item.get("drop_pct")
        corr = item.get("corr")
        sev = item.get("severity", "")

        badges = ""
        if country:
            badges += _badge(country, "#1d4ed8")
        if zone:
            badges += _badge(zone, "#374151")
        if metric:
            badges += _badge(metric, "#6d28d9")
        if change_pct is not None:
            badges += _change_badge(change_pct)
        elif drop_pct is not None:
            badges += _change_badge(drop_pct)
        if corr is not None:
            corr_color = "#0369a1" if corr > 0 else "#9a3412"
            badges += _badge(f"r={corr:.2f}", corr_color)

        dot_color = _SEV_COLORS.get(sev, "")
        sev_dot = f"<span style='position:absolute;top:10px;right:10px;width:9px;height:9px;border-radius:50%;background:{dot_color};' title='Severidad: {sev}'></span>" if dot_color else ""

        drill_url = _drill_down_url(item)
        drill_link = (
            f"<a href='{drill_url}' style='display:inline-block;margin-top:8px;font-size:0.78rem;"
            f"color:#ff4d1f;text-decoration:none;font-weight:600;'>Analizar en el chat &#8594;</a>"
        ) if drill_url else ""

        cards.append(
            f"""<article class='card' style='position:relative;'>
              {sev_dot}
              <div class='card-badges'>{badges}</div>
              <p class='card-title'>{item.get('insight', '')}</p>
              <p class='card-rec'><strong>Acción:</strong> {item.get('recommendation', '')}</p>
              {drill_link}
            </article>"""
        )
    return "\n".join(cards)


def render_report_html(
    report: InsightReport,
    df_metrics: pd.DataFrame | None = None,
    df_orders: pd.DataFrame | None = None,
) -> str:
    """Renderiza el InsightReport como HTML standalone con KPIs, charts Chart.js y estilos de impresión."""
    from datetime import datetime

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    total = (
        len(report.anomalies) + len(report.deteriorating)
        + len(report.benchmarking) + len(report.opportunities)
    )

    # ── KPI 1: % zonas con anomalías ─────────────────────────────────────────
    anomaly_zones = {a["zone"] for a in report.anomalies if a.get("zone")}
    if df_metrics is not None and not df_metrics.empty:
        total_zones = int(df_metrics["ZONE"].nunique())
        pct_anom = round(len(anomaly_zones) / total_zones * 100, 1) if total_zones else 0
        kpi_zones_label = f"{pct_anom}% zonas"
    else:
        kpi_zones_label = f"{len(anomaly_zones)} zonas"

    # ── KPI 2: país con más deterioraciones ──────────────────────────────────
    det_countries = [d["country"] for d in report.deteriorating if d.get("country")]
    kpi_top_country = Counter(det_countries).most_common(1)[0][0] if det_countries else "N/D"

    # ── KPI 3: métrica más problemática ──────────────────────────────────────
    metric_pool = (
        [a["metric"] for a in report.anomalies if a.get("metric")]
        + [d["metric"] for d in report.deteriorating if d.get("metric")]
    )
    kpi_top_metric = Counter(metric_pool).most_common(1)[0][0] if metric_pool else "N/D"

    # ── KPI 4: oportunidades de alto impacto ─────────────────────────────────
    kpi_opps = len(report.opportunities)

    # ── Chart 1 data: anomalías por país ─────────────────────────────────────
    c1_counts = Counter(a["country"] for a in report.anomalies if a.get("country"))
    c1_labels = json.dumps(list(c1_counts.keys()))
    c1_data = json.dumps(list(c1_counts.values()))

    # ── Chart 2 data: top 10 deterioro por zona ───────────────────────────────
    det_pairs = [
        (d["zone"], d.get("drop_pct") or 0)
        for d in report.deteriorating[:10]
        if d.get("zone")
    ]
    c2_labels = json.dumps([p[0] for p in det_pairs])
    c2_data = json.dumps([p[1] for p in det_pairs])

    # ── Chart 3 data: correlaciones entre métricas ────────────────────────────
    corr_pairs = [
        (f"{c['metric_a']} / {c['metric_b']}", c["corr"])
        for c in report.correlations
    ]
    c3_labels = json.dumps([p[0] for p in corr_pairs])
    c3_data = json.dumps([p[1] for p in corr_pairs])
    c3_colors = json.dumps(
        ["rgba(22,163,74,0.75)" if p[1] > 0 else "rgba(220,38,38,0.75)" for p in corr_pairs]
    )

    summary_html       = _render_cards(report.summary,       "No se encontraron hallazgos críticos.")
    anomalies_html     = _render_cards(report.anomalies,     "No se detectaron anomalías >10%.",                       "#dc2626")
    deteriorating_html = _render_cards(report.deteriorating, "No se detectaron deterioros de 4 semanas consecutivas.", "#b45309")
    benchmarking_html  = _render_cards(report.benchmarking,  "No se detectaron gaps materiales versus benchmark.",     "#1d4ed8")
    correlations_html  = _render_cards(report.correlations,  "No se encontraron correlaciones ≥0.35.",                 "#0369a1")
    opportunities_html = _render_cards(report.opportunities, "No se detectaron oportunidades con las reglas actuales.", "#16a34a")

    return f"""<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='UTF-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>
  <title>Rappi · Reporte Ejecutivo</title>
  <link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap' rel='stylesheet'/>
  <style>
    :root {{
      --bg:#0a0a0b; --surface:#111113; --surface2:#18181b; --border:#27272a;
      --text:#fafafa; --muted:#71717a; --accent:#ff4d1f;
      --radius:12px; --sans:'IBM Plex Sans',sans-serif;
    }}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;}}
    .page{{max-width:1280px;margin:0 auto;padding:28px 24px;}}
    .hero{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid var(--border);flex-wrap:wrap;}}
    .hero-left h1{{font-size:1.75rem;font-weight:700;margin-bottom:6px;}}
    .hero-left p{{color:var(--muted);max-width:680px;font-size:0.9rem;}}
    .hero-meta{{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;}}
    .meta-chip{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:6px 12px;font-size:0.78rem;color:var(--muted);}}
    .meta-chip strong{{color:var(--text);}}
    .kpi-row{{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;}}
    .kpi-card{{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 16px;min-width:150px;}}
    .kpi-value{{font-size:1.25rem;font-weight:700;color:var(--accent);line-height:1.2;word-break:break-word;}}
    .kpi-label{{font-size:0.7rem;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.05em;}}
    .hero-actions{{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;}}
    .btn{{text-decoration:none;padding:9px 16px;border-radius:9px;font-size:0.82rem;font-weight:600;transition:all .15s;cursor:pointer;border:none;display:inline-block;}}
    .btn-primary{{background:var(--accent);color:#fff;}}
    .btn-primary:hover{{opacity:.85;}}
    .btn-secondary{{background:var(--surface2);border:1px solid var(--border);color:var(--muted);}}
    .btn-secondary:hover{{color:var(--text);border-color:#52525b;}}
    .section{{margin-bottom:32px;}}
    .section-header{{display:flex;align-items:center;gap:10px;margin-bottom:14px;}}
    .section-title{{font-size:1rem;font-weight:600;}}
    .section-count{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:2px 8px;font-size:0.75rem;color:var(--muted);}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;transition:border-color .15s;}}
    .card:hover{{border-color:#3f3f46;}}
    .card-badges{{margin-bottom:10px;display:flex;flex-wrap:wrap;gap:4px;}}
    .card-title{{font-size:0.88rem;color:var(--text);line-height:1.55;margin-bottom:10px;}}
    .card-rec{{font-size:0.82rem;color:var(--muted);line-height:1.5;border-top:1px solid var(--border);padding-top:10px;margin-top:4px;}}
    .card-rec strong{{color:#a1a1aa;}}
    .empty{{background:var(--surface);border:1px dashed var(--border);border-radius:var(--radius);padding:20px;color:var(--muted);font-size:0.88rem;text-align:center;}}
    .charts-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;}}
    .chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;}}
    .chart-card h3{{font-size:0.78rem;font-weight:600;color:var(--muted);margin-bottom:14px;text-transform:uppercase;letter-spacing:.06em;}}
    .chart-wrap{{position:relative;height:220px;}}
    ::-webkit-scrollbar{{width:5px;}} ::-webkit-scrollbar-track{{background:transparent;}} ::-webkit-scrollbar-thumb{{background:#3f3f46;border-radius:4px;}}
    @media print {{
      body{{ background:#fff !important; color:#111 !important; }}
      .page{{ padding:12px; max-width:100%; }}
      .no-print{{ display:none !important; }}
      .hero{{ border-bottom:1px solid #d1d5db; }}
      h1{{ color:#111 !important; font-size:1.3rem; }}
      .hero-left p{{ color:#374151 !important; }}
      .section-title{{ color:#111 !important; }}
      .section-count{{ background:#f3f4f6 !important; border:1px solid #d1d5db !important; color:#374151 !important; }}
      .meta-chip{{ background:#f9fafb !important; border:1px solid #d1d5db !important; color:#374151 !important; }}
      .meta-chip strong{{ color:#111 !important; }}
      .kpi-card{{ background:#f9fafb !important; border:1px solid #d1d5db !important; }}
      .kpi-value{{ color:#dc2626 !important; }}
      .kpi-label{{ color:#374151 !important; }}
      .card{{ background:#fff !important; border:1px solid #d1d5db !important; break-inside:avoid; page-break-inside:avoid; margin-bottom:8px; }}
      .card-title{{ color:#111 !important; }}
      .card-rec{{ color:#374151 !important; border-top:1px solid #e5e7eb !important; }}
      .card-rec strong{{ color:#374151 !important; }}
      .empty{{ background:#f9fafb !important; border:1px dashed #d1d5db !important; color:#374151 !important; }}
      .grid{{ display:block; }}
    }}
  </style>
</head>
<body>
<div class='page'>
  <div class='hero'>
    <div class='hero-left'>
      <h1>&#128202; Reporte Ejecutivo Automático</h1>
      <p>Anomalías, deterioros, benchmarking, correlaciones y oportunidades detectadas automáticamente sobre los datos operacionales de Rappi.</p>
      <div class='hero-meta'>
        <div class='meta-chip'>Generado: <strong>{generated_at}</strong></div>
        <div class='meta-chip'>Hallazgos totales: <strong>{total}</strong></div>
        <div class='meta-chip'>Anomalías: <strong>{len(report.anomalies)}</strong></div>
        <div class='meta-chip'>Deterioros: <strong>{len(report.deteriorating)}</strong></div>
        <div class='meta-chip'>Oportunidades: <strong>{len(report.opportunities)}</strong></div>
      </div>
      <div class='kpi-row'>
        <div class='kpi-card'>
          <div class='kpi-value'>{kpi_zones_label}</div>
          <div class='kpi-label'>Con anomalías detectadas</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-value'>{kpi_top_country}</div>
          <div class='kpi-label'>País con más deterioraciones</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-value'>{kpi_top_metric}</div>
          <div class='kpi-label'>Métrica más problemática</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-value'>{kpi_opps}</div>
          <div class='kpi-label'>Oportunidades de alto impacto</div>
        </div>
      </div>
    </div>
    <div class='hero-actions no-print'>
      <a class='btn btn-secondary' href='/'>&#8592; Chat</a>
      <a class='btn btn-secondary' href='/excel'>Dataset</a>
      <a class='btn btn-primary' href='/insights/download'>&#11015; Descargar HTML</a>
      <button class='btn btn-secondary' onclick='window.print()'>&#128438; Imprimir / PDF</button>
    </div>
  </div>

  <section class='section no-print'>
    <div class='section-header'>
      <span class='section-title'>&#128202; Visualizaciones</span>
    </div>
    <div class='charts-grid'>
      <div class='chart-card'>
        <h3>Anomalías por País</h3>
        <div class='chart-wrap'><canvas id='chartAnomCountry'></canvas></div>
      </div>
      <div class='chart-card'>
        <h3>Top 10 Deterioro de Zonas</h3>
        <div class='chart-wrap'><canvas id='chartDetZones'></canvas></div>
      </div>
      <div class='chart-card'>
        <h3>Correlaciones entre Métricas</h3>
        <div class='chart-wrap'><canvas id='chartCorr'></canvas></div>
      </div>
    </div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#127381; Resumen ejecutivo</span>
      <span class='section-count'>{len(report.summary)}</span>
    </div>
    <div class='grid'>{summary_html}</div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#9888;&#65039; Anomalías WoW</span>
      <span class='section-count'>{len(report.anomalies)}</span>
    </div>
    <div class='grid'>{anomalies_html}</div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#128200; Tendencias preocupantes</span>
      <span class='section-count'>{len(report.deteriorating)}</span>
    </div>
    <div class='grid'>{deteriorating_html}</div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#127959;&#65039; Benchmarking</span>
      <span class='section-count'>{len(report.benchmarking)}</span>
    </div>
    <div class='grid'>{benchmarking_html}</div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#128279; Correlaciones</span>
      <span class='section-count'>{len(report.correlations)}</span>
    </div>
    <div class='grid'>{correlations_html}</div>
  </section>

  <section class='section'>
    <div class='section-header'>
      <span class='section-title'>&#128161; Oportunidades</span>
      <span class='section-count'>{len(report.opportunities)}</span>
    </div>
    <div class='grid'>{opportunities_html}</div>
  </section>
</div>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<script>
(function() {{
  const gridColor = 'rgba(255,255,255,0.07)';
  const tickColor = '#71717a';
  Chart.defaults.font = {{ family: "'IBM Plex Sans',sans-serif", size: 11 }};
  Chart.defaults.color = tickColor;

  // Chart 1 – Anomalías por país
  const c1Labels = {c1_labels};
  const c1Data   = {c1_data};
  if (c1Labels.length) {{
    new Chart(document.getElementById('chartAnomCountry'), {{
      type: 'bar',
      data: {{
        labels: c1Labels,
        datasets: [{{ label: 'Anomalías', data: c1Data,
          backgroundColor: 'rgba(255,77,31,0.75)', borderColor: '#ff4d1f',
          borderWidth: 1, borderRadius: 4 }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
          y: {{ ticks: {{ color: tickColor, precision: 0 }}, grid: {{ color: gridColor }} }}
        }}
      }}
    }});
  }}

  // Chart 2 – Top 10 deterioro de zonas (horizontal)
  const c2Labels = {c2_labels};
  const c2Data   = {c2_data};
  if (c2Labels.length) {{
    new Chart(document.getElementById('chartDetZones'), {{
      type: 'bar',
      data: {{
        labels: c2Labels,
        datasets: [{{ label: 'Caída %', data: c2Data,
          backgroundColor: 'rgba(249,115,22,0.75)', borderColor: '#f97316',
          borderWidth: 1, borderRadius: 4 }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
          y: {{ ticks: {{ color: tickColor }} }}
        }}
      }}
    }});
  }}

  // Chart 3 – Correlaciones entre métricas (horizontal)
  const c3Labels = {c3_labels};
  const c3Data   = {c3_data};
  const c3Colors = {c3_colors};
  if (c3Labels.length) {{
    new Chart(document.getElementById('chartCorr'), {{
      type: 'bar',
      data: {{
        labels: c3Labels,
        datasets: [{{ label: 'r', data: c3Data,
          backgroundColor: c3Colors, borderColor: c3Colors,
          borderWidth: 1, borderRadius: 4 }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ min: -1, max: 1, ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
          y: {{ ticks: {{ color: tickColor }} }}
        }}
      }}
    }});
  }}
}})();
</script>
</body>
</html>"""
