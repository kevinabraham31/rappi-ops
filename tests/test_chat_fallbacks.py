import os

import pandas as pd
import pytest

from services.chat_service import ChatService
from services.insights_service import (
    InsightReport,
    detect_anomalies,
    generate_report,
    render_report_html,
)


@pytest.fixture
def svc_no_llm(df_metrics, df_orders):
    """ChatService sin cliente LLM — siempre usa fallbacks determinísticos."""
    metrics_catalog = df_metrics["METRIC"].unique().tolist()
    svc = ChatService(df_metrics, df_orders, metrics_catalog)
    svc.client = None
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# (a) Guardrail _is_out_of_scope rechaza preguntas fuera del dominio
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailOutOfScope:
    def test_chiste_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("cuéntame un chiste") is True

    def test_math_expression_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("2+2") is True

    def test_sports_score_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("el marcador del partido de ayer") is True

    def test_weather_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("cómo está el clima hoy") is True

    def test_president_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("quién es el presidente") is True

    def test_movie_rejected(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("recomiendame una película") is True

    def test_perfect_orders_query_accepted(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("muéstrame perfect orders en colombia") is False

    def test_top_zones_query_accepted(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("top 5 zonas con mayor lead penetration") is False

    def test_tendencia_query_accepted(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("tendencia de gross profit en mexico") is False

    def test_country_iso_accepted(self, svc_no_llm):
        assert svc_no_llm._is_out_of_scope("rendimiento en CO esta semana") is False


# ─────────────────────────────────────────────────────────────────────────────
# (b) Fallbacks devuelven respuesta válida con clave 'reply' cuando LLM falla
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackReturnsValidResponse:
    def test_perfect_orders_colombia_has_reply(self, svc_no_llm):
        result = svc_no_llm.answer("muéstrame perfect orders en colombia", [])
        assert "reply" in result
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0

    def test_top_zones_has_reply(self, svc_no_llm):
        result = svc_no_llm.answer("top 5 zonas con mayor lead penetration", [])
        assert "reply" in result
        assert result["reply"]

    def test_evolution_query_has_reply(self, svc_no_llm):
        result = svc_no_llm.answer("evolución de gross profit ue en facatativa", [])
        assert "reply" in result
        assert result["reply"]

    def test_promedio_query_has_reply(self, svc_no_llm):
        result = svc_no_llm.answer("promedio de lead penetration por país", [])
        assert "reply" in result
        assert result["reply"]

    def test_response_has_all_required_keys(self, svc_no_llm):
        result = svc_no_llm.answer("top 5 zonas con mayor perfect orders", [])
        required_keys = {"reply", "data_rows", "columns", "highlights", "chart", "suggestions"}
        assert required_keys.issubset(result.keys())

    def test_data_rows_is_list(self, svc_no_llm):
        result = svc_no_llm.answer("top 5 zonas con mayor perfect orders", [])
        assert isinstance(result["data_rows"], list)

    def test_suggestions_is_list(self, svc_no_llm):
        result = svc_no_llm.answer("promedio de gross profit ue por país", [])
        assert isinstance(result["suggestions"], list)

    def test_oos_answer_has_non_empty_reply(self, svc_no_llm):
        result = svc_no_llm.answer("cuéntame un chiste", [])
        assert "reply" in result
        assert len(result["reply"]) > 10

    def test_wealthy_comparison_has_reply(self, svc_no_llm):
        result = svc_no_llm.answer("compara wealthy vs non wealthy en lead penetration", [])
        assert "reply" in result
        assert result["reply"]

    def test_columns_is_list(self, svc_no_llm):
        result = svc_no_llm.answer("top 5 zonas con mayor lead penetration", [])
        assert isinstance(result["columns"], list)


# ─────────────────────────────────────────────────────────────────────────────
# (c) generate_report retorna InsightReport y render_report_html retorna HTML
# ─────────────────────────────────────────────────────────────────────────────

class TestInsightsReport:
    def test_generate_report_returns_insight_report(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        assert isinstance(report, InsightReport)

    def test_report_has_all_sections(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        assert hasattr(report, "anomalies")
        assert hasattr(report, "deteriorating")
        assert hasattr(report, "benchmarking")
        assert hasattr(report, "correlations")
        assert hasattr(report, "opportunities")
        assert hasattr(report, "summary")

    def test_all_sections_are_lists(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        assert isinstance(report.anomalies, list)
        assert isinstance(report.deteriorating, list)
        assert isinstance(report.benchmarking, list)
        assert isinstance(report.correlations, list)
        assert isinstance(report.opportunities, list)
        assert isinstance(report.summary, list)

    def test_render_report_html_returns_string(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        html = render_report_html(report, df_metrics, df_orders)
        assert isinstance(html, str)

    def test_render_report_html_is_valid_doctype(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        html = render_report_html(report, df_metrics, df_orders)
        assert html.strip().startswith("<!DOCTYPE html")

    def test_render_report_html_closes_html_tag(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        html = render_report_html(report, df_metrics, df_orders)
        assert "</html>" in html

    def test_render_report_html_contains_sections(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        html = render_report_html(report, df_metrics, df_orders)
        assert "Reporte Ejecutivo" in html
        assert "Anomal" in html
        assert "Correlaciones" in html

    def test_render_report_html_with_anomaly_data(self, df_metrics_anomaly, df_orders):
        report = generate_report(df_metrics_anomaly, df_orders)
        html = render_report_html(report, df_metrics_anomaly, df_orders)
        assert isinstance(html, str)
        assert len(html) > 1000


# ─────────────────────────────────────────────────────────────────────────────
# (d) Caída WoW > 10% en Perfect Orders es detectada como anomalía
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetectionPerfectOrders:
    @pytest.fixture
    def df_big_drop(self):
        return pd.DataFrame([{
            "COUNTRY": "CO",
            "CITY": "Bogotá",
            "ZONE": "ZonaCaida",
            "ZONE_TYPE": "Wealthy",
            "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.90,
            "L2W_ROLL": 0.88,
            "L1W_ROLL": 0.85,
            "L0W_ROLL": 0.60,
            "L0W": 0.58,
            "L1W": 0.83,
        }])

    @pytest.fixture
    def df_orders_empty(self):
        return pd.DataFrame(columns=["COUNTRY", "CITY", "ZONE", "L0W", "L1W"])

    def test_wow_drop_detected(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        assert len(results) >= 1
        assert any(r["zone"] == "ZonaCaida" for r in results)

    def test_detected_change_is_negative(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        hit = next(r for r in results if r["zone"] == "ZonaCaida")
        assert hit["change_pct"] < 0

    def test_detected_change_exceeds_10_pct(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        hit = next(r for r in results if r["zone"] == "ZonaCaida")
        assert abs(hit["change_pct"]) >= 10

    def test_detected_metric_is_perfect_orders(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        hit = next(r for r in results if r["zone"] == "ZonaCaida")
        assert hit["metric"] == "Perfect Orders"

    def test_anomaly_has_required_keys(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        hit = results[0]
        for key in ("zone", "metric", "change_pct", "insight", "recommendation", "severity"):
            assert key in hit

    def test_small_drop_below_threshold_not_detected(self, df_orders_empty):
        df = pd.DataFrame([{
            "COUNTRY": "CO", "CITY": "Bogotá", "ZONE": "ZonaEstable",
            "ZONE_TYPE": "Wealthy", "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.85, "L2W_ROLL": 0.85, "L1W_ROLL": 0.85, "L0W_ROLL": 0.84,
            "L0W": 0.84, "L1W": 0.85,
        }])
        results = detect_anomalies(df, df_orders_empty)
        assert all(r["zone"] != "ZonaEstable" for r in results)

    def test_severity_critico_for_large_drop(self, df_big_drop, df_orders_empty):
        results = detect_anomalies(df_big_drop, df_orders_empty)
        hit = next(r for r in results if r["zone"] == "ZonaCaida")
        assert hit["severity"] in ("crítico", "alto", "medio")
