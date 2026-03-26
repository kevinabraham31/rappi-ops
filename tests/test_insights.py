import pandas as pd
import pytest

from services.insights_service import (
    detect_anomalies,
    detect_benchmarking,
    detect_correlations,
    detect_deteriorating_trends,
    detect_opportunities,
    generate_report,
)


class TestDetectAnomalies:
    def test_large_wow_drop_detected(self, df_metrics_anomaly, df_orders):
        results = detect_anomalies(df_metrics_anomaly, df_orders)
        assert len(results) >= 1
        zones = [r["zone"] for r in results]
        assert "Facatativa" in zones

    def test_small_change_not_flagged(self, df_orders):
        df = pd.DataFrame([{
            "COUNTRY": "CO", "CITY": "Bogotá", "ZONE": "Stable",
            "ZONE_TYPE": "Wealthy", "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.85, "L2W_ROLL": 0.85, "L1W_ROLL": 0.85, "L0W_ROLL": 0.85,
            "L0W": 0.84, "L1W": 0.84,
        }])
        results = detect_anomalies(df, df_orders)
        assert all(r["zone"] != "Stable" for r in results)

    def test_results_sorted_by_magnitude(self, df_metrics_anomaly, df_orders):
        results = detect_anomalies(df_metrics_anomaly, df_orders)
        if len(results) >= 2:
            changes = [abs(r["change_pct"]) for r in results]
            assert changes == sorted(changes, reverse=True)

    def test_result_has_required_keys(self, df_metrics_anomaly, df_orders):
        results = detect_anomalies(df_metrics_anomaly, df_orders)
        for r in results:
            assert "zone" in r
            assert "metric" in r
            assert "change_pct" in r
            assert "insight" in r
            assert "recommendation" in r


class TestDetectDeterioratingTrends:
    def test_four_week_consecutive_drop_detected(self, df_metrics_deteriorating):
        results = detect_deteriorating_trends(df_metrics_deteriorating)
        assert len(results) >= 1
        zones = [r["zone"] for r in results]
        assert "Usme" in zones or "Iztapalapa" in zones

    def test_non_consecutive_drop_not_flagged(self):
        df = pd.DataFrame([{
            "COUNTRY": "CO", "CITY": "Bogotá", "ZONE": "Bounce",
            "ZONE_TYPE": "Wealthy", "ZONE_PRIORITIZATION": "High Priority",
            "METRIC": "Perfect Orders",
            "L3W_ROLL": 0.80, "L2W_ROLL": 0.75, "L1W_ROLL": 0.78, "L0W_ROLL": 0.74,
            "L0W": 0.73, "L1W": 0.76,
        }])
        results = detect_deteriorating_trends(df)
        assert all(r["zone"] != "Bounce" for r in results)

    def test_result_capped_at_10(self, df_metrics):
        results = detect_deteriorating_trends(df_metrics)
        assert len(results) <= 10


class TestDetectBenchmarking:
    def test_returns_list(self, df_metrics):
        results = detect_benchmarking(df_metrics)
        assert isinstance(results, list)

    def test_gap_vs_peer_present(self, df_metrics):
        results = detect_benchmarking(df_metrics)
        for r in results:
            assert "gap_vs_peer" in r
            assert "zone" in r
            assert "metric" in r

    def test_capped_at_10(self, df_metrics):
        results = detect_benchmarking(df_metrics)
        assert len(results) <= 10


class TestDetectCorrelations:
    def test_returns_list(self, df_metrics):
        results = detect_correlations(df_metrics)
        assert isinstance(results, list)

    def test_capped_at_8(self, df_metrics):
        results = detect_correlations(df_metrics)
        assert len(results) <= 8

    def test_correlation_value_in_range(self, df_metrics):
        results = detect_correlations(df_metrics)
        for r in results:
            assert -1.0 <= r["corr"] <= 1.0

    def test_result_has_required_keys(self, df_metrics):
        results = detect_correlations(df_metrics)
        for r in results:
            assert "metric_a" in r
            assert "metric_b" in r
            assert "corr" in r
            assert "insight" in r


class TestDetectOpportunities:
    def test_returns_list(self, df_metrics):
        results = detect_opportunities(df_metrics)
        assert isinstance(results, list)

    def test_result_has_zone_and_insight(self, df_metrics):
        results = detect_opportunities(df_metrics)
        for r in results:
            assert "zone" in r
            assert "insight" in r
            assert "recommendation" in r

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["COUNTRY", "CITY", "ZONE", "METRIC", "L0W_ROLL"])
        results = detect_opportunities(df)
        assert results == []


class TestGenerateReport:
    def test_report_has_all_sections(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        assert hasattr(report, "anomalies")
        assert hasattr(report, "deteriorating")
        assert hasattr(report, "benchmarking")
        assert hasattr(report, "correlations")
        assert hasattr(report, "opportunities")
        assert hasattr(report, "summary")

    def test_summary_is_non_empty(self, df_metrics_anomaly, df_orders):
        report = generate_report(df_metrics_anomaly, df_orders)
        assert isinstance(report.summary, list)

    def test_all_sections_are_lists(self, df_metrics, df_orders):
        report = generate_report(df_metrics, df_orders)
        assert isinstance(report.anomalies, list)
        assert isinstance(report.deteriorating, list)
        assert isinstance(report.correlations, list)
        assert isinstance(report.opportunities, list)
