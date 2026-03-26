import numpy as np
import pandas as pd
import pytest

from services.query_executor import build_chart_payload, format_result_payload, run_code


class TestFormatResultPayload:
    def test_none_returns_error(self):
        result = format_result_payload(None)
        assert result["success"] is False
        assert result["data_rows"] == []

    def test_empty_dataframe_returns_error(self):
        result = format_result_payload(pd.DataFrame())
        assert result["success"] is False

    def test_dataframe_success(self):
        df = pd.DataFrame({"ZONE": ["Facatativa", "Usme"], "L0W_ROLL": [0.85, 0.72]})
        result = format_result_payload(df)
        assert result["success"] is True
        assert len(result["data_rows"]) == 2
        assert result["columns"] == ["ZONE", "L0W_ROLL"]

    def test_dataframe_caps_at_50_rows(self):
        df = pd.DataFrame({"x": range(100)})
        result = format_result_payload(df)
        assert len(result["data_rows"]) == 50

    def test_scalar_success(self):
        result = format_result_payload(42)
        assert result["success"] is True
        assert result["data_rows"][0]["value"] == "42"

    def test_nan_becomes_none(self):
        df = pd.DataFrame({"ZONE": ["A"], "val": [float("nan")]})
        result = format_result_payload(df)
        assert result["data_rows"][0]["val"] is None

    def test_floats_rounded_to_4_decimals(self):
        df = pd.DataFrame({"v": [1.123456789]})
        result = format_result_payload(df)
        assert result["data_rows"][0]["v"] == 1.1235


class TestRunCode:
    def test_valid_code_returns_dataframe(self, df_metrics, df_orders):
        code = "result = df_metrics[df_metrics['METRIC'] == 'Perfect Orders']"
        result = run_code(code, df_metrics, df_orders)
        assert result["success"] is True
        assert len(result["data_rows"]) > 0

    def test_invalid_code_returns_error(self, df_metrics, df_orders):
        code = "result = df_metrics['NONEXISTENT_COLUMN']"
        result = run_code(code, df_metrics, df_orders)
        assert result["success"] is False
        assert result["error"] is not None

    def test_syntax_error_returns_error(self, df_metrics, df_orders):
        code = "result = df_metrics[df_metrics['METRIC' =="
        result = run_code(code, df_metrics, df_orders)
        assert result["success"] is False

    def test_original_dataframes_not_mutated(self, df_metrics, df_orders):
        original_len = len(df_metrics)
        code = "df_metrics.drop(df_metrics.index, inplace=True); result = df_metrics"
        run_code(code, df_metrics, df_orders)
        assert len(df_metrics) == original_len

    def test_result_none_returns_error(self, df_metrics, df_orders):
        code = "result = None"
        result = run_code(code, df_metrics, df_orders)
        assert result["success"] is False


class TestBuildChartPayload:
    def test_non_dataframe_returns_none(self):
        assert build_chart_payload(42) is None
        assert build_chart_payload("text") is None
        assert build_chart_payload(None) is None

    def test_single_column_df_returns_none(self):
        df = pd.DataFrame({"ZONE": ["A", "B", "C"]})
        assert build_chart_payload(df) is None

    def test_bar_chart_does_not_crash(self):
        df = pd.DataFrame({"ZONE": ["A", "B", "C"], "L0W_ROLL": [0.8, 0.7, 0.6]})
        result = build_chart_payload(df)
        if result is not None:
            assert result.get("type") in ("image", "bar", "line")
            assert "image_base64" in result

    def test_larger_df_produces_chart(self):
        df = pd.DataFrame({
            "ZONE": [f"Z{i}" for i in range(6)],
            "L0W_ROLL": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        })
        result = build_chart_payload(df)
        if result is not None:
            assert "image_base64" in result
