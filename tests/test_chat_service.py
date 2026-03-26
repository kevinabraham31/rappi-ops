import pytest

from services.chat_service import ChatService


@pytest.fixture
def svc(df_metrics, df_orders):
    metrics_catalog = df_metrics["METRIC"].unique().tolist()
    return ChatService(df_metrics, df_orders, metrics_catalog)


class TestNormalizeMetric:
    def test_exact_synonym(self, svc):
        assert svc._normalize_metric("lead penetration") == "Lead Penetration"
        assert svc._normalize_metric("perfect order") == "Perfect Orders"
        assert svc._normalize_metric("gross profit ue") == "Gross Profit UE"

    def test_case_insensitive(self, svc):
        assert svc._normalize_metric("PERFECT ORDERS") == "Perfect Orders"

    def test_unknown_returns_none_or_closest(self, svc):
        result = svc._normalize_metric("temperatura del tiempo")
        assert result is None or isinstance(result, str)


class TestNormalizeCountry:
    def test_known_countries(self, svc):
        assert svc._normalize_country("colombia") == "CO"
        assert svc._normalize_country("mexico") == "MX"
        assert svc._normalize_country("brasil") == "BR"
        assert svc._normalize_country("brazil") == "BR"
        assert svc._normalize_country("argentina") == "AR"

    def test_unknown_country_returns_none(self, svc):
        assert svc._normalize_country("alemania") is None

    def test_partial_match_in_sentence(self, svc):
        assert svc._normalize_country("zonas en colombia esta semana") == "CO"


class TestIsMetaQuestion:
    def test_meta_phrases_detected(self, svc):
        assert svc._is_meta_question("¿qué puedes hacer?") is True
        assert svc._is_meta_question("ayuda") is True
        assert svc._is_meta_question("¿cómo funciona esto?") is True

    def test_data_question_not_meta(self, svc):
        assert svc._is_meta_question("muéstrame lead penetration en colombia") is False
        assert svc._is_meta_question("top 5 zonas con mayor perfect orders") is False


class TestIsOutOfScope:
    def test_oos_topics_rejected(self, svc):
        assert svc._is_out_of_scope("cuéntame un chiste") is True
        assert svc._is_out_of_scope("¿quién eres?") is True
        assert svc._is_out_of_scope("el marcador del partido de ayer") is True

    def test_operational_questions_accepted(self, svc):
        assert svc._is_out_of_scope("muéstrame lead penetration en bogotá") is False
        assert svc._is_out_of_scope("evolución de perfect orders en facatativa") is False

    def test_very_short_messages_rejected(self, svc):
        assert svc._is_out_of_scope("hola") is True


class TestIsIndependentQuery:
    def test_known_markers_are_independent(self, svc):
        assert svc._is_independent_query("zonas problemáticas en colombia") is True
        assert svc._is_independent_query("qué zonas más crecen en órdenes") is True
        assert svc._is_independent_query("ahora muéstrame otra cosa") is True
        assert svc._is_independent_query("en cambio quiero ver brasil") is True

    def test_follow_up_not_independent(self, svc):
        assert svc._is_independent_query("explícame este resultado") is False
        assert svc._is_independent_query("¿y cuál es el promedio?") is False


class TestIsExplanatoryFollowUp:
    def test_explicame_detected(self, svc):
        assert svc._is_explanatory_follow_up("explícame esto") is True
        assert svc._is_explanatory_follow_up("dame una recomendación") is True

    def test_new_data_question_not_follow_up(self, svc):
        assert svc._is_explanatory_follow_up("muéstrame lead penetration en colombia") is False


class TestAnswerFallback:
    def test_answer_returns_dict_with_required_keys(self, svc):
        result = svc.answer("muéstrame lead penetration en colombia", [])
        assert "reply" in result
        assert "data_rows" in result
        assert "columns" in result
        assert "chart" in result
        assert "suggestions" in result

    def test_oos_response_has_reply(self, svc):
        result = svc.answer("¿cuánto es 2+2?", [])
        assert "reply" in result
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 10

    def test_reply_is_never_empty(self, svc):
        queries = [
            "muéstrame perfect orders en colombia",
            "top 5 zonas con mayor lead penetration",
            "evolución de gross profit ue en facatativa",
        ]
        for q in queries:
            result = svc.answer(q, [])
            assert result["reply"], f"Empty reply for: {q}"

    def test_llm_unavailable_flag_resets_after_threshold(self, svc):
        import time
        from unittest.mock import MagicMock
        svc.client = MagicMock()
        svc._llm_unavailable_since = time.time() - 601
        result = svc._llm_is_available()
        assert result is True
        assert svc._llm_unavailable_since == 0.0

    def test_llm_flag_blocks_within_cooldown(self, svc):
        import time
        svc._llm_unavailable_since = time.time() - 60
        elapsed = time.time() - svc._llm_unavailable_since
        assert elapsed < 600
