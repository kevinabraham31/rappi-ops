from __future__ import annotations

import difflib
import os
import re
import time
from typing import Any

import pandas as pd
from openai import OpenAI
from openai import RateLimitError as LLMRateLimitError, APIStatusError as LLMAPIError

from services.query_executor import build_chart_payload, clean_response, extract_code, format_result_payload, run_code

METRIC_SYNONYMS = {
    "lead penetration": "Lead Penetration",
    "% lead penetration": "Lead Penetration",
    "perfect order": "Perfect Orders",
    "perfect orders": "Perfect Orders",
    "gross profit ue": "Gross Profit UE",
    "turbo adoption": "Turbo Adoption",
    "pro adoption": "Pro Adoption",
    "orders": "Orders",
}

COUNTRY_NAMES = {
    "mexico": "MX",
    "méxico": "MX",
    "colombia": "CO",
    "brazil": "BR",
    "brasil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "costa rica": "CR",
    "ecuador": "EC",
    "peru": "PE",
    "perú": "PE",
    "uruguay": "UY",
}

SEMANTIC_MAPPINGS = {
    "esta semana": "L0W_ROLL / L0W",
    "semana actual": "L0W_ROLL / L0W",
    "semana pasada": "L1W_ROLL / L1W",
    "tendencia": "L3W a L0W",
    "performance": "L3W a L0W",
    "zonas ricas": "ZONE_TYPE = Wealthy",
    "zonas pobres": "ZONE_TYPE = Non Wealthy",
    "zonas criticas": "ZONE_PRIORITIZATION = High Priority",
    "zonas críticas": "ZONE_PRIORITIZATION = High Priority",
}


class ChatService:
    def __init__(self, df_metrics: pd.DataFrame, df_orders: pd.DataFrame, metrics_catalog: list[str]):
        self.df_metrics = df_metrics
        self.df_orders = df_orders
        self.metrics_catalog = metrics_catalog
        self.zone_catalog = sorted(self.df_metrics["ZONE"].dropna().astype(str).unique().tolist())
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None
        self._llm_unavailable_since: float = 0.0
        self.system_prompt = self._build_system_prompt()

    def _llm_is_available(self) -> bool:
        if self.client is None:
            return False
        if self._llm_unavailable_since:
            # 600 s da tiempo suficiente para que DeepSeek libere el rate limit; reintentar antes solo acelera la siguiente penalización
            
            if time.time() - self._llm_unavailable_since > 600:
                self._llm_unavailable_since = 0.0
                return True
            return False
        return True

    def _build_system_prompt(self) -> str:
        metrics_text = ", ".join(self.metrics_catalog)
        countries_text = ", ".join(sorted(self.df_metrics["COUNTRY"].unique().tolist()))
        zones_sample = ", ".join(self.zone_catalog[:20])
        return f"""Eres el Rappi Ops Translator, un analista de operaciones de élite integrado al sistema de datos de Rappi.
Tu único propósito es responder preguntas sobre los datos operacionales disponibles en el dataset.

LÍMITE DE ALCANCE — CRÍTICO:
Si la pregunta NO es sobre métricas, zonas, países, tendencias, órdenes u operaciones de Rappi,
responde EXACTAMENTE esto y nada más:
"Solo puedo ayudarte con análisis operacionales de Rappi. Prueba preguntarme por zonas, métricas o tendencias."

DATOS DISPONIBLES:
- df_metrics: métricas operacionales por zona y semana.
- df_orders: volumen de órdenes por zona y semana.
- Métricas: {metrics_text}
- Países (código ISO): {countries_text}
- Zonas (muestra): {zones_sample}
- Semanas disponibles: L8W_ROLL (hace 8 sem) → L0W_ROLL (semana actual). Para órdenes: L8W → L0W.
- Segmentos: ZONE_TYPE ∈ [Wealthy, Non Wealthy], ZONE_PRIORITIZATION ∈ [High Priority, Prioritized, Not Prioritized]

CÓDIGO OBLIGATORIO — REGLA #1:
Para CUALQUIER pregunta sobre datos, métricas, zonas o países, SIEMPRE debes generar un bloque de código Python ejecutable. No hay excepción. Incluso para preguntas simples como "cuál es el más alto" o "dame el promedio", genera el código que calcule y filtre el resultado exacto que el usuario pide. El código siempre debe terminar con result = ... asignando el DataFrame o valor final. Nunca respondas solo con texto cuando la pregunta involucra datos.

REGLAS DE CÓDIGO:
1. Todo código va dentro de un bloque ```python```.
2. El bloque debe terminar con `result = ...` asignando un DataFrame, Series o valor escalar.
3. Usa SOLO columnas que existan en df_metrics o df_orders.
4. Para filtrar por país usa el código ISO en mayúsculas (ej: "MX", "CO").
5. Para tendencias temporales incluye columnas de semana en orden cronológico (L8W_ROLL → L0W_ROLL).
6. Redondea valores float a 4 decimales.
7. Si una métrica solicitada no existe, NO la inventes. Sugiere la más similar disponible.

MAPEO SEMÁNTICO — aplica siempre:
- "esta semana" / "semana actual" → L0W_ROLL (o L0W para órdenes)
- "semana pasada" → L1W_ROLL
- "últimas N semanas" → usa las últimas N columnas de semana
- "zonas ricas" / "wealthy" → ZONE_TYPE == "Wealthy"
- "zonas no ricas" / "non wealthy" → ZONE_TYPE == "Non Wealthy"
- "zonas críticas" / "prioritarias" → ZONE_PRIORITIZATION == "High Priority"
- "zonas problemáticas" → zonas con L0W_ROLL bajo Y deterioro reciente (L0W_ROLL < L1W_ROLL)
- "crecimiento" / "crecen" → diferencia o porcentaje entre L0W y L5W

ESTILO DE RESPUESTA:
- Responde en español siempre.
- Primero da la respuesta ejecutiva en 1-2 oraciones.
- Luego el bloque de código (si aplica).
- Al final una interpretación de negocio: qué significa el resultado y qué acción sugiere.
- Nunca menciones pandas, DataFrames, columnas técnicas ni código en la interpretación.
- Nunca inventes datos. Si no puedes responder con el dataset, dilo con claridad.

ANTI-AMBIGÜEDAD — CRÍTICO:
Si el usuario hace una pregunta vaga o sin contexto específico (ej. '¿Cómo va todo?', 'Dame un resumen', 'Hola'), NO respondas que no tienes información ni pidas más datos.
En su lugar, el sistema generará automáticamente un análisis proactivo de las métricas más críticas.
Cuando el usuario use lenguaje natural sin especificar la métrica exacta (ej. 'zonas con problemas', 'cómo están las ventas'), infiere la métrica más relevante del contexto e impleméntala.

CONTEXTO DE CONVERSACIÓN:
El historial de la conversación se incluye en cada mensaje. Úsalo para resolver referencias anafóricas como 'eso', 'esa zona', 'el resultado anterior', 'lo mismo pero en Colombia', etc.
"""

    def _is_out_of_scope(self, message: str) -> bool:
        # No usa el LLM: este filtro corre antes de cualquier llamada externa para que
        # preguntas fuera de dominio no consuman tokens ni introduzcan latencia
        lower = message.lower().strip()
        # Long/specific phrases: safe for substring match
        in_scope_phrases = [
            "zona", "country", "país", "pais", "ciudad", "métrica", "metrica",
            "semana", "week", "tendencia", "promedio",
            "evolución", "evolucion", "comparar", "compara",
            "crecimiento", "benchmark", "lead penetration", "perfect order",
            "gross profit", "turbo", "pro adoption", "orders", "ordenes", "órdenes",
            "wealthy", "non wealthy", "prioritized", "high priority", "rappi",
            "mexico", "méxico", "colombia", "brasil", "argentina", "chile",
            "peru", "perú", "uruguay", "ecuador", "costa rica",
            "filtrar", "listar", "mostrar", "cuáles", "cuales",
            "qué zonas", "que zonas", "ranking", "análisis", "analisis",
            "explícame", "explicame", "recomendación", "recomendacion",
            "términos de negocio", "terminos de negocio", "basada en este",
            "muéstrame", "muestrame",
            "a que te refieres", "a qué te refieres", "que significa", "qué significa",
            "como funciona", "cómo funciona", "no entiendo", "ayuda", "help",
            "que quieres decir", "qué quieres decir", "como puedo", "cómo puedo",
        ]
        if any(s in lower for s in in_scope_phrases):
            return False
        # Short/ambiguous tokens: require exact word match to avoid substring collisions
        in_scope_words = {
            "top", "mayor", "menor", "peor", "mejor", "resultado", "dame",
            "ar", "br", "cl", "co", "cr", "ec", "mx", "pe", "uy",
            "cuanto", "cuánto",
        }
        token_set = set(re.split(r"[^a-záéíóúüñ]+", lower))
        if token_set & in_scope_words:
            return False
        out_of_scope_signals = [
            "clima", "tiempo atmosférico", "noticias", "política", "politica",
            "fútbol", "futbol", "deporte", "receta", "cocina", "música", "musica",
            "chiste", "broma", "cuéntame un", "cuentame un", "historia de",
            "quién eres", "quien eres", "qué eres", "que eres", "eres una ia",
            "programar en", "javascript", "html", "css", "sql puro",
            "openai", "chatgpt", "gemini", "gpt",
            "cuanto es", "cuánto es", "cuanto son", "cuánto son",
            "cuantos es", "cuántos es", "2+2", "multiplicar", "dividir",
            "partido", "campeonato", "liga ", "equipo de", "quien gano", "quién ganó",
            "ganó el", "gano el", "marcador", "gol", "pelota", "béisbol", "beisbol",
            "baloncesto", "basquetbol", "tenis", "boxeo", "pelea", "concierto",
            "película", "pelicula", "serie de tv", "actor", "actriz", "cantante",
            "presidente", "gobierno", "elecciones", "congreso",
        ]
        if any(s in lower for s in out_of_scope_signals):
            return True
        if len(token_set - {""}) <= 2:
            return True
        return False

    def _out_of_scope_response(self) -> dict:
        return {
            "reply": "Solo analizo datos operacionales de Rappi — métricas, zonas, países y tendencias. No puedo responder preguntas sobre otros temas.",
            "data": None,
            "data_rows": [],
            "columns": [],
            "highlights": [],
            "chart": None,
            "suggestions": [],
        }

    def _request_analysis(self, messages: list[dict[str, str]], max_tokens: int = 1800) -> str:
        if not self.client:
            return ""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except LLMRateLimitError:
            return "__RATE_LIMIT__"
        except LLMAPIError:
            return ""

    def _request_code_only(self, messages: list[dict[str, str]]) -> str | None:
        # El historial de conversación se omite deliberadamente cuando se llama parageneración de código inicial: respuestas textuales previas contaminan eloutput esperado (result = ...) y el LLM tiende a añadir explicaciones en prosa
        if not self.client:
            return None
        reply = self._request_analysis(
            messages
            + [
                {
                    "role": "user",
                    "content": "Devuélveme solo un bloque ```python``` ejecutable que resuelva la consulta usando df_metrics y/o df_orders. Debe terminar con result = ... y no incluyas explicación.",
                }
            ],
            max_tokens=1200,
        )
        return extract_code(reply)

    def _normalize_metric(self, message: str) -> str | None:
        lower = message.lower()
        for synonym, metric in METRIC_SYNONYMS.items():
            if synonym in lower:
                return metric
        for metric in self.metrics_catalog:
            if metric.lower() in lower:
                return metric
        return None

    def _closest_metric(self, message: str) -> str | None:
        lower = message.lower()
        # 0.55: umbral permisivo adrede — los nombres de métricas tienen palabras comunes
        # y el usuario raramente escribe el nombre técnico exacto
        matches = difflib.get_close_matches(lower, [metric.lower() for metric in self.metrics_catalog], n=1, cutoff=0.55)
        if not matches:
            return None
        match = matches[0]
        return next((metric for metric in self.metrics_catalog if metric.lower() == match), None)

    def _normalize_country(self, message: str) -> str | None:
        lower = message.lower()
        for name, code in COUNTRY_NAMES.items():
            if name in lower:
                return code
        tokens = set(lower.replace("?", " ").replace(",", " ").split())
        for token in tokens:
            upper = token.upper()
            if upper in set(self.df_metrics["COUNTRY"].unique()):
                return upper
        return None

    def _match_zone_from_text(self, message: str) -> str | None:
        lower = message.lower()
        for zone in self.zone_catalog:
            if zone.lower() in lower:
                return zone

        cleaned_tokens = [token for token in lower.replace("?", " ").replace(",", " ").split() if len(token) >= 4]

        # Reverse partial matching: check if a significant part of a zone name appears in the message.
        # Direction: zone_part → message (avoids false positives like 'evolucion' matching 'Revolucion').
        for zone in self.zone_catalog:
            parts = zone.lower().replace(" - ", " ").split()
            for part in parts:
                if len(part) >= 6 and part in lower:
                    return zone

        # 0.80 por token: zonas son nombres propios cortos; un umbral bajo genera
        # falsos positivos frecuentes (ej: "una" matchea "Usna", "Lima" matchea "Limas")
        for token in cleaned_tokens:
            if len(token) < 5:
                continue
            matches = difflib.get_close_matches(token, [zone.lower() for zone in self.zone_catalog], n=1, cutoff=0.80)
            if matches:
                matched = matches[0]
                return next((zone for zone in self.zone_catalog if zone.lower() == matched), None)

        sentence_matches = difflib.get_close_matches(lower, [zone.lower() for zone in self.zone_catalog], n=1, cutoff=0.65)
        if sentence_matches:
            matched = sentence_matches[0]
            return next((zone for zone in self.zone_catalog if zone.lower() == matched), None)
        return None

    def _semantic_context_labels(self, message: str) -> list[str]:
        lower = message.lower()
        found = []
        for phrase, mapping in SEMANTIC_MAPPINGS.items():
            if phrase in lower:
                found.append(f"{phrase} -> {mapping}")
        return found

    def _extract_requested_weeks(self, message: str, default: int = 4) -> int:
        lower = message.lower()
        match = re.search(
            r"(?:últimas?|ultimas?|de|para)\s+(\d+)\s+semanas?|(\d+)\s+semanas?",
            lower,
        )
        if match:
            value = match.group(1) or match.group(2)
            if value:
                return max(1, min(int(value), 9))
        if "semana pasada" in lower and ("esta semana" in lower or "semana actual" in lower):
            return 2
        return default

    def _week_columns_for_request(self, message: str, metrics_dataset: bool = True) -> list[str]:
        count = self._extract_requested_weeks(message)
        if metrics_dataset:
            ordered = ["L8W_ROLL", "L7W_ROLL", "L6W_ROLL", "L5W_ROLL", "L4W_ROLL", "L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]
        else:
            ordered = ["L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W"]
        return ordered[-count:]

    def _transparency_footer(self, message: str, payload: dict[str, Any]) -> str:
        rows = payload.get("data_rows", []) or []
        count = len(rows)
        if count == 0:
            return ""
        metric = self._normalize_metric(message) or self._closest_metric(message)
        if not metric and rows:
            metric = rows[0].get("METRIC") or rows[0].get("metric")
        country = self._normalize_country(message)
        if not country and rows:
            country = rows[0].get("COUNTRY")
        parts = []
        if metric:
            parts.append(metric)
        if country:
            parts.append(country)
        scope = " · ".join(parts) if parts else "dataset completo"
        plural = "s" if count != 1 else ""
        return f"\n\n_{count} registro{plural} encontrado{plural} · {scope}_"

    def _history_text(self, history: list[dict[str, str]]) -> str:
        chunks = []
        for item in history[-6:]:
            content = item.get("content", "")
            if content:
                chunks.append(content)
        return " ".join(chunks)

    def _context_from_history(self, history: list[dict[str, str]]) -> dict[str, str | None]:
        combined = self._history_text(history)
        return {
            "metric": self._normalize_metric(combined) or self._closest_metric(combined),
            "country": self._normalize_country(combined),
            "zone": self._match_zone_from_text(combined),
        }

    def _resolve_contextual_message(self, message: str, history: list[dict[str, str]]) -> str:
        lower = message.lower().strip()
        follow_up_tokens = [
            "eso",
            "esa",
            "esas",
            "esto",
            "estos",
            "esas cosas",
            "por qué",
            "porque",
            "qué significa",
            "que significa",
            "explica",
        ]
        if any(token in lower for token in follow_up_tokens) and history:
            previous_text = self._history_text(history)
            return f"Contexto previo: {previous_text}\nPregunta actual: {message}"
        return message

    def _is_explanatory_follow_up(self, message: str) -> bool:
        lower = message.lower()
        tokens = [
            "explícame", "explicame",
            "qué significa", "que significa",
            "en términos de negocio", "en terminos de negocio",
            "recomendación", "recomendacion",
            "dame una recomendación", "dame una recomendacion",
            "basada en este análisis", "basada en este analisis",
            "basada en este",
            "por qué", "porque",
            "cómo interpreto", "como interpreto",
        ]
        return any(token in lower for token in tokens)

    def _answer_explanatory_follow_up(self, history: list[dict[str, str]], message: str = "") -> str:
        previous_user_queries = [item.get("content", "") for item in history if item.get("role") == "user"]
        if not previous_user_queries:
            return "Puedo explicarte el resultado en lenguaje simple si primero haces una consulta sobre una métrica o una zona."

        last_query = previous_user_queries[-1].lower()

        is_recommendation = any(
            t in message.lower()
            for t in ["recomendación", "recomendacion", "recomendación ejecutiva", "ejecutiva", "basada en este"]
        )

        if is_recommendation:
            if "lead penetration" in last_query:
                return "Recomendación ejecutiva: prioriza las acciones comerciales en las zonas con Lead Penetration más bajo; revisa la propuesta de valor y la ejecución de ventas en campo. Las zonas con mejor desempeño pueden servir como referencia para replicar prácticas."
            if "perfect order" in last_query or "perfect orders" in last_query:
                return "Recomendación ejecutiva: en las zonas con Perfect Order más bajo, investiga las causas raíz (tasa de cancelación, problemas de inventario o tiempos de entrega). Establece un plan de mejora operativa en el corto plazo con seguimiento semanal."
            if "gross profit ue" in last_query:
                return "Recomendación ejecutiva: en zonas con Gross Profit UE en caída, revisa la mezcla de órdenes, los descuentos aplicados y el costo logístico. En zonas con tendencia positiva, evalúa si el crecimiento es sostenible o impulsado por incentivos temporales."
            if "órdenes" in last_query or "ordenes" in last_query:
                return "Recomendación ejecutiva: en zonas con crecimiento de órdenes, asegura capacidad operativa (couriers, restaurantes activos). En zonas con caída, revisa las palancas de demanda: promociones, cobertura y experiencia de usuario."
            return "Recomendación ejecutiva: cruza los resultados de esta consulta con el contexto de negocio de cada zona (inversión reciente, cambios operativos, temporalidad). Prioriza la intervención en las zonas con mayor impacto potencial según volumen de órdenes y posición en el benchmark regional."

        if "lead penetration" in last_query:
            return "Este resultado muestra qué zonas tienen mejor capacidad de convertir prospectos en tiendas activas dentro de Rappi. En negocio, una zona arriba en Lead Penetration sugiere mejor ejecución comercial o mayor madurez del mercado."

        if "perfect order" in last_query or "perfect orders" in last_query:
            return "Este resultado te ayuda a identificar qué zonas entregan una mejor experiencia operativa al usuario. Un Perfect Order más alto suele significar menos cancelaciones, defectos o demoras."

        if "gross profit ue" in last_query:
            return "Aquí estás viendo rentabilidad por orden. Si la tendencia mejora, la zona está generando más margen por pedido; si cae, puede haber presión en costos, descuentos o mezcla comercial."

        if "órdenes" in last_query or "ordenes" in last_query:
            return "Este resultado muestra dónde está creciendo o cayendo la demanda. En negocio sirve para detectar zonas que merecen más inversión, más capacidad o revisión de ejecución."

        return "Este resultado muestra el desempeño operativo de las zonas analizadas. Las zonas con delta negativo están deteriorándose respecto a la semana pasada y merecen atención inmediata. Las zonas con delta positivo están mejorando y pueden ser referencia de buenas prácticas."

    _TERM_GLOSSARY: dict[str, str] = {
        "deterioro": "Deterioro significa que una métrica viene bajando semana a semana de forma sostenida. Es una señal de alerta: esa zona puede necesitar atención operativa o comercial para revertir la tendencia.",
        "escalar": "Escalar lo que funciona significa tomar las prácticas o condiciones que generan buenos resultados en las zonas con mejor desempeño, y replicarlas en zonas con resultados más bajos.",
        "escalar lo que funciona": "Escalar lo que funciona significa tomar las prácticas o condiciones que generan buenos resultados en las zonas con mejor desempeño, y replicarlas en zonas con resultados más bajos.",
        "benchmark": "Benchmark es el valor de referencia (promedio del grupo comparable) con el que se compara una zona. Si una zona está por encima del benchmark, va bien; si está por debajo, hay una brecha que cerrar.",
        "tendencia": "Tendencia es la evolución de una métrica a lo largo de varias semanas. Una tendencia positiva significa que el indicador mejora; negativa, que empeora. Sirve para anticipar problemas antes de que se agraven.",
        "wealthy": "Wealthy es una categoría de zona que indica un perfil socioeconómico alto. Se usa para comparar si el desempeño operativo difiere entre tipos de zona.",
        "non wealthy": "Non Wealthy es una categoría de zona con perfil socioeconómico medio o bajo. Comparar Wealthy vs Non Wealthy ayuda a entender si hay diferencias estructurales en el desempeño.",
        "high priority": "High Priority es el nivel de priorización más alto para una zona según la estrategia operativa de Rappi. Indica que esa zona tiene mayor impacto en los resultados del negocio.",
        "prioritized": "Prioritized es el segundo nivel de importancia de una zona. No es la más crítica, pero sí merece seguimiento activo.",
        "lead penetration": "Lead Penetration es el porcentaje de leads (tiendas o restaurantes potenciales) que se convierten en aliados activos dentro de Rappi. A mayor Lead Penetration, mejor ejecución comercial en esa zona.",
        "perfect order": "Perfect Order (o Perfect Orders) mide qué porcentaje de los pedidos se entrega sin errores: sin cancelaciones, sin demoras excesivas y con la calidad esperada. Es el indicador clave de experiencia operativa.",
        "perfect orders": "Perfect Orders mide qué porcentaje de los pedidos se entrega sin errores: sin cancelaciones, sin demoras excesivas y con la calidad esperada. Es el indicador clave de experiencia operativa.",
        "gross profit ue": "Gross Profit UE (por Unidad de Entrega) es la ganancia bruta que genera Rappi por cada pedido en esa zona. A mayor valor, más rentable es esa zona por orden.",
        "gross profit": "Gross Profit UE es la ganancia bruta que genera Rappi por cada pedido en esa zona. A mayor valor, más rentable es esa zona por orden.",
        "turbo adoption": "Turbo Adoption mide qué porcentaje de los usuarios activos utiliza el servicio Turbo (entrega ultrarrápida). Es un indicador de madurez del servicio premium en la zona.",
        "pro adoption": "Pro Adoption mide qué porcentaje de los usuarios activos tiene suscripción Rappi Pro. A mayor adopción, mayor retención y ticket promedio esperado.",
        "intervenir": "Intervenir significa tomar acción directa en una zona con mal desempeño: puede incluir visitas de campo, cambios operativos, refuerzo comercial o ajustes de producto.",
        "gap": "Gap es la diferencia entre el valor actual de una zona y el benchmark o promedio del grupo comparable. Un gap negativo indica que la zona está por debajo de lo esperado.",
        "l0w": "L0W (o L0W_ROLL) es la semana actual, la más reciente del dataset. L1W es la semana pasada, L2W la anterior, y así sucesivamente hasta L8W.",
        "semana actual": "La semana actual en el dataset se llama L0W (L0W_ROLL para métricas). L1W es la semana pasada, L2W la anterior, y así hasta L8W (hace 8 semanas).",
        "delta": "Delta es la diferencia entre el valor de la semana actual (L0W_ROLL) y el de la semana anterior (L1W_ROLL). Un delta negativo indica deterioro reciente: esa zona empeoró en los últimos 7 días. Un delta positivo indica mejora. Se usa para detectar qué zonas empeoraron o mejoraron de una semana a otra.",
        "l1w": "L1W (o L1W_ROLL) es el valor de la semana pasada, la semana inmediatamente anterior a la actual. Se compara contra L0W para calcular si hubo mejora o deterioro reciente en una zona.",
        "l0w roll": "L0W_ROLL es la semana actual, el dato más reciente del dataset. Es el punto de referencia principal para evaluar el desempeño operativo de esta semana.",
    }

    def _is_meta_question(self, message: str) -> bool:
        lower = message.lower()
        triggers = [
            "a que te refieres", "a qué te refieres",
            "que quieres decir", "qué quieres decir",
            "que significa", "qué significa",
            "que es eso", "qué es eso",
            "no entiendo",
            "como puedo utilizar", "cómo puedo utilizar",
            "como usar", "cómo usar",
            "como funciona", "cómo funciona",
            "para que sirve", "para qué sirve",
            "como se usa", "cómo se usa",
            "que puedo preguntar", "qué puedo preguntar",
            "que puedes hacer", "qué puedes hacer",
            "ayuda", "help",
        ]
        if any(t in lower for t in triggers):
            return True
        prefixes = (
            "que es ", "qué es ",
            "que es el ", "qué es el ",
            "que es la ", "qué es la ",
            "que significa ", "qué significa ",
        )
        for term in self._TERM_GLOSSARY:
            if any((p + term) in lower for p in prefixes):
                return True
        if re.search(r"qu[eé]\s+es\s+\w+", lower):
            return True
        return False

    def _answer_meta_question(self, message: str) -> str:
        lower = message.lower()

        for term, definition in self._TERM_GLOSSARY.items():
            if term in lower:
                return definition

        if any(t in lower for t in [
            "como puedo utilizar", "cómo puedo utilizar", "como usar", "cómo usar",
            "como funciona", "cómo funciona", "para que sirve", "para qué sirve",
            "como se usa", "cómo se usa", "que puedo preguntar", "qué puedo preguntar",
            "que puedes hacer", "qué puedes hacer", "ayuda", "help",
        ]):
            return (
                "Soy el Rappi Ops Translator. Puedes preguntarme directamente en español sobre las operaciones de Rappi. "
                "No necesitas saber programar ni usar términos técnicos. Algunos ejemplos:\n\n"
                "• ¿Cuáles son las 5 zonas con mejor desempeño en Colombia?\n"
                "• Muéstrame cómo evolucionó Gross Profit UE en México las últimas 8 semanas\n"
                "• ¿Qué zonas de Brasil están teniendo problemas?\n"
                "• Compara las zonas ricas vs no ricas en Argentina\n\n"
                "Solo menciona una métrica (Lead Penetration, Perfect Orders, Gross Profit UE…), un país o zona, y el tipo de análisis que buscas."
            )

        return (
            "No estoy seguro a qué término te refieres. Puedo explicarte cualquiera de estos conceptos si lo mencionas: "
            "Lead Penetration, Perfect Orders, Gross Profit UE, deterioro, benchmark, tendencia, Wealthy / Non Wealthy, "
            "High Priority, Turbo Adoption, Pro Adoption."
        )

    _FOOTER_RE = re.compile(r'\d+\s+registros?\s+encontrados?', re.IGNORECASE)

    def _sanitize_reply_text(self, reply_text: str) -> str:
        if not reply_text:
            return ""

        cleaned_lines = []
        blocked_prefixes = (
            "result =",
            "interpreta",
            "interpretación",
            "interpretacion",
            "código:",
            "codigo:",
            "python",
        )
        blocked_contains = (
            "código de análisis",
            "codigo de analisis",
            "interpretación de negocio",
            "interpretacion de negocio",
        )

        for raw_line in reply_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith(blocked_prefixes):
                continue
            _blocked_match = next((p for p in blocked_contains if p in lower), None)
            if _blocked_match:
                after = re.split(r'[:\*]+\s*', line, maxsplit=1)
                if len(after) > 1 and after[-1].strip():
                    line = after[-1].strip()
                    lower = line.lower()
                else:
                    continue
            if line.startswith("```"):
                continue
            if self._FOOTER_RE.search(line):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned

    def _build_executive_summary(self, payload: dict[str, Any], message: str) -> str:
        highlights = payload.get("highlights", []) or []
        data_rows = payload.get("data_rows", []) or []
        lower = message.lower()

        if not data_rows:
            return "No encontré resultados claros para esta consulta."

        if "top" in lower or "mayor" in lower or "5 zonas" in lower:
            zone = next((item["value"] for item in highlights if item.get("label") == "Zona destacada"), None)
            metric = next((item["value"] for item in highlights if item.get("label") not in {"Zona destacada", "Filas encontradas"}), None)
            if zone and metric:
                return f"Estas son las zonas líderes para la consulta. La zona más destacada es {zone}, con un valor de {metric}.\n\n¿Por qué importa? Esto sugiere una ejecución superior o una condición de mercado más favorable frente al resto de zonas comparadas."

        if "promedio" in lower:
            return "Aquí tienes el promedio por país para comparar rápidamente dónde está mejor y peor el desempeño.\n\n¿Por qué importa? Te ayuda a priorizar mercados donde conviene intervenir o replicar buenas prácticas." 

        if "wealthy" in lower and "non wealthy" in lower:
            first_row = data_rows[0]
            zone_type = first_row.get("ZONE_TYPE") or first_row.get("zone_type") or first_row.get("index")
            metric_value = None
            for key, value in first_row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_value = value
            if zone_type and metric_value is not None:
                return f"El mejor resultado dentro de esta comparación lo muestra el segmento {zone_type}, con un valor de {metric_value}.\n\n¿Por qué importa? Esta diferencia sugiere que el contexto socioeconómico o la propuesta de valor puede estar afectando la experiencia y el desempeño operativo entre tipos de zona."
            return "Aquí tienes la comparación entre zonas Wealthy y Non Wealthy.\n\n¿Por qué importa? Te ayuda a ver si el desempeño cambia de forma estructural según el perfil de la zona."

        if "gap_vs_peer" in {str(column).lower() for column in data_rows[0].keys()}:
            first_row = data_rows[0]
            zone_name = first_row.get("ZONE")
            metric_name = first_row.get("METRIC", "la métrica analizada")
            gap = first_row.get("GAP_VS_PEER")
            peer_avg = first_row.get("PEER_AVG")
            current_value = first_row.get("L0W_ROLL") or next(
                (v for k, v in first_row.items() if k.startswith("L0W") and isinstance(v, (int, float)) and not isinstance(v, bool)),
                None,
            )
            if gap is not None and peer_avg is not None and current_value is not None:
                direction = "por encima" if gap >= 0 else "por debajo"
                return f"{zone_name} está {abs(gap):.4f} puntos {direction} del benchmark de zonas similares en {metric_name}. Su valor actual es {current_value} versus un promedio comparable de {peer_avg}.\n\n¿Por qué importa? Esto te dice si la zona está sobre-ejecutando o quedándose atrás frente a pares comparables, lo que sirve para decidir si replicar prácticas o intervenir." 

        if "evoluci" in lower or "últimas 8 semanas" in lower or "ultimas 8 semanas" in lower:
            return "Aquí tienes la evolución semanal para identificar si la métrica viene mejorando, empeorando o se mantiene estable.\n\n¿Por qué importa? Una tendencia sostenida suele anticipar problemas operativos o una oportunidad de escalar buenas prácticas." 

        if "problem" in lower:
            return "Estas zonas merecen atención porque combinan nivel bajo actual o deterioro reciente frente a la semana pasada.\n\n¿Por qué importa? Si además son zonas priorizadas, el impacto operativo y comercial puede ser alto." 

        if "crecen" in lower and ("órdenes" in lower or "ordenes" in lower):
            return "Estas zonas vienen acelerando su volumen de órdenes.\n\n¿Por qué importa? Vale la pena revisar qué palancas comerciales u operativas están funcionando allí para replicarlas en otras zonas." 

        return "Aquí tienes el resultado organizado para identificar rápidamente ubicación, segmento y desempeño principal.\n\n¿Por qué importa? Te permite convertir un dato aislado en una decisión operativa concreta."

    def _should_replace_reply_with_summary(self, reply_text: str) -> bool:
        if not reply_text:
            return True
        normalized = reply_text.strip().lower()
        generic_starts = [
            "top zonas en",
            "promedio de",
            "comparación de",
            "comparacion de",
            "evolución reciente de",
            "evolucion reciente de",
            "zonas con buen lead penetration",
            "zonas potencialmente problemáticas",
        ]
        return any(normalized.startswith(item) for item in generic_starts)

    _LLM_OOS_MARKERS = (
        "solo puedo ayudarte con análisis operacionales",
        "no puedo ayudarte con",
        "no puedo responder preguntas sobre",
        "fuera de mi alcance",
        "no tengo información sobre",
        "no es una consulta operacional",
        "solo analizo datos operacionales",
        "no está relacionad",
        "no puedo ayudarte a responder",
        "no es parte de mis capacidades",
    )

    _NO_SUGGESTION_ERRORS = {
        "out_of_scope_fallback", "meta_question", "explanatory_follow_up",
        "vague_briefing", "fallback_not_matched",
    }

    _PCT_EXEMPT_METRICS: frozenset = frozenset({"orders", "gross profit ue"})
    _PCT_EXEMPT_COL_SUBSTRINGS: tuple = ("growth", "gap", "peer", "delta")

    def _format_as_percentage(self, df: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
        # Las métricas se almacenan como decimal [0, 1] en el dataset; se multiplican por 100 para mostrarlas como porcentaje en la UI. 
        # Excepciones: métricas absolutas (Orders, Gross Profit) y columnas de diferencia (gap, delta, growth) que son valores directos
        # aunque caigan en el rango [0, 1]
        if df.empty:
            return df
        if metric and metric.lower() in self._PCT_EXEMPT_METRICS:
            return df
        result = df.copy()
        rename_map: dict[str, str] = {}
        for col in result.select_dtypes(include=[float]).columns:
            if "(%)" in col:
                continue
            if any(p in col.lower() for p in self._PCT_EXEMPT_COL_SUBSTRINGS):
                continue
            series = result[col].dropna()
            if len(series) > 0 and 0 <= float(series.median()) <= 1:
                result[col] = result[col].apply(
                    lambda v: round(v * 100, 2) if pd.notna(v) and 0 <= v <= 1.5 else (round(float(v), 2) if pd.notna(v) else v)
                )
                rename_map[col] = col.replace("_", " ").strip() + " (%)"
        if rename_map:
            result = result.rename(columns=rename_map)
        return result

    def _build_proactive_suggestions(self, payload: dict[str, Any], message: str) -> list[str]:
        if payload.get("error") in self._NO_SUGGESTION_ERRORS:
            return []
        rows = payload.get("data_rows", []) or []
        if not rows:
            return [
                "Explícame este resultado en términos de negocio.",
                "Dame una recomendación ejecutiva basada en este análisis.",
            ]

        first_row = rows[0]
        zone = first_row.get("ZONE") or first_row.get("zone")
        country = first_row.get("COUNTRY") or first_row.get("country")

        numeric_columns = []
        for key, value in first_row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_columns.append(key)
        metric_column = numeric_columns[-1] if numeric_columns else None
        metric_name = metric_column.replace("_", " ") if metric_column else "esta métrica"

        suggestions = ["Explícame este resultado en términos de negocio."]
        if zone:
            suggestions.append(f"Muéstrame la tendencia de 8 semanas para {zone}.")
        if zone and country:
            suggestions.append(f"Compara {zone} contra zonas similares en {country}.")
        elif country:
            suggestions.append(f"Compara este resultado contra el benchmark de {country}.")
        else:
            suggestions.append(f"Compara este resultado contra el benchmark de {metric_name}.")
        return suggestions[:3]

    _INDEPENDENT_QUERY_MARKERS: tuple = (
        "crecen en órdenes", "crecen en ordenes",
        "zonas problemáticas", "zonas problematicas",
        "más crecen", "mas crecen",
        "más órdenes", "mas ordenes",
        "ahora", "cambia", "otra pregunta", "diferente",
        "en cambio", "en vez",
    )

    def _is_independent_query(self, message: str) -> bool:
        lower = message.lower()
        return any(marker in lower for marker in self._INDEPENDENT_QUERY_MARKERS)

    def _fallback_query(self, message: str, history: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        message = self._resolve_contextual_message(message, history)
        text = message.lower()
        history_context = self._context_from_history(history)
        if self._is_independent_query(message):
            metric = self._normalize_metric(message)
            country = self._normalize_country(message)
            zone_name = self._match_zone_from_text(message)
        else:
            metric = self._normalize_metric(message) or history_context.get("metric")
            country = self._normalize_country(message) or history_context.get("country")
            zone_name = self._match_zone_from_text(message) or history_context.get("zone")

        if self._is_meta_question(message):
            return (
                self._answer_meta_question(message),
                {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "meta_question"},
            )

        if self._is_explanatory_follow_up(message) and history:
            return (
                self._answer_explanatory_follow_up(history, message),
                {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "explanatory_follow_up"},
            )

        if not metric:
            similar_metric = self._closest_metric(message)
            if similar_metric:
                return (
                    f"No encuentro esa métrica exacta, pero tengo {similar_metric}. ¿Te sirve?",
                    {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "metric_not_found"},
                )

        if "promedio" in text and metric:
            _col = f"Promedio_{metric.replace(' ', '_')}"
            result = (
                self.df_metrics[self.df_metrics["METRIC"] == metric]
                .drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])
                .groupby("COUNTRY", as_index=False)["L0W_ROLL"]
                .mean()
                .rename(columns={"L0W_ROLL": _col})
                .sort_values(_col, ascending=False)
            )
            _max_markers = ("más alto", "mas alto", "mayor", "el mejor", "lidera", "top 1", "cuál es el mayor", "cual es el mayor")
            _min_markers = ("más bajo", "mas bajo", "menor", "el peor", "el mínimo", "el minimo")
            if any(m in text for m in _max_markers):
                best = result.iloc[0]
                return (
                    f"El país con mayor {metric} es {best['COUNTRY']} con un valor de {round(float(best[_col]), 4)}.",
                    format_result_payload(result.head(1)),
                )
            if any(m in text for m in _min_markers):
                worst = result.iloc[-1]
                return (
                    f"El país con menor {metric} es {worst['COUNTRY']} con un valor de {round(float(worst[_col]), 4)}.",
                    format_result_payload(result.tail(1)),
                )
            payload = format_result_payload(result)
            return f"Promedio de {metric} por país.", payload

        if ("top" in text or "5 zonas" in text or "mayor" in text) and metric:
            _n_match = re.search(r"\b(\d+)\b", text)
            _n = int(_n_match.group(1)) if _n_match else 5
            _n = max(1, min(_n, 50))
            filtered = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
            if country:
                filtered = filtered[filtered["COUNTRY"] == country]
            filtered = filtered.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])
            result = filtered.nlargest(_n, "L0W_ROLL")[["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "L0W_ROLL"]].rename(columns={"L0W_ROLL": metric.replace(' ', '_')})
            payload = format_result_payload(result)
            return f"Top zonas en {metric}.", payload

        if "wealthy" in text and "non wealthy" in text and metric:
            filtered = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
            if country:
                filtered = filtered[filtered["COUNTRY"] == country]
            filtered = filtered.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])
            result = (
                filtered.groupby("ZONE_TYPE", as_index=False)["L0W_ROLL"]
                .mean()
                .rename(columns={"L0W_ROLL": f"Promedio_{metric.replace(' ', '_')}"})
            )
            payload = format_result_payload(result)
            return f"Comparación de {metric} por tipo de zona.", payload

        if ("similar" in text or "benchmark" in text or "compar" in text) and zone_name and metric:
            base = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
            if country:
                base = base[base["COUNTRY"] == country]
            base = base.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])

            focus = base[base["ZONE"] == zone_name].copy()
            if focus.empty:
                return (
                    f"No encontré la zona {zone_name} para comparar con otras similares.",
                    {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "zone_not_found"},
                )

            focus_row = focus.iloc[0]
            peer_zone_type = focus_row.get("ZONE_TYPE")
            peers = base[base["ZONE_TYPE"] == peer_zone_type].copy()
            peers_avg = peers["L0W_ROLL"].mean() if not peers.empty else None
            result = pd.DataFrame(
                [
                    {
                        "ZONE": zone_name,
                        "COUNTRY": focus_row.get("COUNTRY"),
                        "ZONE_TYPE": peer_zone_type,
                        "METRIC": metric,
                        "L0W_ROLL": round(float(focus_row.get("L0W_ROLL")), 4),
                        "PEER_AVG": round(float(peers_avg), 4) if peers_avg is not None and pd.notna(peers_avg) else None,
                        "GAP_VS_PEER": round(float(focus_row.get("L0W_ROLL") - peers_avg), 4) if peers_avg is not None and pd.notna(peers_avg) else None,
                    }
                ]
            )
            payload = format_result_payload(result)
            return f"Comparación de {zone_name} contra zonas similares en {country or focus_row.get('COUNTRY')} para {metric}.", payload

        if ("evoluci" in text or "tendencia" in text or re.search(r"\d+\s+semanas?", text) or "últimas" in text) \
                and not ("crecen" in text and ("órdenes" in text or "ordenes" in text)):
            metric = metric or "Gross Profit UE"
            _base_trend = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
            if country:
                _base_trend = _base_trend[_base_trend["COUNTRY"] == country]
            _base_trend = _base_trend.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])
            if zone_name:
                _zone_trend = _base_trend[_base_trend["ZONE"] == zone_name]
                if _zone_trend.empty:
                    # Intenta realizar una coincidencia parcial sin distinción entre mayúsculas y minúsculas antes de darse por vencido.
                    _zone_trend = _base_trend[_base_trend["ZONE"].str.lower().str.contains(zone_name.lower(), na=False)]
                trend = _zone_trend if not _zone_trend.empty else _base_trend
            else:
                trend = _base_trend
            week_columns = [column for column in self._week_columns_for_request(message, metrics_dataset=True) if column in trend.columns]
            if len(week_columns) < 3:
                week_columns = [c for c in trend.columns if c.endswith("_ROLL")][-8:]
            selected_columns = [column for column in ["COUNTRY", "CITY", "ZONE", "METRIC"] if column in trend.columns] + week_columns
            result = trend[selected_columns].drop_duplicates().head(1 if zone_name else 5)
            payload = format_result_payload(result)
            if payload.get("success") and payload.get("chart") is None:
                payload["chart"] = build_chart_payload(result)
            _scope = zone_name or country or "todas las zonas"
            return f"Evolución reciente de {metric} — {_scope}.", payload

        if "alto" in text and "bajo" in text and "lead" in text and ("perfect" in text or "order" in text):
            lp = self.df_metrics[self.df_metrics["METRIC"] == "Lead Penetration"][["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"]).rename(columns={"L0W_ROLL": "Lead_Penetration"})
            po = self.df_metrics[self.df_metrics["METRIC"] == "Perfect Orders"][["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"]).rename(columns={"L0W_ROLL": "Perfect_Orders"})
            merged = lp.merge(po, on=["COUNTRY", "CITY", "ZONE"])
            result = merged[(merged["Lead_Penetration"] >= merged["Lead_Penetration"].median()) & (merged["Perfect_Orders"] <= merged["Perfect_Orders"].median())].sort_values("Perfect_Orders").head(10)
            payload = format_result_payload(result)
            return "Zonas con buen lead penetration pero mala calidad operacional.", payload

        if "crecen" in text and ("órdenes" in text or "ordenes" in text):
            orders = self.df_orders.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"]).copy() if all(c in self.df_orders.columns for c in ["COUNTRY", "CITY", "ZONE"]) else self.df_orders.drop_duplicates(subset=["ZONE"]).copy()
            orders["growth_5w"] = orders["L0W"] - orders["L5W"]
            result = orders.nlargest(10, "growth_5w")[["COUNTRY", "CITY", "ZONE", "L5W", "L0W", "growth_5w"]]
            payload = format_result_payload(result)
            return "Zonas con mayor crecimiento de órdenes en las últimas 5 semanas.", payload

        if "problem" in text:
            metric = metric or "Perfect Orders"
            filtered = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
            if country:
                filtered = filtered[filtered["COUNTRY"] == country]
            filtered = filtered.drop_duplicates(subset=["COUNTRY", "CITY", "ZONE"])
            filtered["delta"] = filtered["L0W_ROLL"] - filtered["L1W_ROLL"]
            result = filtered.nsmallest(10, ["L0W_ROLL", "delta"])[["COUNTRY", "CITY", "ZONE", "METRIC", "L1W_ROLL", "L0W_ROLL", "delta"]]
            payload = format_result_payload(result)
            return "Zonas potencialmente problemáticas por nivel actual y deterioro reciente.", payload

        has_operational_context = bool(
            metric or country or zone_name
            or any(s in text for s in [
                "zona", "semana", "week", "tendencia", "promedio", "top",
                "evolución", "evolucion", "rappi", "wealthy", "prioritized",
                "ranking", "analisis", "análisis", "benchmark",
            ])
        )
        if not has_operational_context:
            return (
                "Solo analizo datos operacionales de Rappi — métricas, zonas, países y tendencias. "
                "Intenta preguntarme por Lead Penetration, Perfect Orders, Gross Profit UE u otras métricas en un país o zona específica.",
                {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "out_of_scope_fallback"},
            )

        briefing = self._build_daily_briefing()
        return (
            briefing,
            {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "vague_briefing"},
        )

    def _is_vague_message(self, message: str) -> bool:
        lower = message.lower().strip().rstrip("?!¿¡.")
        greetings = {"hola", "hi", "hello", "buenas", "hey", "saludos", "ey", "ola"}
        vague_phrases = [
            "buenos días", "buenos dias", "buenas tardes", "buenas noches",
            "cómo va todo", "como va todo", "qué hay de nuevo", "que hay de nuevo",
            "cómo estamos", "como estamos", "cómo vamos", "como vamos",
            "qué novedades", "que novedades", "cuéntame algo", "cuentame algo",
            "cómo está todo", "como esta todo", "qué pasa", "que pasa",
            "qué tenemos", "que tenemos", "dame un resumen", "dame el resumen",
            "resumen del día", "resumen del dia", "cómo va rappi", "como va rappi",
            "qué me cuentas", "que me cuentas", "alguna novedad", "hay novedades",
            "cómo andamos", "como andamos", "qué onda", "que onda",
        ]
        if lower in greetings:
            return True
        return any(t in lower for t in vague_phrases)

    _CHART_KEYWORDS: tuple = ("gráfica", "grafica", "chart", "visualiza", "graf", "plot", "diagrama")

    def _try_inject_evolution_chart(self, message: str, payload: dict) -> None:
        """If user asked for a chart but payload has none, build one from 8-week history."""
        if payload.get("chart") is not None:
            return
        text = message.lower()
        if not any(kw in text for kw in self._CHART_KEYWORDS):
            return
        metric = self._normalize_metric(message)
        if not metric:
            return
        zone_name = self._match_zone_from_text(message)
        country = self._normalize_country(message)
        trend = self.df_metrics[self.df_metrics["METRIC"] == metric].copy()
        if zone_name:
            trend = trend[trend["ZONE"] == zone_name]
        if country:
            trend = trend[trend["COUNTRY"] == country]
        if trend.empty:
            return
        week_cols = [c for c in trend.columns if c.endswith("_ROLL")][-8:]
        if len(week_cols) < 3:
            return
        id_cols = [c for c in ["COUNTRY", "CITY", "ZONE", "METRIC"] if c in trend.columns]
        result = trend[id_cols + week_cols].drop_duplicates().head(1 if zone_name else 5)
        chart = build_chart_payload(result)
        if chart:
            payload["chart"] = chart

    def _format_data_for_interpretation(self, data_rows: list[dict]) -> str:
        if not data_rows:
            return ""
        lines = []
        for row in data_rows[:5]:
            parts = []
            for k, v in row.items():
                if v is None:
                    continue
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            lines.append(", ".join(parts))
        return "\n".join(lines)[:300]

    def _request_interpretation(self, message: str, payload: dict, history: list[dict] | None = None) -> str:
        if not self.client:
            return ""
        data_rows = payload.get("data_rows") or []
        if not data_rows:
            return ""
        data_summary = self._format_data_for_interpretation(data_rows)
        interp_messages: list[dict] = [
            {
                "role": "system",
                "content": "Eres un analista de operaciones de Rappi. Genera interpretaciones ejecutivas breves y precisas basadas únicamente en los datos proporcionados.",
            },
        ]
        if history:
            interp_messages.extend(history[-4:])
        interp_messages.append({
            "role": "user",
            "content": (
                f"El usuario preguntó: {message}\n\n"
                f"Los datos reales del análisis son:\n{data_summary}\n\n"
                "Escribe 2-3 oraciones de interpretación ejecutiva en español basadas "
                "EXCLUSIVAMENTE en estos datos. No inventes valores ni uses datos distintos a los proporcionados."
            ),
        })
        reply = self._request_analysis(interp_messages, max_tokens=400)
        if not reply or reply == "__RATE_LIMIT__":
            return ""
        return self._sanitize_reply_text(reply)

    def _build_daily_briefing(self) -> str:
        df = self.df_metrics.copy()
        findings: list[str] = []

        if "L0W_ROLL" in df.columns and "L1W_ROLL" in df.columns:
            df_drop = df.dropna(subset=["L0W_ROLL", "L1W_ROLL"]).copy()
            df_drop["delta"] = (df_drop["L0W_ROLL"] - df_drop["L1W_ROLL"]).astype(float)
            df_drop["delta_rank"] = df_drop.groupby("METRIC")["delta"].rank(pct=True)
            worst = df_drop[df_drop["delta"] < 0].nsmallest(1, "delta_rank")
            if not worst.empty:
                row = worst.iloc[0]
                l1_val = abs(float(row["L1W_ROLL"]))
                if l1_val > 0.001:
                    pct = abs(round(float(row["delta"]) / l1_val * 100, 1))
                    change_text = f"{pct}%" if pct <= 200 else f"{abs(round(float(row['delta']), 4))} unidades"
                else:
                    change_text = f"{abs(round(float(row['delta']), 4))} unidades"
                findings.append(
                    f"**{row['METRIC']} en {row['ZONE']} ({row['COUNTRY']})** "
                    f"cayó {change_text} vs la semana pasada — la mayor caída relativa de esta semana."
                )

        if all(c in df.columns for c in ["L0W_ROLL", "L2W_ROLL", "L4W_ROLL"]):
            df_sus = df.dropna(subset=["L0W_ROLL", "L2W_ROLL", "L4W_ROLL"]).copy()
            df_sus["declining"] = (
                (df_sus["L0W_ROLL"] < df_sus["L2W_ROLL"]) & (df_sus["L2W_ROLL"] < df_sus["L4W_ROLL"])
            )
            metric_decline = df_sus.groupby("METRIC")["declining"].sum().sort_values(ascending=False)
            if not metric_decline.empty and int(metric_decline.iloc[0]) > 0:
                worst_metric = str(metric_decline.index[0])
                count = int(metric_decline.iloc[0])
                findings.append(
                    f"**{worst_metric}** muestra deterioro sostenido en {count} zonas "
                    f"durante las últimas 4 semanas — señal de alerta temprana."
                )

        key_metric = next((m for m in ["Perfect Orders", "Lead Penetration", "Gross Profit UE"] if m in self.metrics_catalog), None)
        if key_metric:
            mdf = df[df["METRIC"] == key_metric].dropna(subset=["L0W_ROLL"])
            if not mdf.empty:
                country_avg = mdf.groupby("COUNTRY")["L0W_ROLL"].mean()
                worst_country = str(country_avg.idxmin())
                worst_val = float(country_avg.min())
                findings.append(
                    f"**{key_metric} en {worst_country}** tiene el promedio más bajo de la región: "
                    f"{worst_val:.2%} — requiere atención prioritaria."
                )

        if not findings:
            return (
                "¡Hola! Estoy listo para analizar los datos operacionales de Rappi. "
                "Puedes preguntarme por métricas como Lead Penetration, Perfect Orders o Gross Profit UE "
                "en cualquier país o zona."
            )

        intro = "¡Hola! He analizado los datos mientras entrabas. He detectado estos puntos que requieren tu atención:\n\n"
        body = "\n\n".join(f"{i + 1}. {f}" for i, f in enumerate(findings[:3]))
        outro = "\n\n¿Quieres que profundice en alguno de estos puntos, o tienes otra consulta en mente?"
        return intro + body + outro

    def answer(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        """Resuelve una consulta en lenguaje natural: intenta LLM y cae al fallback determinístico si falla."""
        if self._is_vague_message(message):
            return {
                "reply": self._build_daily_briefing(),
                "data": None,
                "data_rows": [],
                "columns": [],
                "highlights": [],
                "chart": None,
                "suggestions": [
                    "¿Cuáles son las zonas con mayor caída de Perfect Orders esta semana?",
                    "Muéstrame el deterioro sostenido de Lead Penetration en Colombia",
                    "¿Qué zonas High Priority están por debajo del benchmark regional?",
                ],
            }

        if self._is_out_of_scope(message):
            return self._out_of_scope_response()

        reply_text = ""
        payload = None
        _llm_responded = False

        _is_meta = self._is_meta_question(message)

        if self._llm_is_available():
            resolved_message = self._resolve_contextual_message(message, history)
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": resolved_message})

            _rate_limit_reply = {
                "reply": (
                    " Se alcanzó el límite de la API de DeepSeek. "
                    "Por favor actualiza la `DEEPSEEK_API_KEY` en el archivo `.env` y reinicia el servidor."
                ),
                "data": None, "data_rows": [], "columns": [], "highlights": [], "chart": None,
                "suggestions": [],
            }

            if _is_meta:
               
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.extend(history[-8:])
                messages.append({"role": "user", "content": resolved_message})
                llm_reply = self._request_analysis(messages)
                if llm_reply == "__RATE_LIMIT__":
                    self._llm_unavailable_since = time.time()
                    return _rate_limit_reply
                if llm_reply and any(m in llm_reply.lower() for m in self._LLM_OOS_MARKERS):
                    return self._out_of_scope_response()
                _llm_responded = bool(llm_reply)
                reply_text = self._sanitize_reply_text(clean_response(llm_reply)) if llm_reply else ""
            else:
                # Dos llamadas separadas porque una sola mezclaría generación de código e interpretación:
                # la primera produce un DataFrame real, la segunda lo interpreta no al
                code_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": resolved_message},
                ]
                raw_code_reply = self._request_analysis(
                    code_messages + [{
                        "role": "user",
                        "content": (
                            "Responde ÚNICAMENTE con el bloque ```python``` que resuelve la consulta "
                            "usando df_metrics y/o df_orders. Sin texto antes ni después. "
                            "Sin interpretación. Solo el código que termine con result = ..."
                        ),
                    }],
                    max_tokens=1200,
                )
                if raw_code_reply == "__RATE_LIMIT__":
                    self._llm_unavailable_since = time.time()
                    return _rate_limit_reply
                if raw_code_reply and any(m in raw_code_reply.lower() for m in self._LLM_OOS_MARKERS):
                    return self._out_of_scope_response()
                _llm_responded = bool(raw_code_reply)
                code = extract_code(raw_code_reply) if raw_code_reply else None

                if code:
                    payload = run_code(code, self.df_metrics, self.df_orders)
                    if not payload["success"] and payload["error"]:
                        fix_code = self._request_code_only(
                            messages + [
                                {"role": "assistant", "content": raw_code_reply},
                                {"role": "user", "content": f"El código dio este error: {payload['error']}. Corrígelo y devuelve solo un bloque ```python``` válido."},
                            ]
                        )
                        if fix_code:
                            payload = run_code(fix_code, self.df_metrics, self.df_orders)

                    if payload and payload["success"]:
                        # La interpretacion recibe los datos ejecutados, no una estimacon,
                        # aqui si se incluye el historial para resolver referencias contextuales
                        reply_text = self._request_interpretation(message, payload, history)

        # El fallback entra cuando el LLM no generó código válido, el código falló en ejecución o el resultado fue vacío.
        if not payload or not payload["success"]:
            if _llm_responded:
                fallback_reply, fallback_payload = self._fallback_query(message, history)
                if fallback_payload.get("success"):
                    payload = fallback_payload
                else:
                    payload = payload or {"success": False, "data_markdown": "", "data_rows": [], "columns": [], "highlights": [], "chart": None, "error": "llm_text_only"}
                if not reply_text and fallback_reply:
                    reply_text = fallback_reply
            else:
                reply_text, payload = self._fallback_query(message, history)

        if payload["success"]:
            if self._should_replace_reply_with_summary(reply_text):
                reply_text = self._build_executive_summary(payload, message)
            reply_text = f"{reply_text}{self._transparency_footer(message, payload)}"

        self._try_inject_evolution_chart(message, payload)
        proactive = self._build_proactive_suggestions(payload, message)

        return {
            "reply": reply_text,
            "data": payload["data_markdown"] if payload["success"] else None,
            "data_rows": payload.get("data_rows", []) if payload["success"] else [],
            "columns": payload.get("columns", []) if payload["success"] else [],
            "highlights": payload.get("highlights", []) if payload["success"] else [],
            "chart": payload.get("chart"),
            "suggestions": proactive,
        }
