<div align="center">

<pre>
██████╗  █████╗ ██████╗ ██████╗ ██╗      ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║     ██╔═══██╗██╔══██╗██╔════╝
██████╔╝███████║██████╔╝██████╔╝██║     ██║   ██║██████╔╝███████╗
██╔══██╗██╔══██║██╔═══╝ ██╔═══╝ ██║     ██║   ██║██╔═══╝ ╚════██║
██║  ██║██║  ██║██║     ██║     ██║     ╚██████╔╝██║     ███████║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝
</pre>

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek_Chat-4A90E2?style=flat-square)](https://platform.deepseek.com)
[![Pandas](https://img.shields.io/badge/Pandas-en_memoria-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/Licencia-MIT-22C55E?style=flat-square)](LICENSE)

**Chatbot conversacional para análisis operacional de Rappi.**  
Cualquier persona del equipo puede consultar métricas en español — sin SQL, sin Python.

</div>

---

## Demo

> Escribe una pregunta en lenguaje natural y obtén tablas, gráficos e interpretación ejecutiva en segundos.

![Demo del chat](img/demo_chat.gif)

---

## Reporte de insights automático

> El endpoint `/insights` detecta anomalías, deterioros sostenidos y oportunidades sin que el usuario haga ninguna pregunta.

![Reporte de insights](img/demo_insights.gif)

---

## ¿Qué es esto?

**Rappi Ops** convierte preguntas en lenguaje natural en análisis ejecutivos sobre las métricas semanales de operaciones. El equipo puede explorar **Perfect Orders**, **Lead Penetration**, **Gross Profit UE** y más — sin necesidad de saber SQL ni Python.

```
"¿Qué zonas tienen 4 semanas consecutivas de caída en Perfect Orders?"
         ↓
   Código pandas generado por LLM → ejecutado sobre datos reales → interpretación ejecutiva
```

El bot **nunca inventa números**: primero ejecuta el código y obtiene los datos reales, luego los interpreta. Si el LLM no está disponible, responde igual con fallbacks determinísticos auditables.

---

## Inicio rápido

**Con Docker (recomendado)**

```bash
git clone <URL_DEL_REPO> && cd rappi-ops
cp .env.example .env          # agrega tu DEEPSEEK_API_KEY (opcional)
# coloca el Excel en data/Rappi Operations Analysis Dummy Data.xlsx
docker compose up --build
```

**Sin Docker**

```bash
git clone <URL_DEL_REPO> && cd rappi-ops
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # agrega tu DEEPSEEK_API_KEY (opcional)
# coloca el Excel en data/Rappi Operations Analysis Dummy Data.xlsx
uvicorn main:app --reload
```

Abre → [http://127.0.0.1:8000](http://127.0.0.1:8000)

> Sin `DEEPSEEK_API_KEY` el bot funciona completamente con fallbacks determinísticos. No se necesita ninguna clave para la demo.

---


## Variables de entorno

| Variable | Descripción | Requerida |
|----------|-------------|:---------:|
| `DEEPSEEK_API_KEY` | API key para `deepseek-chat`. Obtener en [platform.deepseek.com](https://platform.deepseek.com). Costo estimado < $0.01 por sesión de demo. Sin ella el bot usa fallbacks determinísticos. | No |

---

## Rutas disponibles

| Ruta | Método | Descripción |
|------|:------:|-------------|
| `/` | `GET` | Interfaz de chat principal |
| `/excel` | `GET` | Vista previa del dataset (primeras 200 filas) |
| `/excel/download` | `GET` | Descarga el Excel original |
| `/insights` | `GET` | Reporte ejecutivo automático — anomalías, tendencias y oportunidades |
| `/insights/download` | `GET` | Descarga el reporte como HTML standalone |
| `/chat` | `POST` | Endpoint del chatbot — recibe `message` + `history`, devuelve respuesta estructurada |
| `/chat/stream` | `POST` | Igual que `/chat` pero con respuesta en streaming (SSE) |
| `/export/csv` | `POST` | Exporta el resultado de la consulta actual como CSV |
| `/docs` | `GET` | Swagger UI con documentación interactiva |

---

## Casos de uso

| Tipo de consulta | Ejemplo |
|-----------------|---------|
| Evolución temporal | *"Muéstrame la evolución de Gross Profit UE en Facatativa las últimas 8 semanas"* |
| Top N zonas | *"¿Cuáles son las 5 zonas con menor Perfect Orders en Colombia?"* |
| Promedio por país | *"¿Cuál es el país con el promedio de Lead Penetration más alto?"* |
| Crecimiento de órdenes | *"¿Cuáles son las zonas que más crecen en órdenes en las últimas 5 semanas?"* |
| Benchmarking | *"Compara Usme contra zonas similares en Colombia"* |
| Deterioro sostenido | *"¿Qué zonas tienen 4 semanas consecutivas de caída en Perfect Orders?"* |
| Recomendación ejecutiva | *"Dame una recomendación ejecutiva basada en este análisis"* |
| Anomalías WoW | *"¿Qué zonas tuvieron el mayor cambio semana a semana?"* |

---

## Arquitectura

Pipeline de dos llamadas LLM para evitar alucinaciones numéricas:

```
Usuario
  └─► chat.html  (SPA vanilla JS · dark theme)
        └─► POST /chat
              └─► ChatService.answer()
                    ├─► _is_out_of_scope()           guardrail sin gastar tokens LLM
                    ├─► Call 1 · DeepSeek            genera código Python — sin historial
                    │     └─► QueryExecutor.run_code()  → pandas → DataFrame real
                    ├─► Call 2 · DeepSeek            interpreta los datos reales + historial
                    └─► Fallback determinístico       _fallback_query() si el LLM falla

Capa de insights  (/insights)
  InsightsService
    ├─► detect_anomalies()             cambios WoW > 10%
    ├─► detect_deteriorating_trends()  4 semanas consecutivas de caída
    ├─► detect_benchmarking()          gaps vs. promedio del mismo tipo de zona
    ├─► detect_correlations()          Pearson entre métricas + pares de negocio
    └─► detect_opportunities()         alta LP + bajo PO o bajo GP
```

**¿Por qué dos llamadas y no una?**  
La primera llamada genera código pandas puro (sin contexto de conversación) para que el LLM no se "contamine" con el historial y fabrique números. La segunda recibe los datos reales ya ejecutados y solo entonces los interpreta. Esto garantiza que cada cifra en la respuesta viene de los datos, no del modelo.

---

## Tests

```bash
pytest tests/ -v
```

Cobertura actual: guardrail de out-of-scope, fallbacks determinísticos, generación de reporte de insights y detección de anomalías WoW.

---

## Decisiones técnicas

| Decisión | Alternativa considerada | Razón |
|----------|------------------------|-------|
| **DeepSeek** como LLM | OpenAI GPT-4, Groq + Llama | ~20× más económico que GPT-4; API compatible con OpenAI SDK; rendimiento suficiente para generación de código pandas |
| **FastAPI** como backend | Flask, Streamlit | Async nativo, Swagger automático en `/docs`, validación Pydantic, mayor rendimiento |
| **Dos llamadas LLM** | Una sola llamada | Evita que el LLM invente valores antes de ver datos reales |
| **Fallbacks determinísticos** | Solo LLM | Si el LLM falla o tiene rate limit, el bot sigue funcionando con respuestas auditables |
| **Jinja2 templates** | HTML en f-strings | Separación entre lógica y presentación; más mantenible |
| **HTML puro + vanilla JS** | React, Vue | Sin build step, sin dependencias de node; deployable como archivo estático |
| **Pandas en memoria** | PostgreSQL, DuckDB | Dataset pequeño (<10k filas); latencia mínima; sin infraestructura adicional |

---

## Mejoras para producción

- **PostgreSQL** — persistencia del historial de conversaciones y auditoría de consultas
- **Redis** — caché de respuestas frecuentes para reducir costos de API
- **JWT + rate limiting** — autenticación por usuario
- **Railway / Render** — CI/CD desde GitHub con variables de entorno gestionadas
- **Fine-tuning** — entrenamiento incremental sobre queries exitosos para reducir dependencia del LLM externo
