import asyncio
import csv
import io
import json
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.chat_service import ChatService
from services.data_loader import load_data
from services.insights_service import generate_report, render_report_html

app = FastAPI(title="Rappi Operations Intelligence")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/img", StaticFiles(directory=os.path.join(BASE_DIR, "img")), name="img")
data_bundle = load_data(BASE_DIR)
chat_service = ChatService(
    data_bundle.df_metrics,
    data_bundle.df_orders,
    data_bundle.metrics_catalog,
)


class ChatRequest(BaseModel):
    message: str
    history: list = Field(default_factory=list)


@app.post(
    "/chat",
    summary="Enviar mensaje al chatbot",
    description="Recibe un mensaje en lenguaje natural y devuelve una respuesta estructurada con texto ejecutivo, tabla de datos, gráfico y sugerencias proactivas. El historial de conversación se pasa en el body para mantener contexto entre turnos.",
)
async def chat(req: ChatRequest):
    response = await asyncio.to_thread(chat_service.answer, req.message, req.history)
    return JSONResponse(response)


@app.post(
    "/chat/stream",
    summary="Enviar mensaje al chatbot con respuesta en streaming",
    description="Igual que /chat pero devuelve la respuesta como SSE. Emite eventos 'chunk' con fragmentos de texto y un evento 'final' con el payload completo (data_rows, chart, suggestions).",
)
async def chat_stream(req: ChatRequest):
    async def generate():
        response = await asyncio.to_thread(chat_service.answer, req.message, req.history)
        reply = response.get("reply") or ""
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.018 if word.endswith(('.', '!', '?')) else 0.006)
        final = {k: v for k, v in response.items() if k != "reply"}
        final["type"] = "final"
        final["reply"] = reply
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/",
    summary="Interfaz de chat",
    description="Devuelve la SPA de chat principal (chat.html). Desde aquí el usuario puede hacer preguntas en lenguaje natural sobre los datos operacionales de Rappi.",
)
async def root():
    return FileResponse(os.path.join(BASE_DIR, "chat.html"))


@app.get(
    "/excel",
    summary="Vista previa del dataset",
    description="Muestra las primeras 200 filas de cada hoja del Excel cargado (RAW_INPUT_METRICS y RAW_ORDERS) en formato HTML navegable. Útil para verificar que los datos se cargaron correctamente.",
)
async def view_excel():
    metrics_html = data_bundle.df_metrics.head(200).to_html(index=False, classes="excel-table", border=0)
    orders_html = data_bundle.df_orders.head(200).to_html(index=False, classes="excel-table", border=0)

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Vista del Excel</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #0f0f0f; color: #ececec; }}
    .page {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 20px; }}
    .title h1 {{ margin: 0; font-size: 1.5rem; }}
    .title p {{ margin: 6px 0 0; color: #999; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .btn {{ text-decoration: none; background: #ff441f; color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; }}
    .btn.secondary {{ background: #1e1e1e; border: 1px solid #333; }}
    .sheet {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 14px; margin-bottom: 20px; overflow: hidden; }}
    .sheet-header {{ padding: 14px 16px; border-bottom: 1px solid #2a2a2a; background: #1a1a1a; }}
    .sheet-header h2 {{ margin: 0; font-size: 1rem; }}
    .sheet-header p {{ margin: 6px 0 0; color: #888; font-size: 0.85rem; }}
    .table-wrap {{ overflow: auto; max-height: 70vh; }}
    table.excel-table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    table.excel-table th, table.excel-table td {{ padding: 10px 12px; border-bottom: 1px solid #2a2a2a; border-right: 1px solid #222; text-align: left; font-size: 0.82rem; white-space: nowrap; }}
    table.excel-table th {{ position: sticky; top: 0; background: #202020; color: #fff; z-index: 1; }}
    table.excel-table td {{ color: #d4d4d4; }}
    .note {{ color: #888; font-size: 0.82rem; margin-bottom: 18px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div class="title">
        <h1>Vista del Excel</h1>
        <p>Archivo: {os.path.basename(data_bundle.data_path)}</p>
      </div>
      <div class="actions">
        <a class="btn secondary" href="/">Volver al chat</a>
        <a class="btn secondary" href="/insights" target="_blank" rel="noopener noreferrer">Ver reporte</a>
        <a class="btn" href="/excel/download">Descargar Excel</a>
      </div>
    </div>
    <div class="note">Se muestran las primeras 200 filas de cada hoja para revisión rápida en la demo.</div>
    <section class="sheet">
      <div class="sheet-header">
        <h2>Hoja: RAW_INPUT_METRICS</h2>
        <p>{len(data_bundle.df_metrics)} filas totales</p>
      </div>
      <div class="table-wrap">{metrics_html}</div>
    </section>
    <section class="sheet">
      <div class="sheet-header">
        <h2>Hoja: RAW_ORDERS</h2>
        <p>{len(data_bundle.df_orders)} filas totales</p>
      </div>
      <div class="table-wrap">{orders_html}</div>
    </section>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get(
    "/excel/download",
    summary="Descargar Excel original",
    description="Descarga el archivo Excel fuente tal como fue cargado al servidor.",
)
async def download_excel():
    return FileResponse(
        data_bundle.data_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(data_bundle.data_path),
    )


@app.get(
    "/insights",
    summary="Reporte ejecutivo de insights",
    description="Genera y devuelve el reporte HTML con anomalías WoW, deterioros sostenidos, benchmarking, correlaciones y oportunidades detectadas automáticamente sobre el dataset completo.",
)
async def insights_report():
    report = generate_report(data_bundle.df_metrics, data_bundle.df_orders)
    return HTMLResponse(render_report_html(report, data_bundle.df_metrics, data_bundle.df_orders))


@app.post(
    "/export/csv",
    summary="Exportar resultado como CSV",
    description="Re-ejecuta la consulta enviada en el body y devuelve los datos resultantes como archivo CSV descargable. Requiere que la consulta produzca una tabla de datos.",
)
async def export_csv(req: ChatRequest):
    response = chat_service.answer(req.message, req.history)
    rows = response.get("data_rows", [])
    columns = response.get("columns", [])
    if not rows or not columns:
        return JSONResponse({"error": "No hay datos para exportar."}, status_code=400)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rappi_analisis.csv"},
    )


@app.get(
    "/insights/download",
    summary="Descargar reporte de insights como HTML",
    description="Genera el reporte ejecutivo y lo devuelve como archivo HTML standalone descargable, listo para compartir o imprimir como PDF sin conexión a internet.",
)
async def download_insights_html():
    report = generate_report(data_bundle.df_metrics, data_bundle.df_orders)
    html_content = render_report_html(report, data_bundle.df_metrics, data_bundle.df_orders)
    return StreamingResponse(
        iter([html_content.encode("utf-8")]),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=rappi_insights_report.html"},
    )