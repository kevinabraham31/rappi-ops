import asyncio
import csv
import io
import json
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from services.chat_service import ChatService
from services.data_loader import load_data
from services.insights_service import generate_report, render_report_html

app = FastAPI(title="Rappi Operations Intelligence")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/img", StaticFiles(directory=os.path.join(BASE_DIR, "img")), name="img")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
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
async def view_excel(request: Request):
    metrics_html = data_bundle.df_metrics.head(200).to_html(index=False, classes="excel-table", border=0)
    orders_html = data_bundle.df_orders.head(200).to_html(index=False, classes="excel-table", border=0)
    return templates.TemplateResponse("excel_view.html", {
        "request": request,
        "filename": os.path.basename(data_bundle.data_path),
        "metrics_html": metrics_html,
        "metrics_total": len(data_bundle.df_metrics),
        "orders_html": orders_html,
        "orders_total": len(data_bundle.df_orders),
    })


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