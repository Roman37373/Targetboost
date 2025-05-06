# main.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
from custom_transformers import FeatureEngineer, HitPagePathTransformer, SafeCatBoostEncoder
import io
from typing import Optional

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load pipelines for different models
pipelines = {
    "model1": joblib.load('final_conversion_pipeline_lead.pkl'),
    "model2": joblib.load('final_conversion_pipeline_engagement.pkl'),
    "model3": joblib.load('final_conversion_pipeline_upsell.pkl'),
    "model4": joblib.load('final_conversion_pipeline_churn.pkl ')
}

model_names = {
    "model1": "Прогнозирование ключевых точек конверсии",
    "model2": "Анализ пользовательского опыта",
    "model3": "Оптимизация дополнительных сервисов",
    "model4": "Предотвращение оттока пользователей"
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-page")
async def predict_page(
    request: Request,
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    model: str = Form(...)
):
    if not train_file or not test_file:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Пожалуйста, загрузите оба файла"},
            status_code=400
        )

    try:
        # Read and decode files
        sessions_df = pd.read_csv(io.StringIO((await train_file.read()).decode('utf-8')))
        hits_df = pd.read_csv(io.StringIO((await test_file.read()).decode('utf-8')))

        if 'session_id' not in sessions_df.columns or 'session_id' not in hits_df.columns:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Требуется столбец session_id"},
                status_code=400
            )

        # Process data with selected pipeline
        merged = pd.merge(sessions_df, hits_df, on='session_id', how='left')
        pipeline = pipelines[model]
        proba = pipeline.predict_proba(merged)[:, 1]
        pred = (proba > 0.5).astype(int)

        results = []
        for i in range(len(merged)):
            results.append({
                'session_id': merged.iloc[i]['session_id'],
                'probability': float(proba[i]),
                'prediction': int(pred[i]),
                'model_name': model_names[model]
            })

        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "results": results,
                "model_name": model_names[model]
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5555)