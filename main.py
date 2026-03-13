from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import shutil
import os
import tensorflow as tf

from models.cry_detection_model import CryDetectionModel
from models.cry_analysis_model import CryAnalysisModel

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Medical AI API",
    description="Cry analysis + Delivery prediction",
    version="1.0"
)

# -----------------------------
# Load Cry Models
# -----------------------------
detector = CryDetectionModel("model/cry_detection_model.h5")
analyzer = CryAnalysisModel("model/cry_analysis_model.h5")

# -----------------------------
# Load Delivery Model
# -----------------------------
try:
    delivery_model = joblib.load("model/delivery_model.pkl")
    logger.info("Delivery model loaded successfully")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model could not be loaded.")


# -----------------------------
# Cry Analysis Endpoint
# -----------------------------
@app.post("/analyze-cry")
async def analyze_cry(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Detection
    det_result = detector.predict_file(temp_path)

    if not det_result["cry_detected"]:
        os.remove(temp_path)
        return det_result

    # Analysis
    cry_type, class_probs = analyzer.predict(temp_path)

    result = {
        **det_result,
        "cry_type": cry_type,
        "analysis_probs": class_probs
    }

    os.remove(temp_path)

    return result


# -----------------------------
# Input Schema
# -----------------------------
class PatientData(BaseModel):

    PrimaryIndicationforCaesarean_fetal_compromise: int
    PrimaryIndicationforCaesarean_previous_uterine_surgery: int
    No_Of_previous_Csections: int
    Risk_Factors_twins_or_more: int
    Height: float
    Parity: int
    Gestation: float
    PrimaryIndicationforCaesarean_multiple_pregnancy: int
    PrimaryIndicationforCaesarean_antepartum_haemorrhage: int
    PrimaryIndicationforCaesarean_pre_eclampsia: int
    SystolicBloodPressureCuff: float
    DiastolicBloodPressure: float


# -----------------------------
# Delivery Prediction Endpoint
# -----------------------------
@app.post("/predict-delivery")
def predict(data: PatientData):

    try:

        patient = data.dict()

        systolic = patient["SystolicBloodPressureCuff"]
        diastolic = patient["DiastolicBloodPressure"]

        # MAP calculation
        map_value = (systolic + 2 * diastolic) / 3

        model_input = {

            "PrimaryIndicationforCaesarean_fetal compromise":
            patient["PrimaryIndicationforCaesarean_fetal_compromise"],

            "PrimaryIndicationforCaesarean_previous uterine surgery":
            patient["PrimaryIndicationforCaesarean_previous_uterine_surgery"],

            "No_Of_previous_Csections":
            patient["No_Of_previous_Csections"],

            "Risk Factors_twins or more":
            patient["Risk_Factors_twins_or_more"],

            "Height":
            patient["Height"],

            "Parity":
            patient["Parity"],

            "Gestation":
            patient["Gestation"],

            "PrimaryIndicationforCaesarean_multiple pregnancy":
            patient["PrimaryIndicationforCaesarean_multiple_pregnancy"],

            "PrimaryIndicationforCaesarean_antepartum haemorrhage":
            patient["PrimaryIndicationforCaesarean_antepartum_haemorrhage"],

            "PrimaryIndicationforCaesarean_pre eclampsia":
            patient["PrimaryIndicationforCaesarean_pre_eclampsia"],

            "SystolicBloodPressureCuff":
            systolic,

            "MAP":
            map_value
        }

        input_df = pd.DataFrame([model_input])

        prediction = delivery_model.predict(input_df)[0]

        probability = delivery_model.predict_proba(input_df)[0][1]

        result = "C-Section" if prediction == 1 else "Vaginal"

        return {

            "prediction": result,
            "c_section_probability": float(probability),
            "calculated_MAP": map_value

        }

    except Exception as e:

        logger.error(f"Prediction failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Health check (important)
# -----------------------------
@app.get("/health")
def health():

    return {"status": "running"}