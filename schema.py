from pydantic import BaseModel

class PatientData(BaseModel):

    AgeAtStartOfSpell: float
    BMI: float
    WeightMeasured: float
    Height: float
    Gravida: int
    Parity: int
    PreviousCaesarean: int
    GestationalDiabetes: int
    Obese: int
    SystolicBloodPressure: float
    DiastolicBloodPressure: float