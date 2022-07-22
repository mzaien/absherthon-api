from pydantic import BaseModel

# Defines Various Schemas (JSON as calsses) for accepting request and returning response


class Crime(BaseModel):
    crime: str


class predictionResponse(BaseModel):
    type: str
    score: float
    criem_report: str
