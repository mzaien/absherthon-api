from pydantic import BaseModel  # ,constr

# Defines Various Schemas (JSON as calsses) for accepting request and returning response


class Crime(BaseModel):
    crime: str


class predictionResponse(BaseModel):
    type: str
    score: float
    text: str
