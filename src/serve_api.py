import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from .config import MODEL_PATH


class Ticket(BaseModel):
    Body: str
    Department: str
    Tags: list[str] = []


def get_model():
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Ticket Priority API")
model = get_model()


@app.post("/predict")
def predict(ticket: Ticket):
    feats = {
        "Body": [ticket.Body],
        "Department": [ticket.Department],
        "n_tags": [len(ticket.Tags)],
        "len_words": [len(ticket.Body.split())],
    }
    pred = model.predict(feats)[0]
    return {"priority": pred}
