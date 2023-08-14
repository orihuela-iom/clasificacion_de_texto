from fastapi import FastAPI, Body, APIRouter, Request, Response, Query
from pydantic import BaseModel
from train_model import make_predit
from typing import Union
from typing import Callable
from fastapi.routing import APIRoute



app = FastAPI()


class RawText(BaseModel):
    ticket_text: str


class Prediction(BaseModel):
    """
    Data class para el formato de respuesta
    """
    prediction: str
    raw_text: str
    probs: Union[dict, None] = None
    key_words: Union[dict, None] = None


@app.post("/predict/")
async def read_query(ticket: RawText) -> Prediction:
    """
    Función principal dentro de la API que recibe una cadena de texto y lo clasifica
    """
    prediction = make_predit(ticket.ticket_text)
    return Prediction(prediction=prediction["Predicted category"],
                      raw_text=prediction["Input text"],
                      probs=prediction["proba"],
                      key_words=prediction["Key words"],
                     )
