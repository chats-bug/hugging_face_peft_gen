from fastapi import FastAPI
from typing import List

from models import InputPayload, Output
from generation import inference


app = FastAPI()


@app.get("/")
def read_root() -> List[Output]:
    return [Output(generated_text="Welcome to the Huggingface mock API!")]


@app.post("/generate")
def generate_text(payload: InputPayload) -> List[Output]:
    output = inference(payload)
    return_value: List[Output] = []
    for o in output:
        return_value.append(Output(generated_text=o))
    return return_value
