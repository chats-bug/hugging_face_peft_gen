from finetuned_model import FinetunedModel
from models import InputPayload
from transformers import (
    BitsAndBytesConfig,
)
from typing import Union
import torch


MODEL_REPO_ID = "chats-bug/test-llama2-7b-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

finetuned_model = FinetunedModel(
    model_id=MODEL_REPO_ID,
    quantization_config=quantization_config,
    load_in_4bit=True
)


# Some Helper functions
def process_output(llm_responses: Union[str, list[str]]):
	if isinstance(llm_responses, str):
		llm_responses = [llm_responses]
	try:
		outputs = []
		for llm_response in llm_responses:
			llm_response = llm_response.strip()
			output = llm_response.split("OUTPUT:\n")[1]
			output = output.split("]")[0]+"]"
			outputs.append(output)
		return outputs
	except Exception as e:
		print(f"Error processing the response: {e}")
		return llm_response


def inference(payload: InputPayload, superagi_task_gen: bool = False):
    generated_text = finetuned_model.generate(
        message=payload.inputs,
        max_new_tokens=payload.parameters.max_new_tokens,
        temperature=payload.parameters.temperature,
        top_p=payload.parameters.top_p,
        num_return_sequences=payload.parameters.num_return_sequences,
        device=DEVICE
	)
    if superagi_task_gen:
        generated_text = process_output(generated_text)
    return generated_text
