from finetuned_model import FinetunedModel
from hf_models import HfModel
from models import InputPayload
import os
from transformers import BitsAndBytesConfig
from typing import Union
import torch


MODEL_REPO_ID = os.environ.get("MODEL_ID")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

# Edit the quantization config here
# For more information on the quantization config, refer to huggingface documentation
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

####################################
# Using a finetuned model (using peft adapters)
####################################
model = FinetunedModel(
    model_id=MODEL_REPO_ID, quantization_config=quantization_config, load_in_4bit=True
)

####################################
# To use a normal model (not using peft adapters), uncomment and use the following code:
####################################
# model = HfModel(
#     model_id=MODEL_REPO_ID, 
#     quantization_config=quantization_config, 
#     load_in_4bit=LOAD_IN_4BIT,
#     load_in_8bit=LOAD_IN_8BIT,
# ) # Pass any other arguments to the model here


# Some Helper functions
def process_output(llm_responses: Union[str, list[str]]):
    if isinstance(llm_responses, str):
        llm_responses = [llm_responses]
    try:
        outputs = []
        for llm_response in llm_responses:
            llm_response = llm_response.strip()
            output = llm_response.split("OUTPUT:\n")[1]
            output = output.split("]")[0] + "]"
            outputs.append(output)
        return outputs
    except Exception as e:
        print(f"Error processing the response: {e}")
        return llm_response


def inference(payload: InputPayload):
    generated_text = model.generate(
        message=payload.inputs,
        max_new_tokens=payload.parameters.max_new_tokens,
        temperature=payload.parameters.temperature,
        top_p=payload.parameters.top_p,
        num_return_sequences=payload.parameters.num_return_sequences,
        device=DEVICE,
    )
    if payload.superagi_task_gen:
        generated_text = process_output(generated_text)
    return generated_text
