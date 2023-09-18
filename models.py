from pydantic import BaseModel, Field
from typing import Optional, List, Union


class TextGenerationParameters(BaseModel):
    """
    top_k: (Default: None).
    Integer to define the top tokens considered within the sample operation to create new text.

    top_p: (Default: None).
    Float to define the tokens that are within the sample operation of text generation.
    Add tokens in the sample for more probable to least probable until the sum of  the probabilities is greater than top_p.

    temperature: (Default: 1.0). Float (0.0-100.0).
    The temperature of the sampling operation.
    1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability.

    repetition_penalty: (Default: None). Float (0.0-100.0).
    The more a token is used within generation the more it is penalized to not be picked in successive generation passes.

    max_new_tokens: (Default: None). Int (0-250).
    The amount of new tokens to be generated, this does not include the input length it is a estimate of the size of generated text you want. Each new tokens slows down the request, so look for balance between response times and length of text generated.

    max_time: (Default: None). Float (0-120.0).
    The amount of time in seconds that the query should take maximum.
    Network can cause some overhead so it will be a soft limit.
    Use that in combination with max_new_tokens for best results.

    return_full_text: (Default: True). Bool.
    If set to False, the return results will not contain the original query making it easier for prompting.

    num_return_sequences: (Default: 1). Integer.
    The number of proposition you want to be returned.

    do_sample: (Optional: True). Bool.
    Whether or not to use sampling, use greedy decoding otherwise.
    """

    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=100.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=0, le=250)
    max_time: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    return_full_text: Optional[bool] = Field(default=True)
    num_return_sequences: Optional[int] = Field(default=1, ge=1)
    do_sample: Optional[bool] = Field(default=True)

    class Config:
        orm_mode = True


class PayloadOptions(BaseModel):
    use_cache: bool = False
    wait_for_model: bool = True

    class Config:
        orm_mode = True


class InputPayload(BaseModel):
    inputs: Union[str, List[str]]
    parameters: TextGenerationParameters = Field(
        default_factory=TextGenerationParameters
    )
    options: PayloadOptions = Field(default_factory=PayloadOptions)

    class Config:
        orm_mode = True


class Output(BaseModel):
    generated_text: str

    class Config:
        orm_mode = True
