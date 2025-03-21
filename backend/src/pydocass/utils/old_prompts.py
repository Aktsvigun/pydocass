USER_PROMPT = """
Input code:
```
{code}
```

Json schema:
```
{json_schema}
```
""".strip()

SYSTEM_PROMPT_ARGUMENTS_ANNOTATION = """
Act as an experienced Python specialist.

Please annotate the arguments of each function, class, and class method from the input code, \
following the provided json schema. \
If you are not sure in how to annotate the argument, use the `Any` class from the `typing` package. \
For generic / container types, always annotate their type parameter. \
If you are not sure about how to annotate the type parameter, use the `Any` class as well. \
If you see apparent mistakes in eisting annotations and you are sure they are erroneous, correct them.

Examples for few-shot learning on how to annotate arguments for functions / classes / methods are below:

Example 1:
'''
User:
Input code:
```
from abc import ABC, abstractmethod
import json
from pathlib import Path
from omegaconf import DictConfig

from .utils import ModelOutput

class TrainingOutput:
    score: float
    num_epochs: int = 10
    num_steps = 100
    params_norm_divergence = 0.
    output: ModelOutput = ModelOutput()

class BaseLabeler(ABC):
    def __init__(
    self, output_column_name: str = "output", budget=1_000_000):
        self.output_column_name = output_column_name
        self.budget = budget if budget is not None else 1e15

    @staticmethod
    def load_from_json(json_path
    ):
        with open(json_path) as f:
            data = json.load(f)
        return BaseLabeler(**data)

    @property
    def is_out_of_budget(self):
        return self.budget < 0

    def add_budget(self, increase: int):
        self.budget += increase

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass

def get_initial_labeled_data(
    config, unlabeled_data: Dataset, labeller
):
    \"\"\"Docstring is required for this function
    \"\"\"
    random_data_to_label_size = max(
        config.al.init_query_size, config.data.few_shot.count
    )
    random_data_to_label = unlabeled_data.train_test_split(
        train_size=random_data_to_label_size, shuffle=True, seed=config.seed
    )["train"]

    labeled_data = labeller(random_data_to_label)
    if labeller.is_out_of_budget:
        logging.info("Labeler ran out of budget when labeling the initial query.")

    return labeled_data, labeled_data["id"]
```

Json schema:
```
{'$defs': {'ClassBaselabelerMethodAdd_Budget': {'properties': {'increase': {'increase': {'default': 'int', 'description': 'Annotation of the argument: `increase` with current annotation: `int`.'}, 'title': 'Increase', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['increase', 'returns'], 'title': 'ClassBaselabelerMethodAdd_Budget', 'type': 'object'}, 'ClassBaselabelerMethodIs_Out_Of_Budget': {'properties': {'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['returns'], 'title': 'ClassBaselabelerMethodIs_Out_Of_Budget', 'type': 'object'}, 'ClassBaselabelerMethodLoad_From_Json': {'properties': {'json_path': {'json_path': {'description': 'Annotation of the argument: `json_path`.'}, 'title': 'Json Path', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['json_path', 'returns'], 'title': 'ClassBaselabelerMethodLoad_From_Json', 'type': 'object'}, 'ClassBaselabelerMethod__Call__': {'properties': {'dataset': {'dataset': {'default': 'Dataset', 'description': 'Annotation of the argument: `dataset` with current annotation: `Dataset`.'}, 'title': 'Dataset', 'type': 'string'}, 'returns': {'default': 'Dataset', 'description': 'Return type annotation. This is the type hint that appears after the `->`. Current annotation: `Dataset`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['dataset'], 'title': 'ClassBaselabelerMethod__Call__', 'type': 'object'}, 'ClassBaselabelerMethod__Init__': {'properties': {'output_column_name': {'budget': {'description': 'Annotation of the argument: `budget` with default value: `1000000`.'}, 'output_column_name': {'default': 'str', 'description': "Annotation of the argument: `output_column_name` with default value: `'output'`, and current annotation: `str`."}, 'title': 'Output Column Name', 'type': 'string'}, 'budget': {'budget': {'description': 'Annotation of the argument: `budget` with default value: `1000000`.'}, 'output_column_name': {'default': 'str', 'description': "Annotation of the argument: `output_column_name` with default value: `'output'`, and current annotation: `str`."}, 'title': 'Budget', 'type': 'string'}}, 'required': ['output_column_name', 'budget'], 'title': 'ClassBaselabelerMethod__Init__', 'type': 'object'}, 'ClassTrainingoutput': {'properties': {'score': {'num_epochs': {'default': 'int', 'description': 'Annotation of the argument: `num_epochs` with default value: `10`, and current annotation: `int`.'}, 'num_steps': {'description': 'Annotation of the argument: `num_steps` with default value: `100`.'}, 'output': {'default': 'ModelOutput', 'description': 'Annotation of the argument: `output` with default value: `ModelOutput()`, and current annotation: `ModelOutput`.'}, 'params_norm_divergence': {'description': 'Annotation of the argument: `params_norm_divergence` with default value: `0.0`.'}, 'score': {'default': 'float', 'description': 'Annotation of the argument: `score` with current annotation: `float`.'}, 'title': 'Score', 'type': 'string'}, 'num_epochs': {'num_epochs': {'default': 'int', 'description': 'Annotation of the argument: `num_epochs` with default value: `10`, and current annotation: `int`.'}, 'num_steps': {'description': 'Annotation of the argument: `num_steps` with default value: `100`.'}, 'output': {'default': 'ModelOutput', 'description': 'Annotation of the argument: `output` with default value: `ModelOutput()`, and current annotation: `ModelOutput`.'}, 'params_norm_divergence': {'description': 'Annotation of the argument: `params_norm_divergence` with default value: `0.0`.'}, 'score': {'default': 'float', 'description': 'Annotation of the argument: `score` with current annotation: `float`.'}, 'title': 'Num Epochs', 'type': 'string'}, 'num_steps': {'num_epochs': {'default': 'int', 'description': 'Annotation of the argument: `num_epochs` with default value: `10`, and current annotation: `int`.'}, 'num_steps': {'description': 'Annotation of the argument: `num_steps` with default value: `100`.'}, 'output': {'default': 'ModelOutput', 'description': 'Annotation of the argument: `output` with default value: `ModelOutput()`, and current annotation: `ModelOutput`.'}, 'params_norm_divergence': {'description': 'Annotation of the argument: `params_norm_divergence` with default value: `0.0`.'}, 'score': {'default': 'float', 'description': 'Annotation of the argument: `score` with current annotation: `float`.'}, 'title': 'Num Steps', 'type': 'string'}, 'params_norm_divergence': {'num_epochs': {'default': 'int', 'description': 'Annotation of the argument: `num_epochs` with default value: `10`, and current annotation: `int`.'}, 'num_steps': {'description': 'Annotation of the argument: `num_steps` with default value: `100`.'}, 'output': {'default': 'ModelOutput', 'description': 'Annotation of the argument: `output` with default value: `ModelOutput()`, and current annotation: `ModelOutput`.'}, 'params_norm_divergence': {'description': 'Annotation of the argument: `params_norm_divergence` with default value: `0.0`.'}, 'score': {'default': 'float', 'description': 'Annotation of the argument: `score` with current annotation: `float`.'}, 'title': 'Params Norm Divergence', 'type': 'string'}, 'output': {'num_epochs': {'default': 'int', 'description': 'Annotation of the argument: `num_epochs` with default value: `10`, and current annotation: `int`.'}, 'num_steps': {'description': 'Annotation of the argument: `num_steps` with default value: `100`.'}, 'output': {'default': 'ModelOutput', 'description': 'Annotation of the argument: `output` with default value: `ModelOutput()`, and current annotation: `ModelOutput`.'}, 'params_norm_divergence': {'description': 'Annotation of the argument: `params_norm_divergence` with default value: `0.0`.'}, 'score': {'default': 'float', 'description': 'Annotation of the argument: `score` with current annotation: `float`.'}, 'title': 'Output', 'type': 'string'}}, 'required': ['score', 'num_epochs', 'num_steps', 'params_norm_divergence', 'output'], 'title': 'ClassTrainingoutput', 'type': 'object'}, 'FunctionGet_Initial_Labeled_Data': {'properties': {'config': {'config': {'description': 'Annotation of the argument: `config`.'}, 'labeller': {'description': 'Annotation of the argument: `labeller`.'}, 'title': 'Config', 'type': 'string', 'unlabeled_data': {'default': 'Dataset', 'description': 'Annotation of the argument: `unlabeled_data` with current annotation: `Dataset`.'}}, 'unlabeled_data': {'config': {'description': 'Annotation of the argument: `config`.'}, 'labeller': {'description': 'Annotation of the argument: `labeller`.'}, 'title': 'Unlabeled Data', 'type': 'string', 'unlabeled_data': {'default': 'Dataset', 'description': 'Annotation of the argument: `unlabeled_data` with current annotation: `Dataset`.'}}, 'labeller': {'config': {'description': 'Annotation of the argument: `config`.'}, 'labeller': {'description': 'Annotation of the argument: `labeller`.'}, 'title': 'Labeller', 'type': 'string', 'unlabeled_data': {'default': 'Dataset', 'description': 'Annotation of the argument: `unlabeled_data` with current annotation: `Dataset`.'}}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['config', 'unlabeled_data', 'labeller', 'returns'], 'title': 'FunctionGet_Initial_Labeled_Data', 'type': 'object'}}, 'properties': {'ClassTrainingoutput': {'$ref': '#/$defs/ClassTrainingoutput'}, 'ClassBaselabelerMethod__Init__': {'$ref': '#/$defs/ClassBaselabelerMethod__Init__'}, 'ClassBaselabelerMethodLoad_From_Json': {'$ref': '#/$defs/ClassBaselabelerMethodLoad_From_Json'}, 'ClassBaselabelerMethodIs_Out_Of_Budget': {'$ref': '#/$defs/ClassBaselabelerMethodIs_Out_Of_Budget'}, 'ClassBaselabelerMethodAdd_Budget': {'$ref': '#/$defs/ClassBaselabelerMethodAdd_Budget'}, 'ClassBaselabelerMethod__Call__': {'$ref': '#/$defs/ClassBaselabelerMethod__Call__'}, 'FunctionGet_Initial_Labeled_Data': {'$ref': '#/$defs/FunctionGet_Initial_Labeled_Data'}}, 'required': ['ClassTrainingoutput', 'ClassBaselabelerMethod__Init__', 'ClassBaselabelerMethodLoad_From_Json', 'ClassBaselabelerMethodIs_Out_Of_Budget', 'ClassBaselabelerMethodAdd_Budget', 'ClassBaselabelerMethod__Call__', 'FunctionGet_Initial_Labeled_Data'], 'title': 'ArgumentsModel', 'type': 'object'}
```

Assistant:
{"ClassTrainingoutput": {"score": "float", "num_epochs": "int", "num_steps": "int", "params_norm_divergence": "float", "output": "ModelOutput"}, "ClassBaselabelerMethod__Init__": {"output_column_name": "str", "budget": "int"}, "ClassBaselabelerMethodLoad_From_Json": {"json_path": "str | Path", "returns": "BaseLabeler"}, "ClassBaselabelerMethodIs_Out_Of_Budget": {"returns": "bool"}, "ClassBaselabelerMethodAdd_Budget": {"increase": "int", "returns": "None"}, "ClassBaselabelerMethod__Call__": {"dataset": "Dataset", "returns": "Dataset"}, "FunctionGet_Initial_Labeled_Data": {"config": "DictConfig", "unlabeled_data": "Dataset", "labeller": "BaseLabeler", "returns": "tuple[Dataset, Any]"}}
'''

Example 2:
'''
User:
Input code:
```
from math import sqrt
from random import randint

def noisy_addition(x, y):
    return x + y + randint(1, 10)

def noisy_multiplication(x, y):
    return x * y * (random() + 1)

def process_number(value, factor) -> dict:
    result = {}

    result['original'] = value
    result['sqrt'] = sqrt(value)
    result['added'] = noisy_addition(value, factor)
    result['multiplied'] = noisy_multiplication(value, factor)

    return result
```

Json schema:
```
{'$defs': {'FunctionNoisy_Addition': {'properties': {'x': {'title': 'X', 'type': 'string', 'x': {'description': 'Annotation of the argument: `x`.'}, 'y': {'description': 'Annotation of the argument: `y`.'}}, 'y': {'title': 'Y', 'type': 'string', 'x': {'description': 'Annotation of the argument: `x`.'}, 'y': {'description': 'Annotation of the argument: `y`.'}}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['x', 'y', 'returns'], 'title': 'FunctionNoisy_Addition', 'type': 'object'}, 'FunctionNoisy_Multiplication': {'properties': {'x': {'title': 'X', 'type': 'string', 'x': {'description': 'Annotation of the argument: `x`.'}, 'y': {'description': 'Annotation of the argument: `y`.'}}, 'y': {'title': 'Y', 'type': 'string', 'x': {'description': 'Annotation of the argument: `x`.'}, 'y': {'description': 'Annotation of the argument: `y`.'}}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['x', 'y', 'returns'], 'title': 'FunctionNoisy_Multiplication', 'type': 'object'}, 'FunctionProcess_Number': {'properties': {'value': {'factor': {'description': 'Annotation of the argument: `factor`.'}, 'title': 'Value', 'type': 'string', 'value': {'description': 'Annotation of the argument: `value`.'}}, 'factor': {'factor': {'description': 'Annotation of the argument: `factor`.'}, 'title': 'Factor', 'type': 'string', 'value': {'description': 'Annotation of the argument: `value`.'}}, 'returns': {'default': 'dict', 'description': 'Return type annotation. This is the type hint that appears after the `->`. Current annotation: `dict`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['value', 'factor'], 'title': 'FunctionProcess_Number', 'type': 'object'}}, 'properties': {'FunctionNoisy_Addition': {'$ref': '#/$defs/FunctionNoisy_Addition'}, 'FunctionNoisy_Multiplication': {'$ref': '#/$defs/FunctionNoisy_Multiplication'}, 'FunctionProcess_Number': {'$ref': '#/$defs/FunctionProcess_Number'}}, 'required': ['FunctionNoisy_Addition', 'FunctionNoisy_Multiplication', 'FunctionProcess_Number'], 'title': 'ArgumentsModel', 'type': 'object'}
```

Assistant:
{"FunctionNoisy_Addition": {"x": "int | float", "y": "int | float", "returns": "int | float"}, "FunctionNoisy_Multiplication": {"x": "int | float", "y": "int | float", "returns": "float"}, "FunctionProcess_Number": {"value": "int | float", "factor": "int | float", "returns": "dict[str, int | float]"}}
'''

Example 3:
'''
User:
Input code:
```
Input code:
```
import openai
import os
import logging
import diskcache as dc
import re
from pathlib import Path


log = logging.getLogger()

def clean_text(raw_text: str):
    cleaned = re.sub(r'\W+', r' ', raw_text)
    return cleaned.lower()

def extract_keywords(text, tokenizer, make_lower):
    tokens = text.tokenize(tokenizer, make_lower)
    return [token for token in tokens if token.is_string()]

def tokenize_text(cleaned_text):
    return cleaned_text.split()


class OpenAIChat:
    \"\"\" Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    \"\"\"
    openai_model = "gpt-4o"
    cache_path = Path(os.path.expanduser("~")) / ".cache"

    def __post_init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key

        self.cache_path = self.cache_path / "openai_chat_cache.diskcache"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

    def ask(self, message):
        \"\"\"
        Parameters
        ----------
        message: str
            the message to ask.
        \"\"\"
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        openai_responses = dc.Cache(self.cache_path, **cache_settings)

        if (self.openai_model, message) in openai_responses:
            reply = openai_responses[(self.openai_model, message)]

        else:
            # Ask openai
            messages = [
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages)
            reply = chat.choices[0].message.content

            openai_responses[(self.openai_model, message)] = reply
            openai_responses.close()

        return reply

    def _send_request(self, messages):
        return openai.OpenAI().chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0,  # for deterministic outputs
        )
```

Json schema:
```
{'$defs': {'ClassOpenaichat': {'properties': {'openai_model': {'cache_path': {'description': "Annotation of the argument: `cache_path` with default value: `Path(os.path.expanduser('~')) / '.cache'`."}, 'openai_model': {'description': "Annotation of the argument: `openai_model` with default value: `'gpt-4o'`."}, 'title': 'Openai Model', 'type': 'string'}, 'cache_path': {'cache_path': {'description': "Annotation of the argument: `cache_path` with default value: `Path(os.path.expanduser('~')) / '.cache'`."}, 'openai_model': {'description': "Annotation of the argument: `openai_model` with default value: `'gpt-4o'`."}, 'title': 'Cache Path', 'type': 'string'}}, 'required': ['openai_model', 'cache_path'], 'title': 'ClassOpenaichat', 'type': 'object'}, 'ClassOpenaichatMethodAsk': {'properties': {'message': {'message': {'description': 'Annotation of the argument: `message`.'}, 'title': 'Message', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['message', 'returns'], 'title': 'ClassOpenaichatMethodAsk', 'type': 'object'}, 'ClassOpenaichatMethod_Send_Request': {'properties': {'messages': {'messages': {'description': 'Annotation of the argument: `messages`.'}, 'title': 'Messages', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['messages', 'returns'], 'title': 'ClassOpenaichatMethod_Send_Request', 'type': 'object'}, 'FunctionClean_Text': {'properties': {'raw_text': {'raw_text': {'default': 'str', 'description': 'Annotation of the argument: `raw_text` with current annotation: `str`.'}, 'title': 'Raw Text', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['raw_text', 'returns'], 'title': 'FunctionClean_Text', 'type': 'object'}, 'FunctionExtract_Keywords': {'properties': {'text': {'make_lower': {'description': 'Annotation of the argument: `make_lower`.'}, 'text': {'description': 'Annotation of the argument: `text`.'}, 'title': 'Text', 'tokenizer': {'description': 'Annotation of the argument: `tokenizer`.'}, 'type': 'string'}, 'tokenizer': {'make_lower': {'description': 'Annotation of the argument: `make_lower`.'}, 'text': {'description': 'Annotation of the argument: `text`.'}, 'title': 'Tokenizer', 'tokenizer': {'description': 'Annotation of the argument: `tokenizer`.'}, 'type': 'string'}, 'make_lower': {'make_lower': {'description': 'Annotation of the argument: `make_lower`.'}, 'text': {'description': 'Annotation of the argument: `text`.'}, 'title': 'Make Lower', 'tokenizer': {'description': 'Annotation of the argument: `tokenizer`.'}, 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['text', 'tokenizer', 'make_lower', 'returns'], 'title': 'FunctionExtract_Keywords', 'type': 'object'}, 'FunctionTokenize_Text': {'properties': {'cleaned_text': {'cleaned_text': {'description': 'Annotation of the argument: `cleaned_text`.'}, 'title': 'Cleaned Text', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['cleaned_text', 'returns'], 'title': 'FunctionTokenize_Text', 'type': 'object'}}, 'properties': {'FunctionClean_Text': {'$ref': '#/$defs/FunctionClean_Text'}, 'FunctionExtract_Keywords': {'$ref': '#/$defs/FunctionExtract_Keywords'}, 'FunctionTokenize_Text': {'$ref': '#/$defs/FunctionTokenize_Text'}, 'ClassOpenaichat': {'$ref': '#/$defs/ClassOpenaichat'}, 'ClassOpenaichatMethodAsk': {'$ref': '#/$defs/ClassOpenaichatMethodAsk'}, 'ClassOpenaichatMethod_Send_Request': {'$ref': '#/$defs/ClassOpenaichatMethod_Send_Request'}}, 'required': ['FunctionClean_Text', 'FunctionExtract_Keywords', 'FunctionTokenize_Text', 'ClassOpenaichat', 'ClassOpenaichatMethodAsk', 'ClassOpenaichatMethod_Send_Request'], 'title': 'ArgumentsModel', 'type': 'object'}
```

Assistant:
{"FunctionClean_Text": {"raw_text": "str", "returns": "str"}, "FunctionExtract_Keywords": {"text": "Any", "tokenizer": "Any", "make_lower": "bool", "returns": "list[str]"}, "FunctionTokenize_Text": {"cleaned_text": "str", "returns": "list[str]"}, "ClassOpenaichat": {"openai_model": "str", "cache_path": "Path"}, "ClassOpenaichatMethodAsk": {"message": "str", "returns": "str"}, "ClassOpenaichatMethod_Send_Request": {"messages": "list[dict[str, str]]", "returns": "Any"}}
'''

Example 4:
'''
User:
import asyncio
import base64
from typing import Literal

from openai import Client

from ..utils.validate_llm_kwargs import validate_llm_kwargs


# Function to encode the image to base64
def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to call GPT-4o API with the base64 image and a question
async def _acall_model(
    client: Client,
    base64_image: str,
    input_type: Literal["png", "jpg", "jpeg", "gif", "webp"],
    prompt: str,
    model_checkpoint: str = "gpt-4o-2024-08-06",
    **llm_kwargs,
):
    llm_kwargs = validate_llm_kwargs(llm_kwargs)
    response = client.chat.completions.create(
        model=model_checkpoint,  # Use GPT-4o model identifier
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{input_type};base64,{base64_image}",
                            "detail": "high",  # Adjust to "low" for a less detailed analysis
                        },
                    },
                ],
            }
        ],
        **llm_kwargs,
    )
    # Return the response from the model
    return response.choices[0].message.content.strip()


async def _process_page(
    client, base64_page, input_type, prompt, model_checkpoint, page_num
):
    result = f"\n\n=== Page {page_num} ===\n\n"
    result += await _acall_model(
        client, base64_page, input_type, prompt, model_checkpoint
    )
    return result, page_num


async def _process_all_pages(
    client, base64_pages, input_type, prompt, model_checkpoint
):
    tasks = [
        _process_page(client, page, input_type, prompt, model_checkpoint, i)
        for i, page in enumerate(base64_pages, 1)
    ]

    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1])
    return "".join(result[0] for result in sorted_results)
```

Json schema:
```
{'$defs': {'Function_Acall_Model': {'properties': {'client': {'base64_image': {'default': 'str', 'description': 'Annotation of the argument: `base64_image` with current annotation: `str`.'}, 'client': {'default': 'Client', 'description': 'Annotation of the argument: `client` with current annotation: `Client`.'}, 'input_type': {'default': "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", 'description': "Annotation of the argument: `input_type` with current annotation: `Literal['png', 'jpg', 'jpeg', 'gif', 'webp']`."}, 'model_checkpoint': {'default': 'str', 'description': "Annotation of the argument: `model_checkpoint` with default value: `'gpt-4o-2024-08-06'`, and current annotation: `str`."}, 'prompt': {'default': 'str', 'description': 'Annotation of the argument: `prompt` with current annotation: `str`.'}, 'title': 'Client', 'type': 'string'}, 'base64_image': {'base64_image': {'default': 'str', 'description': 'Annotation of the argument: `base64_image` with current annotation: `str`.'}, 'client': {'default': 'Client', 'description': 'Annotation of the argument: `client` with current annotation: `Client`.'}, 'input_type': {'default': "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", 'description': "Annotation of the argument: `input_type` with current annotation: `Literal['png', 'jpg', 'jpeg', 'gif', 'webp']`."}, 'model_checkpoint': {'default': 'str', 'description': "Annotation of the argument: `model_checkpoint` with default value: `'gpt-4o-2024-08-06'`, and current annotation: `str`."}, 'prompt': {'default': 'str', 'description': 'Annotation of the argument: `prompt` with current annotation: `str`.'}, 'title': 'Base64 Image', 'type': 'string'}, 'input_type': {'base64_image': {'default': 'str', 'description': 'Annotation of the argument: `base64_image` with current annotation: `str`.'}, 'client': {'default': 'Client', 'description': 'Annotation of the argument: `client` with current annotation: `Client`.'}, 'input_type': {'default': "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", 'description': "Annotation of the argument: `input_type` with current annotation: `Literal['png', 'jpg', 'jpeg', 'gif', 'webp']`."}, 'model_checkpoint': {'default': 'str', 'description': "Annotation of the argument: `model_checkpoint` with default value: `'gpt-4o-2024-08-06'`, and current annotation: `str`."}, 'prompt': {'default': 'str', 'description': 'Annotation of the argument: `prompt` with current annotation: `str`.'}, 'title': 'Input Type', 'type': 'string'}, 'prompt': {'base64_image': {'default': 'str', 'description': 'Annotation of the argument: `base64_image` with current annotation: `str`.'}, 'client': {'default': 'Client', 'description': 'Annotation of the argument: `client` with current annotation: `Client`.'}, 'input_type': {'default': "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", 'description': "Annotation of the argument: `input_type` with current annotation: `Literal['png', 'jpg', 'jpeg', 'gif', 'webp']`."}, 'model_checkpoint': {'default': 'str', 'description': "Annotation of the argument: `model_checkpoint` with default value: `'gpt-4o-2024-08-06'`, and current annotation: `str`."}, 'prompt': {'default': 'str', 'description': 'Annotation of the argument: `prompt` with current annotation: `str`.'}, 'title': 'Prompt', 'type': 'string'}, 'model_checkpoint': {'base64_image': {'default': 'str', 'description': 'Annotation of the argument: `base64_image` with current annotation: `str`.'}, 'client': {'default': 'Client', 'description': 'Annotation of the argument: `client` with current annotation: `Client`.'}, 'input_type': {'default': "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", 'description': "Annotation of the argument: `input_type` with current annotation: `Literal['png', 'jpg', 'jpeg', 'gif', 'webp']`."}, 'model_checkpoint': {'default': 'str', 'description': "Annotation of the argument: `model_checkpoint` with default value: `'gpt-4o-2024-08-06'`, and current annotation: `str`."}, 'prompt': {'default': 'str', 'description': 'Annotation of the argument: `prompt` with current annotation: `str`.'}, 'title': 'Model Checkpoint', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['client', 'base64_image', 'input_type', 'prompt', 'model_checkpoint', 'returns'], 'title': 'Function_Acall_Model', 'type': 'object'}, 'Function_Encode_Image_To_Base64': {'properties': {'image_path': {'image_path': {'default': 'str', 'description': 'Annotation of the argument: `image_path` with current annotation: `str`.'}, 'title': 'Image Path', 'type': 'string'}, 'returns': {'default': 'str', 'description': 'Return type annotation. This is the type hint that appears after the `->`. Current annotation: `str`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['image_path'], 'title': 'Function_Encode_Image_To_Base64', 'type': 'object'}, 'Function_Process_All_Pages': {'properties': {'client': {'base64_pages': {'description': 'Annotation of the argument: `base64_pages`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Client', 'type': 'string'}, 'base64_pages': {'base64_pages': {'description': 'Annotation of the argument: `base64_pages`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Base64 Pages', 'type': 'string'}, 'input_type': {'base64_pages': {'description': 'Annotation of the argument: `base64_pages`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Input Type', 'type': 'string'}, 'prompt': {'base64_pages': {'description': 'Annotation of the argument: `base64_pages`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Prompt', 'type': 'string'}, 'model_checkpoint': {'base64_pages': {'description': 'Annotation of the argument: `base64_pages`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Model Checkpoint', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['client', 'base64_pages', 'input_type', 'prompt', 'model_checkpoint', 'returns'], 'title': 'Function_Process_All_Pages', 'type': 'object'}, 'Function_Process_Page': {'properties': {'client': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Client', 'type': 'string'}, 'base64_page': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Base64 Page', 'type': 'string'}, 'input_type': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Input Type', 'type': 'string'}, 'prompt': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Prompt', 'type': 'string'}, 'model_checkpoint': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Model Checkpoint', 'type': 'string'}, 'page_num': {'base64_page': {'description': 'Annotation of the argument: `base64_page`.'}, 'client': {'description': 'Annotation of the argument: `client`.'}, 'input_type': {'description': 'Annotation of the argument: `input_type`.'}, 'model_checkpoint': {'description': 'Annotation of the argument: `model_checkpoint`.'}, 'page_num': {'description': 'Annotation of the argument: `page_num`.'}, 'prompt': {'description': 'Annotation of the argument: `prompt`.'}, 'title': 'Page Num', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['client', 'base64_page', 'input_type', 'prompt', 'model_checkpoint', 'page_num', 'returns'], 'title': 'Function_Process_Page', 'type': 'object'}}, 'properties': {'Function_Encode_Image_To_Base64': {'$ref': '#/$defs/Function_Encode_Image_To_Base64'}, 'Function_Acall_Model': {'$ref': '#/$defs/Function_Acall_Model'}, 'Function_Process_Page': {'$ref': '#/$defs/Function_Process_Page'}, 'Function_Process_All_Pages': {'$ref': '#/$defs/Function_Process_All_Pages'}}, 'required': ['Function_Encode_Image_To_Base64', 'Function_Acall_Model', 'Function_Process_Page', 'Function_Process_All_Pages'], 'title': 'ArgumentsModel', 'type': 'object'}
```

Assistant:
{"Function_Encode_Image_To_Base64": {"image_path": "str", "returns": "str"}, "Function_Acall_Model": {"client": "Client", "base64_image": "str", "input_type": "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", "prompt": "str", "model_checkpoint": "str", "returns": "str"}, "Function_Process_Page": {"client": "Client", "base64_page": "str", "input_type": "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", "prompt": "str", "model_checkpoint": "str", "page_num": "int", "returns": "tuple[str, int]"}, "Function_Process_All_Pages": {"client": "Client", "base64_pages": "list[str]", "input_type": "Literal['png', 'jpg', 'jpeg', 'gif', 'webp']", "prompt": "str", "model_checkpoint": "str", "returns": "str"}}
'''

Example 5:
'''
Input code:
```
import torch
from typing import Union
from diffusers.pipelines import FluxPipeline
from PIL import Image
import numpy as np
import cv2

from .common.base import BaseCondition

condition_dict = {
    "depth": 0,
    "canny": 1,
    "subject": 4,
    "coloring": 6,
    "deblurring": 7,
    "fill": 9,
}


class Condition(BaseCondition):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
        type_id: int = 0
    ):
        super().__init__()
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        self.type_id = type_id
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    @staticmethod
    def calculate_condition_score(image, weights = None):
        \"\"\"
        Calculates a weighted condition score based on image brightness, contrast, and edge density.
        Args:
            image: Input PIL Image
            weights: Optional list of weights for [brightness, contrast, edges]. Defaults to [0.4, 0.3, 0.3]
        Returns:
            float: Weighted condition score between 0 and 1
        \"\"\"
        weights = weights or [0.4, 0.3, 0.3]
        img_array = np.array(image.convert('L'))
        brightness = np.mean(img_array) / 255
        contrast = np.std(img_array) / 128
        edges = cv2.Canny(img_array, 100, 200).sum() / (img_array.size * 255)
        return np.clip(np.dot([brightness, contrast, edges], weights), 0, 1)

    def validate_and_resize(self, target_size: tuple):
        \"\"\"`target_size`: Desired dimensions (width, height) for the condition image.\"\"\"
        if self.condition.size != target_size:
            self.condition = self.condition.resize(target_size, Image.Resampling.LANCZOS)
        if self.condition.mode != 'RGB':
            self.condition = self.condition.convert('RGB')
    @property
    def type_id(self) -> int:
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str):
        \"\"\"Returns the type id of the condition.\"\"\"
        return condition_dict[condition_type]

    async def _encode_image(
        self, pipe, cond_img
    ):
        # Convert synchronous operations to async using asyncio.to_thread if needed
        cond_img = pipe.image_processor.preprocess(cond_img)
        cond_img = cond_img.to(pipe.device).to(pipe.dtype)

        # VAE encoding might be computationally intensive, but since it's using PyTorch,
        # it's already optimized for GPU operations
        cond_img = pipe.vae.encode(cond_img).latent_dist.sample()
        cond_img = (
                           cond_img - pipe.vae.config.shift_factor
                   ) * pipe.vae.config.scaling_factor

        cond_tokens = pipe._pack_latents(cond_img, *cond_img.shape)
        cond_ids = pipe._prepare_latent_image_ids(
            cond_img.shape[0],
            cond_img.shape[2],
            cond_img.shape[3],
            pipe.device,
            pipe.dtype,
        )

        return cond_tokens, cond_ids

    async def encode(self, pipe):
        if self.condition_type in [
            "depth",
            "canny",
            "subject",
            "coloring",
            "deblurring",
            "fill",
        ]:
            tokens, ids = await self._encode_image(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )

        type_id = ids[:, 0].item() * self.type_id
        return tokens, ids, type_id
```

Json schema:
```
{'$defs': {'ClassConditionMethodCalculate_Condition_Score': {'properties': {'image': {'image': {'description': 'Annotation of the argument: `image`.'}, 'title': 'Image', 'type': 'string', 'weights': {'description': 'Annotation of the argument: `weights` with default value: `None`.'}}, 'weights': {'image': {'description': 'Annotation of the argument: `image`.'}, 'title': 'Weights', 'type': 'string', 'weights': {'description': 'Annotation of the argument: `weights` with default value: `None`.'}}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['image', 'weights', 'returns'], 'title': 'ClassConditionMethodCalculate_Condition_Score', 'type': 'object'}, 'ClassConditionMethodEncode': {'properties': {'pipe': {'pipe': {'description': 'Annotation of the argument: `pipe`.'}, 'title': 'Pipe', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['pipe', 'returns'], 'title': 'ClassConditionMethodEncode', 'type': 'object'}, 'ClassConditionMethodGet_Type_Id': {'properties': {'condition_type': {'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'title': 'Condition Type', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['condition_type', 'returns'], 'title': 'ClassConditionMethodGet_Type_Id', 'type': 'object'}, 'ClassConditionMethodType_Id': {'properties': {'returns': {'default': 'int', 'description': 'Return type annotation. This is the type hint that appears after the `->`. Current annotation: `int`.', 'title': 'Returns', 'type': 'string'}}, 'title': 'ClassConditionMethodType_Id', 'type': 'object'}, 'ClassConditionMethodValidate_And_Resize': {'properties': {'target_size': {'target_size': {'default': 'tuple', 'description': 'Annotation of the argument: `target_size` with current annotation: `tuple`.'}, 'title': 'Target Size', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['target_size', 'returns'], 'title': 'ClassConditionMethodValidate_And_Resize', 'type': 'object'}, 'ClassConditionMethod_Encode_Image': {'properties': {'pipe': {'cond_img': {'description': 'Annotation of the argument: `cond_img`.'}, 'pipe': {'description': 'Annotation of the argument: `pipe`.'}, 'title': 'Pipe', 'type': 'string'}, 'cond_img': {'cond_img': {'description': 'Annotation of the argument: `cond_img`.'}, 'pipe': {'description': 'Annotation of the argument: `pipe`.'}, 'title': 'Cond Img', 'type': 'string'}, 'returns': {'description': 'Return type annotation. This is the type hint that appears after the `->`.', 'title': 'Returns', 'type': 'string'}}, 'required': ['pipe', 'cond_img', 'returns'], 'title': 'ClassConditionMethod_Encode_Image', 'type': 'object'}, 'ClassConditionMethod__Init__': {'properties': {'condition_type': {'condition': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `condition` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'mask': {'description': 'Annotation of the argument: `mask` with default value: `None`.'}, 'raw_img': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `raw_img` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'title': 'Condition Type', 'type': 'string', 'type_id': {'default': 'int', 'description': 'Annotation of the argument: `type_id` with default value: `0`, and current annotation: `int`.'}}, 'raw_img': {'condition': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `condition` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'mask': {'description': 'Annotation of the argument: `mask` with default value: `None`.'}, 'raw_img': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `raw_img` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'title': 'Raw Img', 'type': 'string', 'type_id': {'default': 'int', 'description': 'Annotation of the argument: `type_id` with default value: `0`, and current annotation: `int`.'}}, 'condition': {'condition': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `condition` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'mask': {'description': 'Annotation of the argument: `mask` with default value: `None`.'}, 'raw_img': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `raw_img` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'title': 'Condition', 'type': 'string', 'type_id': {'default': 'int', 'description': 'Annotation of the argument: `type_id` with default value: `0`, and current annotation: `int`.'}}, 'mask': {'condition': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `condition` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'mask': {'description': 'Annotation of the argument: `mask` with default value: `None`.'}, 'raw_img': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `raw_img` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'title': 'Mask', 'type': 'string', 'type_id': {'default': 'int', 'description': 'Annotation of the argument: `type_id` with default value: `0`, and current annotation: `int`.'}}, 'type_id': {'condition': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `condition` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'condition_type': {'default': 'str', 'description': 'Annotation of the argument: `condition_type` with current annotation: `str`.'}, 'mask': {'description': 'Annotation of the argument: `mask` with default value: `None`.'}, 'raw_img': {'default': 'Union[Image.Image, torch.Tensor]', 'description': 'Annotation of the argument: `raw_img` with default value: `None`, and current annotation: `Union[Image.Image, torch.Tensor]`.'}, 'title': 'Type Id', 'type': 'string', 'type_id': {'default': 'int', 'description': 'Annotation of the argument: `type_id` with default value: `0`, and current annotation: `int`.'}}}, 'required': ['condition_type', 'raw_img', 'condition', 'mask', 'type_id'], 'title': 'ClassConditionMethod__Init__', 'type': 'object'}}, 'properties': {'ClassConditionMethod__Init__': {'$ref': '#/$defs/ClassConditionMethod__Init__'}, 'ClassConditionMethodCalculate_Condition_Score': {'$ref': '#/$defs/ClassConditionMethodCalculate_Condition_Score'}, 'ClassConditionMethodValidate_And_Resize': {'$ref': '#/$defs/ClassConditionMethodValidate_And_Resize'}, 'ClassConditionMethodType_Id': {'$ref': '#/$defs/ClassConditionMethodType_Id'}, 'ClassConditionMethodGet_Type_Id': {'$ref': '#/$defs/ClassConditionMethodGet_Type_Id'}, 'ClassConditionMethod_Encode_Image': {'$ref': '#/$defs/ClassConditionMethod_Encode_Image'}, 'ClassConditionMethodEncode': {'$ref': '#/$defs/ClassConditionMethodEncode'}}, 'required': ['ClassConditionMethod__Init__', 'ClassConditionMethodCalculate_Condition_Score', 'ClassConditionMethodValidate_And_Resize', 'ClassConditionMethodType_Id', 'ClassConditionMethodGet_Type_Id', 'ClassConditionMethod_Encode_Image', 'ClassConditionMethodEncode'], 'title': 'ArgumentsModel', 'type': 'object'}
```

Assistant:
{"ClassConditionMethod__Init__": {"condition_type": "str", "raw_img": "Union[Image.Image, torch.Tensor]", "condition": "Union[Image.Image, torch.Tensor]", "mask": "Any", "type_id": "int"}, "ClassConditionMethodCalculate_Condition_Score": {"image": "Image.Image", "weights": "Optional[list[float]]", "returns": "float"}, "ClassConditionMethodValidate_And_Resize": {"target_size": "tuple[int, int]", "returns": "None"}, "ClassConditionMethodType_Id": {"returns": "int"}, "ClassConditionMethodGet_Type_Id": {"condition_type": "str", "returns": "int"}, "ClassConditionMethod_Encode_Image": {"pipe": "FluxPipeline", "cond_img": "Union[Image.Image, torch.Tensor]", "returns": "tuple[torch.Tensor, torch.Tensor]"}, "ClassConditionMethodEncode": {"pipe": "FluxPipeline", "returns": "tuple[torch.Tensor, torch.Tensor, int]"}}
'''
""".strip()

SYSTEM_PROMPT_DOCSTRING = """
Act as an experienced Python specialist.

Write the docstring for each function, class, and class method from the input code, following the provided json schema. \
Always include the `Args` and `Returns` sections for methods and functions. \
For classes, include the `Args` section if possible. \
Write the docstring only when you are confident that it is correct, valid, and necessary \
for better understanding of the code. \
If you are not sure how the docstring should look like, leave it empty. \
If the function / class / method is very simple to understand even without documentation or is dummy, \
leave it empty as well.

If you believe that the docstring for the function / method needs to be written, write it following this template:
```
<General description the purpose of the function / method>

Args:
  arg_1 (`<type of the arg_1>`, <default value if exists>):
    <description of the arg_1>
  ...
  arg_n (`<type of the arg_n>, <default value if exists>):
    <description of the arg_n>

Returns:
  `<type of the returns>`: <description of the returns>
```

For classes, if you believe that the docstring needs to be written, write it following the examples below.

Examples for few-shot learning on how to write docstrings for the input code:

Example 1:
'''
User:
Input code:
```
import warnings
import weakref
from collections import Counter
from random import sample
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import config
from .features.features import (
    _align_features,
    _check_if_features_can_be_aligned,
)
from .fingerprint import (
    update_fingerprint,
)
from .info import DatasetInfo
from .splits import NamedSplit
from .table import (
    InMemoryTable,
    concat_tables,
)
from .utils import logging
from .utils.py_utils import (
    Literal,
)
from .dataset import Dataset
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf

logger = logging.get_logger(__name__)

PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED = (
    "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.parquet"
)


class TensorflowDatasetMixin:
    name: str = ''
    cache_dir: Optional[str] = None
    _TF_DATASET_REFS = set()

    @staticmethod
    def _get_output_signature(
        dataset: "Dataset",
        collate_fn: Callable,
        collate_fn_args: dict,
        cols_to_retain: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        num_test_batches: int = 20,
    ):
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        if len(dataset) == 0:
            raise ValueError("Unable to get the output signature because the dataset is empty.")
        if batch_size is not None:
            batch_size = min(len(dataset), batch_size)
        test_batch_size = 1

        if cols_to_retain is not None:
            cols_to_retain = list(set(cols_to_retain + ["label_ids", "label", "labels"]))

        test_batches = []
        for _ in range(num_test_batches):
            indices = sample(range(len(dataset)), test_batch_size)
            test_batch = dataset[indices]
            if cols_to_retain is not None:
                test_batch = {key: value for key, value in test_batch.items() if key in cols_to_retain}
            test_batch = [{key: value[i] for key, value in test_batch.items()} for i in range(test_batch_size)]
            test_batch = collate_fn(test_batch, **collate_fn_args)
            test_batches.append(test_batch)

        tf_columns_to_signatures = {}
        np_columns_to_dtypes = {}
        for column in test_batches[0].keys():
            raw_arrays = [batch[column] for batch in test_batches]
            # In case the collate_fn returns something strange
            np_arrays = []
            for array in raw_arrays:
                if isinstance(array, np.ndarray):
                    np_arrays.append(array)
                elif isinstance(array, tf.Tensor):
                    np_arrays.append(array.numpy())
                else:
                    np_arrays.append(np.array(array))

            if np.issubdtype(np_arrays[0].dtype, np.integer) or np_arrays[0].dtype == bool:
                tf_dtype = tf.int64
                np_dtype = np.int64
            elif np.issubdtype(np_arrays[0].dtype, np.number):
                tf_dtype = tf.float32
                np_dtype = np.float32
            elif np_arrays[0].dtype.kind == "U":  # Unicode strings
                np_dtype = np.str_
                tf_dtype = tf.string
            else:
                raise RuntimeError(
                    f"Unrecognized array dtype {np_arrays[0].dtype}. \\n"
                    "Nested types and image/audio types are not supported yet."
                )
            shapes = [array.shape for array in np_arrays]
            static_shape = []
            for dim in range(len(shapes[0])):
                sizes = {shape[dim] for shape in shapes}
                if dim == 0:
                    static_shape.append(batch_size)
                    continue
                if len(sizes) == 1:  # This dimension looks constant
                    static_shape.append(sizes.pop())
                else:  # Use None for variable dimensions
                    static_shape.append(None)
            tf_columns_to_signatures[column] = tf.TensorSpec(shape=static_shape, dtype=tf_dtype)
            np_columns_to_dtypes[column] = np_dtype

        return tf_columns_to_signatures, np_columns_to_dtypes

    def to_tf_dataset(
        self,
        batch_size: Optional[int] = None,
        columns: Optional[Union[str, List[str]]] = None,
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None,
        drop_remainder: bool = False,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        label_cols: Optional[Union[str, List[str]]] = None,
        prefetch: bool = True,
        num_workers: int = 0,
        num_test_batches: int = 20,
    ):
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        if (isinstance(columns, list) and len(columns) == 1) or (
            isinstance(label_cols, list) and len(label_cols) == 1
        ):
            warnings.warn(
                "The output of `to_tf_dataset` will change when a passing single element list for `labels` or "
                "`columns` in the next datasets version. To return a tuple structure rather than dict, pass a "
                "single string.\\n"
                "Old behaviour: columns=['a'], labels=['labels'] -> (tf.Tensor, tf.Tensor)  \\n"
                "             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor)  \\n"
                "New behaviour: columns=['a'],labels=['labels'] -> ({'a': tf.Tensor}, {'labels': tf.Tensor})  \\n"
                "             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor) ",
                FutureWarning,
            )

        if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
            logger.warning(
                "Note that to_tf_dataset() loads the data with a generator rather than a full tf.data "
                "pipeline and is not compatible with remote TPU connections. If you encounter errors, please "
                "try using a TPU VM or, if your data can fit in memory, loading it into memory as a dict of "
                "Tensors instead of streaming with to_tf_dataset()."
            )

        if collate_fn is None:
            # Set a very simple default collator that just stacks things together
            collate_fn = minimal_tf_collate_fn
        if collate_fn_args is None:
            collate_fn_args = {}
        if label_cols and not columns:
            raise ValueError("Cannot specify label_cols without specifying columns!")
        if label_cols is None:
            label_cols = []
        elif isinstance(label_cols, str):
            label_cols = [label_cols]
        if len(set(label_cols)) < len(label_cols):
            raise ValueError("List of label_cols contains duplicates.")
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            if len(set(columns)) < len(columns):
                raise ValueError("List of columns contains duplicates.")
            cols_to_retain = list(set(columns + label_cols))
        else:
            cols_to_retain = None  # Indicates keeping all valid columns
            columns = []

        if self.format["type"] not in ["custom", "numpy"]:
            dataset = self.with_format("numpy")
        else:
            dataset = self

        # TODO(Matt, QL): deprecate the retention of label_ids and label

        output_signature, columns_to_np_types = dataset._get_output_signature(
            dataset,
            collate_fn=collate_fn,
            collate_fn_args=collate_fn_args,
            cols_to_retain=cols_to_retain,
            batch_size=batch_size if drop_remainder else None,
            num_test_batches=num_test_batches,
        )

        if "labels" in output_signature:
            if ("label_ids" in columns or "label" in columns) and "labels" not in columns:
                columns = [col for col in columns if col not in ["label_ids", "label"]] + ["labels"]
            if ("label_ids" in label_cols or "label" in label_cols) and "labels" not in label_cols:
                label_cols = [col for col in label_cols if col not in ["label_ids", "label"]] + ["labels"]

        for col in columns:
            if col not in output_signature:
                raise ValueError(f"Column {col} not found in dataset!")

        for col in label_cols:
            if col not in output_signature:
                raise ValueError(f"Label column {col} not found in dataset!")

        if num_workers == 0:
            tf_dataset = dataset_to_tf(
                dataset=dataset,
                cols_to_retain=cols_to_retain,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                columns_to_np_types=columns_to_np_types,
                output_signature=output_signature,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        elif num_workers > 0:
            if batch_size is None:
                raise NotImplementedError(
                    "`batch_size` must be specified when using multiple workers, as unbatched multiprocessing "
                    "is not supported yet. Please provide a `batch_size` if `num_workers` is greater than 0."
                )
            tf_dataset = multiprocess_dataset_to_tf(
                dataset=dataset,
                cols_to_retain=cols_to_retain,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                columns_to_np_types=columns_to_np_types,
                output_signature=output_signature,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                num_workers=num_workers,
            )
        else:
            raise ValueError("num_workers must be >= 0")

        def split_features_and_labels(input_batch):
            # TODO(Matt, QL): deprecate returning the dict content when there's only one key
            features = {key: tensor for key, tensor in input_batch.items() if key in columns}
            labels = {key: tensor for key, tensor in input_batch.items() if key in label_cols}
            if len(features) == 1:
                features = list(features.values())[0]
            if len(labels) == 1:
                labels = list(labels.values())[0]
            if isinstance(labels, dict) and len(labels) == 0:
                return features
            else:
                return features, labels

        if cols_to_retain is not None:
            tf_dataset = tf_dataset.map(split_features_and_labels)

        if prefetch:
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Remove a reference to the open Arrow file on delete
        def cleanup_callback(ref):
            dataset.__del__()
            self._TF_DATASET_REFS.remove(ref)

        self._TF_DATASET_REFS.add(weakref.ref(tf_dataset, cleanup_callback))

        return tf_dataset

    def save_to_cache(self) -> str:
        if not self.name:
            raise ValueError("Dataset name must be set to save")
            
        save_path = os.path.join(self.cache_dir, self.name) if self.cache_dir else self.name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if config.TF_AVAILABLE:
            import tensorflow as tf
            tf.data.Dataset.save(self, save_path)
        else:
            np.save(save_path, self.with_format("numpy"))
            
        return save_path


class DatasetTransformationNotAllowedError(Exception):
    pass


def _check_column_names(column_names: List[str]):
    \"\"\"Check the column names to make sure they don't contain duplicates.\"\"\"
    counter = Counter(column_names)
    if not all(count == 1 for count in counter.values()):
        duplicated_columns = [col for col in counter if counter[col] > 1]
        raise ValueError(f"The table can't have duplicated columns but columns {duplicated_columns} are duplicated.")

def _concatenate_map_style_datasets(
    dsets: List[Dataset],
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    axis: int = 0,
):
    # Ignore datasets with no rows
    if any(dset.num_rows > 0 for dset in dsets):
        dsets = [dset for dset in dsets if dset.num_rows > 0]
    else:
        # Return first dataset if all datasets are empty
        return dsets[0]

    # Perform checks (and a potentional cast if axis=0)
    if axis == 0:
        _check_if_features_can_be_aligned([dset.features for dset in dsets])
    else:
        if not all(dset.num_rows == dsets[0].num_rows for dset in dsets):
            raise ValueError("Number of rows must match for all datasets")
        _check_column_names([col_name for dset in dsets for col_name in dset._data.column_names])

    # Find common format or reset format
    format = dsets[0].format
    if any(dset.format != format for dset in dsets):
        format = {}
        logger.info("Some of the datasets have disparate format. Resetting the format of the concatenated dataset.")

    def apply_offset_to_indices_table(table, offset):
        if offset == 0:
            return table
        else:
            array = table["indices"]
            new_array = pc.add(array, pa.scalar(offset, type=pa.uint64()))
            return InMemoryTable.from_arrays([new_array], names=["indices"])

    # Concatenate indices if they exist
    if any(dset._indices is not None for dset in dsets):
        if axis == 0:
            # Datasets with no indices tables are replaced with a dataset with an indices table in memory.
            # Applying an offset to an indices table also brings the table in memory.
            indices_tables = []
            for i in range(len(dsets)):
                if dsets[i]._indices is None:
                    dsets[i] = dsets[i]._select_with_indices_mapping(range(len(dsets[i])))
                indices_tables.append(dsets[i]._indices)

            # An offset needs to be applied to the indices before concatenating
            offset = 0
            for i in range(len(dsets)):
                indices_tables[i] = apply_offset_to_indices_table(indices_tables[i], offset)
                offset += len(dsets[i]._data)

            # Concatenate indices
            indices_tables = [t for t in indices_tables if len(t) > 0]
            if indices_tables:
                indices_table = concat_tables(indices_tables)
            else:
                indices_table = InMemoryTable.from_batches([], schema=pa.schema({"indices": pa.int64()}))
        else:
            if len(dsets) == 1:
                indices_table = dsets[0]._indices
            else:
                for i in range(len(dsets)):
                    dsets[i] = dsets[i].flatten_indices()
                indices_table = None
    else:
        indices_table = None

    table = concat_tables([dset._data for dset in dsets], axis=axis)
    if axis == 0:
        features_list = _align_features([dset.features for dset in dsets])
    else:
        features_list = [dset.features for dset in dsets]
    table = update_metadata_with_features(table, {k: v for features in features_list for k, v in features.items()})

    # Concatenate infos
    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint(
        "".join(dset._fingerprint for dset in dsets), _concatenate_map_style_datasets, {"info": info, "split": split}
    )

    # Make final concatenated dataset
    concatenated_dataset = Dataset(
        table,
        info=info,
        split=split,
        indices_table=indices_table,
        fingerprint=fingerprint,
    )
    concatenated_dataset.set_format(**format)
    return concatenated_dataset


def _interleave_map_style_datasets(
    datasets: List["Dataset"],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    **kwargs,
) -> "Dataset":
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(
            f"{stopping_strategy} stopping strategy in `interleave_datasets` is not implemented yet with a list of {type(datasets[0])}"
        )

    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = _concatenate_map_style_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])

    # if stopping_strategy is "first_exhausted", it is an undersampling situation whereas it is an oversampling situation if it is "all_exhausted"
    oversampling = stopping_strategy == "all_exhausted"

    if probabilities is None and not oversampling:
        # Undersampling situation with cycling between each sources
        # Example:: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples

        # Reasoning behind the following operation: keeping the min_length first indices of each dataset
        # while offsetting in order to correspond to the right indices of the concatenated dataset
        # and flattening to effectively interleave the datasets
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    elif probabilities is None:
        # Oversampling situation with cycling between each sources
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 5, 9, 0, 6, 10, 1, 3, 11]
        # Note that we have 5 examples per dataset with a rolling window since the longest dataset has 5 samples

        # Reasoning behind the following operation: for each dataset indices (i.e column) repeat the indices to have max_length indices per dataset
        # For example, if the max_length is 5 and the i-th dataset has 3 samples, the i-th column will be [0,1,2,0,1]
        indices = np.mod(np.arange(max(lengths)).reshape(-1, 1), np.array(lengths).reshape(1, -1))

        # We have to keep the indices to their respective dataset offsets and to flatten to effectively interleave the datasets
        indices = (indices + offsets).flatten().tolist()

    else:
        # boolean array indicating if at index i if the dataset_i has been fully exhausted
        is_exhausted = np.full(len(lengths), False)

        # if undersampling ("first_exhausted"), we stop as soon as one dataset is exhausted
        # if oversampling ("all_exhausted"), we stop as soons as every dataset is exhausted, i.e as soon as every samples of every dataset has been visited at least once
        bool_strategy_func = np.all if oversampling else np.any

        def iter_random_indices():
            \"\"\"Get an infinite iterator that randomly samples the index of the source to pick examples from.\"\"\"
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))

        current_index = [0] * len(datasets)
        indices = []
        for source_idx in iter_random_indices():
            # If no oversampling, we stop as soon as a dataset has ran out of examples (np.any)
            # Otherwise, we stop as soon as every dataset has ran out of examples (np.all)
            if bool_strategy_func(is_exhausted):
                # the stopping condition was reached, let's stop
                break

            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1

            # we've ran out of examples for the current dataset, let's update our boolean array and bring the current_index back to 0
            if current_index[source_idx] >= lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0

    return concatenated_datasets.select(indices, **kwargs)


def _split_by_node_map_style_dataset(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    return dataset.shard(num_shards=world_size, index=rank, contiguous=True)
```

Json schema:
```
{'properties': {'class_TensorflowDatasetMixin': {'description': 'Docstring for the class `TensorflowDatasetMixin`', 'title': 'Class Tensorflowdatasetmixin', 'type': 'string'}, 'class_TensorflowDatasetMixin_method__get_output_signature': {'description': 'Docstring for the method `_get_output_signature` of the class `TensorflowDatasetMixin`', 'title': 'Class Tensorflowdatasetmixin Method  Get Output Signature', 'type': 'string'}, 'class_TensorflowDatasetMixin_method_to_tf_dataset': {'description': 'Docstring for the method `to_tf_dataset` of the class `TensorflowDatasetMixin`', 'title': 'Class Tensorflowdatasetmixin Method To Tf Dataset', 'type': 'string'}, 'class_TensorflowDatasetMixin_method_save_to_cache': {'description': 'Docstring for the method `save_to_cache` of the class `TensorflowDatasetMixin`', 'title': 'Class Tensorflowdatasetmixin Method Save To Cache', 'type': 'string'}, 'class_DatasetTransformationNotAllowedError': {'description': 'Docstring for the class `DatasetTransformationNotAllowedError`', 'title': 'Class Datasettransformationnotallowederror', 'type': 'string'}, 'function__check_column_names': {'description': 'Docstring for the function `_check_column_names`', 'title': 'Function  Check Column Names', 'type': 'string'}, 'function__concatenate_map_style_datasets': {'description': 'Docstring for the function `_concatenate_map_style_datasets`', 'title': 'Function  Concatenate Map Style Datasets', 'type': 'string'}, 'function__interleave_map_style_datasets': {'description': 'Docstring for the function `_interleave_map_style_datasets`', 'title': 'Function  Interleave Map Style Datasets', 'type': 'string'}, 'function__split_by_node_map_style_dataset': {'description': 'Docstring for the function `_split_by_node_map_style_dataset`', 'title': 'Function  Split By Node Map Style Dataset', 'type': 'string'}}, 'required': ['class_TensorflowDatasetMixin', 'class_TensorflowDatasetMixin_method__get_output_signature', 'class_TensorflowDatasetMixin_method_to_tf_dataset', 'class_TensorflowDatasetMixin_method_save_to_cache', 'class_DatasetTransformationNotAllowedError', 'function__check_column_names', 'function__concatenate_map_style_datasets', 'function__interleave_map_style_datasets', 'function__split_by_node_map_style_dataset'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_TensorflowDatasetMixin": "A mixin class that provides functionality to convert a `Dataset` object to a TensorFlow dataset.\\\\n\\\\nArgs:\\\\n\\\\tname (`str`):\\\\n\\\\t\\\\tName identifier for the dataset. Used for cache file naming.\\\\n\\\\tcache_dir (`str`, *optional*):\\\\n\\\\t\\\\tDirectory to cache the dataset. If None, uses current directory.\\\\n\\\\t_TF_DATASET_REFS (`set`):\\\\n\\\\t\\\\tSet to store weak references to TensorFlow datasets. Used for cleanup of Arrow file references.\\\\n\\\\nMethods:\\\\n\\\\t`save_to_cache()`: Saves dataset to disk using `cache_dir`/`name` or `name` as path.\\\\n\\\\t`to_tf_dataset()`: Converts dataset to TensorFlow format with batching and preprocessing options.\\\\n\\\\nPrivate Methods:\\\\n\\\\t`_get_output_signature()`: Determines TensorFlow output signatures for dataset elements.", "class_TensorflowDatasetMixin_method__get_output_signature": "Private method used by `to_tf_dataset()` to find the shapes and dtypes of samples from this dataset\\\\n   after being passed through the collate_fn. Tensorflow needs an exact signature for tf.numpy_function, so\\\\n   the only way to do this is to run test batches - the collator may add or rename columns, so we can\'t figure\\\\n   it out just by inspecting the dataset.\\\\n\\\\nArgs:\\\\n\\\\tdataset (`Dataset`):\\\\n\\\\t\\\\tDataset to load samples from.\\\\n\\\\tcollate_fn(`bool`):\\\\n\\\\t\\\\tShuffle the dataset order when loading. Recommended True for training, False for validation/evaluation.\\\\n\\\\tcollate_fn(`Callable`):\\\\n\\\\t\\\\tA function or callable object (such as a `DataCollator`) that will collate\\\\n\\\\t\\\\tlists of samples into a batch.\\\\n\\\\tcollate_fn_args (`Dict`):\\\\n\\\\t\\\\tA `dict` of keyword arguments to be passed to the\\\\n\\\\t\\\\t`collate_fn`.\\\\n\\\\tbatch_size (`int`, optional):\\\\n\\\\t\\\\tThe size of batches loaded from the dataset. Used for shape inference.\\\\n\\\\t\\\\tCan be None, which indicates that batch sizes can be variable.\\\\n\\\\tnum_test_batches (`int`):\\\\n\\\\t\\\\tThe number of batches to load from the dataset for shape inference.\\\\n\\\\nReturns:\\\\n\\\\t`dict`: Dict mapping column names to tf.Tensorspec objects\\\\n\\\\t`dict`: Dict mapping column names to np.dtype objects", "class_TensorflowDatasetMixin_method_to_tf_dataset": "Create a `tf.data.Dataset` from the underlying Dataset. This `tf.data.Dataset` will load and collate batches from\\\\nthe Dataset, and is suitable for passing to methods like `model.fit()` or `model.predict()`. The dataset will yield\\\\n`dicts` for both inputs and labels unless the `dict` would contain only a single key, in which case a raw\\\\n`tf.Tensor` is yielded instead.\\\\n\\\\nArgs:\\\\n\\\\tbatch_size (`int`, *optional*):\\\\n\\\\t\\\\tSize of batches to load from the dataset. Defaults to `None`, which implies that the dataset won\'t be\\\\n\\\\t\\\\tbatched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.\\\\n\\\\tcolumns (`List[str]` or `str`, *optional*):\\\\n\\\\t\\\\tDataset column(s) to load in the `tf.data.Dataset`.\\\\n\\\\t\\\\tColumn names that are created by the `collate_fn` and that do not exist in the original dataset can be used.\\\\n\\\\tshuffle(`bool`, defaults to `False`):\\\\n\\\\t\\\\tShuffle the dataset order when loading. Recommended `True` for training, `False` for\\\\n\\\\t\\\\tvalidation/evaluation.\\\\n\\\\tdrop_remainder(`bool`, defaults to `False`):\\\\n\\\\t\\\\tDrop the last incomplete batch when loading. Ensures\\\\n\\\\t\\\\tthat all batches yielded by the dataset will have the same length on the batch dimension.\\\\n\\\\tcollate_fn(`Callable`, *optional*):\\\\n\\\\t\\\\tA function or callable object (such as a `DataCollator`) that will collate\\\\n\\\\t\\\\tlists of samples into a batch.\\\\n\\\\tcollate_fn_args (`Dict`, *optional*):\\\\n\\\\t\\\\tAn optional `dict` of keyword arguments to be passed to the\\\\n\\\\t\\\\t`collate_fn`.\\\\n\\\\tlabel_cols (`List[str]` or `str`, defaults to `None`):\\\\n\\\\t\\\\tDataset column(s) to load as labels.\\\\n\\\\t\\\\tNote that many models compute loss internally rather than letting Keras do it, in which case\\\\n\\\\t\\\\tpassing the labels here is optional, as long as they\'re in the input `columns`.\\\\n\\\\tprefetch (`bool`, defaults to `True`):\\\\n\\\\t\\\\tWhether to run the dataloader in a separate thread and maintain\\\\n\\\\t\\\\ta small buffer of batches for training. Improves performance by allowing data to be loaded in the\\\\n\\\\t\\\\tbackground while the model is training.\\\\n\\\\tnum_workers (`int`, defaults to `0`):\\\\n\\\\t\\\\tNumber of workers to use for loading the dataset.\\\\n\\\\tnum_test_batches (`int`, defaults to `20`):\\\\n\\\\t\\\\tNumber of batches to use to infer the output signature of the dataset.\\\\n\\\\t\\\\tThe higher this number, the more accurate the signature will be, but the longer it will take to\\\\n\\\\t\\\\tcreate the dataset.\\\\n\\\\nReturns:\\\\n\\\\t`tf.data.Dataset`: Dataset class from the TensorFlow library.", "class_TensorflowDatasetMixin_method_save_to_cache": "Saves the current dataset to the cache directory.\\\\n\\\\nReturns:\\\\n\\\\tstr: Path where dataset was saved.\\\\n\\\\nRaises:\\\\n\\\\tValueError: If neither cache_dir nor name is set.", "class_DatasetTransformationNotAllowedError": "", "function__check_column_names": "Check the column names to make sure they don\'t contain duplicates.\\\\n\\\\nArgs:\\\\n\\\\tcolumn_names (`list[str]`):\\\\n\\\\t\\\\tList of column names.\\\\n\\\\nReturns:\\\\n\\\\tNone", "function__concatenate_map_style_datasets": "Converts a list of :class:`Dataset` with the same schema into a single :class:`Dataset`.\\\\nWhen you concatenate on axis 0, missing data are filled with None values.\\\\n\\\\nArgs:\\\\n\\\\tdsets (`List[datasets.Dataset]`):\\\\n\\\\t\\\\tList of Datasets to concatenate.\\\\n\\\\tinfo (:class:`DatasetInfo`, optional):\\\\n\\\\t\\\\tDataset information, like description, citation, etc.\\\\n\\\\tsplit (:class:`NamedSplit`, optional):\\\\n\\\\t\\\\tName of the dataset split.\\\\n\\\\taxis (``{0, 1}``, default ``0``, meaning over rows):\\\\n\\\\t\\\\tAxis to concatenate over, where ``0`` means over rows (vertically) and ``1`` means over columns\\\\n\\\\t\\\\t(horizontally).\\\\n\\\\nReturns:\\\\n\\\\t `Dataset`: Dataset, consisting of the input list of datasets.", "function__interleave_map_style_datasets": "Interleave several map-style datasets (sources) into a single map-style dataset.\\\\nThe new dataset is constructed by alternating between the sources to get the examples.\\\\nIf `probabilities = None` (default) the new dataset is constructed by cycling between each source to get the examples.\\\\nIf `probabilities` is not `None, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.\\\\n\\\\nArgs:\\\\n\\\\tdatasets (`List[Dataset]`):\\\\n\\\\t\\\\tlist of datasets to interleave\\\\n\\\\tprobabilities (`List[float]`, optional, default None):\\\\n\\\\t\\\\tIf specified, the new dataset is constructed by sampling\\\\n\\\\t\\\\texamples from one source at a time according to these probabilities.\\\\n\\\\tseed (`int`, optional, default None):\\\\n\\\\t\\\\tThe random seed used to choose a source for each example.\\\\n\\\\tinfo (:class:`DatasetInfo`, optional):\\\\n\\\\t\\\\tDataset information, like description, citation, etc.\\\\n\\\\tsplit (:class:`NamedSplit`, optional):\\\\n\\\\t\\\\tName of the dataset split.\\\\n\\\\tstopping_strategy (`str`, defaults to `first_exhausted`):\\\\n\\\\t\\\\tTwo strategies are proposed right now.\\\\n\\\\t\\\\tBy default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.\\\\n\\\\t\\\\tIf the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.\\\\n\\\\t\\\\tNote that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:\\\\n\\\\t\\\\t- with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.\\\\n\\\\t\\\\t- with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.\\\\n\\\\t**kwargs (additional keyword arguments):\\\\n\\\\t\\\\tKeyword arguments to be passed to :meth:`datasets.Datasets.select` when selecting the indices used to interleave the datasets.\\\\n\\\\nReturns:\\\\n\\\\t`Dataset`: a single map-style dataset, constructed by alternating between the sources to get the examples.", "function__split_by_node_map_style_dataset": "Split a dataset for the node at rank `rank` in a pool of nodes of size `world_size`.\\\\nEach node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset.\\\\nTo maximize data loading throughput, chunks are made of contiguous data on disk if possible.\\\\n\\\\nArgs:\\\\n\\\\tdataset (`Dataset`):\\\\n\\\\t\\\\tThe dataset to split by node.\\\\n\\\\trank (`int`):\\\\n\\\\t\\\\tRank of the current node.\\\\n\\\\tworld_size (`int`):\\\\n\\\\t\\\\tTotal number of nodes.\\\\n\\\\nReturns:\\\\n\\\\t`Dataset`: The dataset to be used on the node at rank `rank`."}
\"\"\"
'''

Example 2:
'''
User:
Input code:
```
from typing import, Any, Dict, Optional, Tuple

import torch

from ..utils import is_sklearn_available


if is_sklearn_available():
    from sklearn.metrics import roc_curve

from ..cache_utils import DynamicCache
from ..pytorch_utils import isin_mps_friendly

class BaseGenerator:
    def generate(self) -> Any:
        raise NotImplementedError("All generators must implement generate method")

class CandidateGenerator(BaseGenerator):
    \"\"\"Add docstring\"\"\"
    
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )

class PromptLookupCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        eos_token_id: torch.Tensor = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        max_length: int = 20,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        input_length = input_ids.size(1)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True

                    # remove remaining candidate ids if an "eos" token is found, otherwise the target model may
                    # accept eos and the rest as valid, thus not stopping generation after "eos"
                    # NOTE: below code is written based on the fact that assisted decoding supports only bs=1
                    mask = isin_mps_friendly(chosen_ids, self.eos_token_id)
                    match_indices_eos = torch.nonzero(mask)
                    if match_indices_eos.numel() > 0:
                        first_eos_index = match_indices_eos[0].item()
                        chosen_ids = chosen_ids[:first_eos_index]
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        # Currently does nothing, but will be extended in the future
        return

class EarlyExitCandidateGenerator(AssistedCandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        super().__init__(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
        )
        # We have to move early exit out of the generation config, otherwise the assistant will also call `generate`
        # with early exit
        self.assistant_early_exit = self.generation_config.assistant_early_exit
        self.generation_config.assistant_early_exit = None

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        # Temporarily sets the number of hidden layers to the early exit value
        base_model = getattr(self.assistant_model, self.assistant_model.base_model_prefix)
        original_num_hidden_layers = base_model.config.num_hidden_layers
        base_model.config.num_hidden_layers = self.assistant_early_exit
        candidate_ids, candidate_logits = super().get_candidates(input_ids)
        base_model.config.num_hidden_layers = original_num_hidden_layers
        return candidate_ids, candidate_logits
```

Json schema:
```
{'properties': {'class_BaseGenerator': {'description': 'Docstring for the class `BaseGenerator`', 'title': 'Class Basegenerator', 'type': 'string'}, 'class_BaseGenerator_method_generate': {'description': 'Docstring for the method `generate` of the class `BaseGenerator`', 'title': 'Class Basegenerator Method Generate', 'type': 'string'}, 'class_CandidateGenerator': {'description': 'Docstring for the class `CandidateGenerator`', 'title': 'Class Candidategenerator', 'type': 'string'}, 'class_CandidateGenerator_method_get_candidates': {'description': 'Docstring for the method `get_candidates` of the class `CandidateGenerator`', 'title': 'Class Candidategenerator Method Get Candidates', 'type': 'string'}, 'class_CandidateGenerator_method_update_candidate_strategy': {'description': 'Docstring for the method `update_candidate_strategy` of the class `CandidateGenerator`', 'title': 'Class Candidategenerator Method Update Candidate Strategy', 'type': 'string'}, 'class_PromptLookupCandidateGenerator': {'description': 'Docstring for the class `PromptLookupCandidateGenerator`', 'title': 'Class Promptlookupcandidategenerator', 'type': 'string'}, 'class_PromptLookupCandidateGenerator_method___init__': {'description': 'Docstring for the method `__init__` of the class `PromptLookupCandidateGenerator`', 'title': 'Class Promptlookupcandidategenerator Method   Init  ', 'type': 'string'}, 'class_PromptLookupCandidateGenerator_method_get_candidates': {'description': 'Docstring for the method `get_candidates` of the class `PromptLookupCandidateGenerator`', 'title': 'Class Promptlookupcandidategenerator Method Get Candidates', 'type': 'string'}, 'class_PromptLookupCandidateGenerator_method_update_candidate_strategy': {'description': 'Docstring for the method `update_candidate_strategy` of the class `PromptLookupCandidateGenerator`', 'title': 'Class Promptlookupcandidategenerator Method Update Candidate Strategy', 'type': 'string'}, 'class_EarlyExitCandidateGenerator': {'description': 'Docstring for the class `EarlyExitCandidateGenerator`', 'title': 'Class Earlyexitcandidategenerator', 'type': 'string'}, 'class_EarlyExitCandidateGenerator_method___init__': {'description': 'Docstring for the method `__init__` of the class `EarlyExitCandidateGenerator`', 'title': 'Class Earlyexitcandidategenerator Method   Init  ', 'type': 'string'}, 'class_EarlyExitCandidateGenerator_method_get_candidates': {'description': 'Docstring for the method `get_candidates` of the class `EarlyExitCandidateGenerator`', 'title': 'Class Earlyexitcandidategenerator Method Get Candidates', 'type': 'string'}}, 'required': ['class_BaseGenerator', 'class_BaseGenerator_method_generate', 'class_CandidateGenerator', 'class_CandidateGenerator_method_get_candidates', 'class_CandidateGenerator_method_update_candidate_strategy', 'class_PromptLookupCandidateGenerator', 'class_PromptLookupCandidateGenerator_method___init__', 'class_PromptLookupCandidateGenerator_method_get_candidates', 'class_PromptLookupCandidateGenerator_method_update_candidate_strategy', 'class_EarlyExitCandidateGenerator', 'class_EarlyExitCandidateGenerator_method___init__', 'class_EarlyExitCandidateGenerator_method_get_candidates'], 'title': 'DocstringModel', 'type': 'object'}```

Assistant:
\"\"\"
{"class_BaseGenerator": "", "class_BaseGenerator_method_generate": "", "class_CandidateGenerator": "Abstract base class for all candidate generators that can be applied during assisted generation.", "class_CandidateGenerator_method_get_candidates": "Fetches the candidates to be tried for the current input.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\nReturn:\\\\n\\\\t`torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be\\\\n\\\\tassessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,\\\\n\\\\tvocabulary_size)` containing the logits associated to each candidate.", "class_CandidateGenerator_method_update_candidate_strategy": "Updates the candidate generation strategy based on the outcomes.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\tscores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):\\\\n\\\\t\\\\tPrediction scores of a language modeling head. These can be logits for each vocabulary when not using\\\\n\\\\t\\\\tbeam search or log softmax for each vocabulary token when using beam search\\\\n\\\\tnum_matches (`int`):\\\\n\\\\t\\\\tThe number of matches between the candidate sequences and the model predictions.", "class_PromptLookupCandidateGenerator": "`CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up\\\\nlikely continuations in the provided prompt (input_ids) itself.\\\\nArgs:\\\\n\\\\tmax_matching_ngram_size (`int`):\\\\n\\\\t\\\\tThe maximum ngram size to be considered for matching in the prompt\\\\n\\\\tnum_output_tokens (`int`):\\\\n\\\\t\\\\tThe number of tokens to be output as candidate tokens.\\\\n\\\\tmax_length (`int`):\\\\n\\\\t\\\\tThe number of total maximum tokens that can be generated. For decoder-only models that includes the prompt length.\\\\n\\\\t\\\\tDefaults to 20, which is the max length used as default in generation config.", "class_PromptLookupCandidateGenerator_method___init__": "", "class_PromptLookupCandidateGenerator_method_get_candidates": "Fetches the candidates to be tried for the current input.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\nReturn:\\\\n\\\\t`torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.", "class_PromptLookupCandidateGenerator_method_update_candidate_strategy": "Updates the candidate generation strategy based on the outcomes.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\tscores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):\\\\n\\\\t\\\\tPrediction scores of a language modeling head. These can be logits for each vocabulary when not using\\\\n\\\\t\\\\tbeam search or log softmax for each vocabulary token when using beam search\\\\n\\\\tnum_matches (`int`):\\\\n\\\\t\\\\tThe number of matches between the candidate sequences and the model predictions.", "class_EarlyExitCandidateGenerator": "`CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates\\\\ncandidates through the use of **the model itself**, exiting early.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\tassistant_model (`PreTrainedModel`):\\\\n\\\\t\\\\tThe original model. This model must support early exit (i.e. is trained to compute logits in earlier\\\\n\\\\t\\\\tlayers).\\\\n\\\\tgeneration_config (`~generation.GenerationConfig`, *optional*):\\\\n\\\\t\\\\tThe generation configuration to be used as base parametrization for the generation call.\\\\n\\\\tlogits_processor (`LogitsProcessorList`):\\\\n\\\\t\\\\tAn instance of `LogitsProcessorList`. List of instances of class derived from `LogitsProcessor`\\\\n\\\\t\\\\tused to modify the prediction scores of the language modeling head applied at each generation step.\\\\n\\\\tmodel_kwargs (`Dict`):\\\\n\\\\t\\\\tThe keyword arguments that will be passed to the main model, and are used as base inputs for the assistant\\\\n\\\\t\\\\tmodel as well.\\\\n\\\\tinputs_tensor (`torch.Tensor`, *optional*):\\\\n\\\\t\\\\tThe model input tensor. In encoder-decoder models, this is the encoder input.", "class_EarlyExitCandidateGenerator_method___init__": "", "class_EarlyExitCandidateGenerator_method_get_candidates": "Fetches the candidates to be tried for the current input.\\\\n\\\\nArgs:\\\\n\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\nReturn:\\\\n\\\\t`torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be\\\\n\\\\tassessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,\\\\n\\\\tvocabulary_size)` containing the logits associated to each candidate."}
\"\"\"
'''

Example 3:
'''
User:
Input code:
```
from typing import Union, Any, Optional, NamedTuple

class BestRun(NamedTuple):
    \"\"\"best run of hyperparameter search.
    \"\"\"

    run_id: str
    objective: Union[float, list[float]]
    hyperparameters: dict[str, Any]
    run_summary: Optional[Any] = None
```

Json schema:
```
{'properties': {'class_BestRun': {'description': 'Docstring for the class `BestRun`', 'title': 'Class Bestrun', 'type': 'string'}}, 'required': ['class_BestRun'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_BestRun": "The best run found by a hyperparameter search.\\\\n\\\\nArgs:\\\\n\\\\trun_id (`str`):\\\\n\\\\t\\\\tThe id of the best run.\\\\n\\\\tobjective (`Union[float, list[float]`):\\\\n\\\\t\\\\tThe objective value or list of objectives values obtained for this run.\\\\n\\\\thyperparameters (`dict[str, Any]`):\\\\n\\\\t\\\\tThe hyperparameters used to get this run.\\\\n\\\\trun_summary (`Optional[Any]`):\\\\n\\\\t\\\\tThe summary of tuning experiments."}
\"\"\"
'''

Example 4:
'''
User:
Input code:
```
def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    return attention_mask
```

Json schema:
```
{'properties': {'function__create_4d_causal_attention_mask': {'description': 'Docstring for the function `_create_4d_causal_attention_mask`', 'title': 'Function  Create 4D Causal Attention Mask', 'type': 'string'}}, 'required': ['function__create_4d_causal_attention_mask'], 'title': 'FunctionsDocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"function__create_4d_causal_attention_mask": "Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`\\\\n\\\\nArgs:\\\\n\\\\\\\\tinput_shape (`tuple(int)` or `list(int)` or `torch.Size`):\\\\n\\\\\\\\t\\\\\\\\tThe input shape should be a tuple that defines `(batch_size, query_length)`.\\\\n\\\\\\\\tdtype (`torch.dtype`):\\\\n\\\\\\\\t\\\\\\\\tThe torch dtype the created mask shall have.\\\\n\\\\\\\\tdevice (`int`):\\\\n\\\\\\\\t\\\\\\\\tThe torch device the created mask shall have.\\\\n\\\\\\\\tsliding_window (`int`, *optional*):\\\\n\\\\\\\\t\\\\\\\\tIf the model uses windowed attention, a sliding window should be passed."}
\"\"\"
'''

Example 5:
'''
User:
Input code:
```
import dataclasses
import json
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class TrainerState:
    is_world_process_zero: bool = False
    epoch: int = 0
    global_step: int = 0
    best_model_score: float = float('inf')
    improvement_threshold: float = 1e-4
    inner_flag_matrix: dict[str, list[Union[bool, Optional[int]]]] = None

    def __post_init__(self):
        if self.inner_flag_matrix is None:
            self.inner_flag_matrix = {}

    def update_best_score(self, new_score: float) -> bool:
        if (self.best_model_score - new_score) > self.improvement_threshold:
            self.best_model_score = new_score
            return True
        return False

    def save_to_json(self, json_path: str):
        \"\"\"Saves the content of this instance inside `json_path`.\"\"\"
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
```

Json schema:
```
{'properties': {'class_TrainerState': {'description': 'Docstring for the class `TrainerState`', 'title': 'Class Trainerstate', 'type': 'string'}, 'class_TrainerState_method___post_init__': {'description': 'Docstring for the method `__post_init__` of the class `TrainerState`', 'title': 'Class Trainerstate Method   Post Init  ', 'type': 'string'}, 'class_TrainerState_method_update_best_score': {'description': 'Docstring for the method `update_best_score` of the class `TrainerState`', 'title': 'Class Trainerstate Method Update Best Score', 'type': 'string'}, 'class_TrainerState_method_save_to_json': {'description': 'Docstring for the method `save_to_json` of the class `TrainerState`', 'title': 'Class Trainerstate Method Save To Json', 'type': 'string'}, 'class_TrainerState_method_load_from_json': {'description': 'Docstring for the method `load_from_json` of the class `TrainerState`', 'title': 'Class Trainerstate Method Load From Json', 'type': 'string'}}, 'required': ['class_TrainerState', 'class_TrainerState_method___post_init__', 'class_TrainerState_method_update_best_score', 'class_TrainerState_method_save_to_json', 'class_TrainerState_method_load_from_json'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_TrainerState": "Class to track the state of a model training process.\\\\n\\\\nArgs:\\\\n\\\\tis_world_process_zero (`bool`)\\\\n\\\\tepoch (`int`):\\\\n\\\\t\\\\tCurrent epoch number during training\\\\n\\\\tglobal_step (`int`):\\\\n\\\\t\\\\tTotal number of training steps performed\\\\n\\\\tbest_model_score (`float`):\\\\n\\\\t\\\\tBest validation score achieved during training\\\\n\\\\timprovement_threshold (`float`):\\\\n\\\\t\\\\tMinimum difference required to consider a new score as an improvement\\\\n\\\\t\\\\t(helps avoid updating on insignificant improvements)\\\\n\\\\tinner_flag_matrix (`dict[str, list[Union[bool, Optional[int]]]]`)", "class_TrainerState_method___post_init__": "", "class_TrainerState_method_update_best_score": "Updates best_model_score if the new score is better (lower) by at least improvement_threshold.\\\\n\\\\nArgs:\\\\n\\\\tnew_score (`float`):\\\\n\\\\t\\\\tNew score to compare against the best score.\\\\n\\\\t\\\\nReturns:\\\\n\\\\t`bool`: True if the new score is better by at least `improvement_threshold`, False otherwise", "class_TrainerState_method_save_to_json": "Save the content of this instance in JSON format inside `json_path`.\\\\n\\\\nArgs:\\\\n\\\\tjson_path (`str`):\\\\n\\\\t\\\\tPath to the JSON file in which this instance is saved.\\\\nReturns:\\\\n\\\\tNone (saves the content in JSON format)", "class_TrainerState_method_load_from_json": "Create an instance of `TrainerState` from the content of `json_path`.\\\\n\\\\nArgs:\\\\n\\\\tjson_path (`str`):\\\\n\\\\t\\\\tPath to the JSON file that will be used to recreate the instance.\\\\n\\\\nReturns:\\\\n\\\\t`TrainerState` from the content of `json_path` file."}
\"\"\"
'''

Example 6:
'''
User:
Input code:
```
import numpy as np
from sacrebleu.metrics import BLEU

from typing import List, Dict
from .generation_metric import GenerationMetric


class BLEUMetric(GenerationMetric):
    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")
        self.scorer = BLEU(effective_order=True, lowercase=True)

    def __str__(self):
        return "BLEU"

    def _score_single(self, t1: str, t2: str):
        return self.scorer.sentence_score(
            t1.strip().rstrip("."), [t2.strip().rstrip(".")]
        ).score

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        \"\"\"
        Calculates BLEU score between stats['greedy_texts'] and target_texts.
        \"\"\"
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
```

Json schema:
```
{'properties': {'class_BLEUMetric': {'description': 'Docstring for the class `BLEUMetric`', 'title': 'Class Bleumetric', 'type': 'string'}, 'class_BLEUMetric_method___init__': {'description': 'Docstring for the method `__init__` of the class `BLEUMetric`', 'title': 'Class Bleumetric Method   Init  ', 'type': 'string'}, 'class_BLEUMetric_method___str__': {'description': 'Docstring for the method `__str__` of the class `BLEUMetric`', 'title': 'Class Bleumetric Method   Str  ', 'type': 'string'}, 'class_BLEUMetric_method__score_single': {'description': 'Docstring for the method `_score_single` of the class `BLEUMetric`', 'title': 'Class Bleumetric Method  Score Single', 'type': 'string'}, 'class_BLEUMetric_method___call__': {'description': 'Docstring for the method `__call__` of the class `BLEUMetric`', 'title': 'Class Bleumetric Method   Call  ', 'type': 'string'}}, 'required': ['class_BLEUMetric', 'class_BLEUMetric_method___init__', 'class_BLEUMetric_method___str__', 'class_BLEUMetric_method__score_single', 'class_BLEUMetric_method___call__'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_BLEUMetric": "A class for calculating the BLEU (Bilingual Evaluation Understudy) score, which is a metric for evaluating the quality of text generated by a machine translation system compared to a reference translation.\\\\n\\\\nThe BLEUMetric class inherits from the GenerationMetric class and is specifically designed to compute BLEU scores between model-generated texts and ground-truth texts.\\\\n\\\\nArgs:\\\\n\\\\tscorer (`BLEU`): An instance of the BLEU scorer from the sacrebleu library, configured with effective order and lowercase options.\\\\n\\\\nMethods:\\\\n\\\\t__init__(): Initializes the BLEUMetric object with the necessary attributes.\\\\n\\\\t__str__(): Returns the string representation of the metric, which is \'BLEU\'.\\\\n\\\\t_score_single(t1: str, t2: str): Computes the BLEU score for a single pair of hypothesis and reference texts.\\\\n\\\\t__call__(stats: Dict[str, np.ndarray], target_texts: List[str]) -> np.ndarray: Calculates BLEU scores for multiple samples by comparing the model-generated texts in `stats[\'greedy_texts\']` \\\\n\\\\twith the ground-truth texts in `target_texts`.", "class_BLEUMetric_method___init__": "", "class_BLEUMetric_method___str__": "", "class_BLEUMetric_method__score_single": "", "class_BLEUMetric_method___call__": "Calculates the BLEU score between `stats[\'greedy_texts\']` and `target_texts`.\\\\n\\\\nArgs:\\\\n\\\\tstats (`dict[str, np.ndarray]`):\\\\n\\\\t\\\\tinput statistics, which for multiple samples includes: * model-generated texts in \'greedy_texts\'\\\\n\\\\ttarget_texts (`list[str]`):\\\\n\\\\t\\\\tground-truth texts\\\\nReturns:\\\\n\\\\t`np.ndarray` with BLEU Scores for each sample in input."}
\"\"\"
'''
""".strip()

SYSTEM_PROMPT_DOCSTRING_ADDITION = """

Example 7:
'''
User:
Input code:
```
import math
from numbers import Real

import numpy as np
from scipy.special import betaln, digamma, gammaln


def _log_dirichlet_norm(dirichlet_concentration):
    return gammaln(np.sum(dirichlet_concentration)) - np.sum(
        gammaln(dirichlet_concentration)
    )


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    \"\"\"Compute log Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
    log_det_precision_chol : array-like of shape (n_components,)
    n_features : int

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
    \"\"\"
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),
            0,
        )
    )
```

Json schema:
```
{'properties': {'function__log_dirichlet_norm': {'description': 'Docstring for the function `_log_dirichlet_norm`', 'title': 'Function  Log Dirichlet Norm', 'type': 'string'}, 'function__log_wishart_norm': {'description': 'Docstring for the function `_log_wishart_norm`', 'title': 'Function  Log Wishart Norm', 'type': 'string'}}, 'required': ['function__log_dirichlet_norm', 'function__log_wishart_norm'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"function__log_dirichlet_norm": "Compute the log of the Dirichlet distribution normalization term.\\\\n\\\\nArgs:\\\\n\\\\tdirichlet_concentration (`np.ndarray`):\\\\n\\\\t\\\\tThe parameters values of the Dirichlet distribution.\\\\n\\\\nReturns:\\\\n\\\\t`float`: The log normalization of the Dirichlet distribution.", "function__log_wishart_norm": "Compute the log of the Wishart distribution normalization term.\\\\n\\\\nArgs:\\\\n\\\\tdegrees_of_freedom (`np.ndarray`):\\\\n\\\\t\\\\tThe number of degrees of freedom on the covariance Wishart distributions.\\\\n\\\\t\\\\n\\\\tlog_det_precision_chol (`np.ndarray`):\\\\n\\\\t\\\\t The determinant of the precision matrix for each component.\\\\n\\\\t\\\\n\\\\tn_features (`int`):\\\\n\\\\t\\\\tThe number of features.\\\\n\\\\nReturns:\\\\n\\\\t`np.ndarray`: The log normalization of the Wishart distribution."}
\"\"\"
'''

Example 8:
'''
User:
Input code:
```
class IterableDatasetDict(dict):
    def __repr__(self):
        repr = "\\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"IterableDatasetDict({{\\n{repr}\\n}})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Here `del` is used to del the pyarrow tables. This properly closes the files used for memory mapped tables
        for dataset in self.values():
            if hasattr(dataset, "_data"):
                del dataset._data
            if hasattr(dataset, "_indices"):
                del dataset._indices

    @property
    def data(self) -> dict[str, Table]:
        self._check_values_type()
        return {k: dataset.data for k, dataset in self.items()}

    def rename_column(self, original_column_name: str, new_column_name: str) -> "IterableDatasetDict":
        \"\"\"Rename column in dataset\"\"\"
        return IterableDatasetDict(
            {
                k: dataset.rename_column(original_column_name=original_column_name, new_column_name=new_column_name)
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: Dict[str, str]) -> "IterableDatasetDict":

        return IterableDatasetDict(
            {k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()}
        )

    def remove_columns(self, column_names: Union[str, List[str]]) -> "IterableDatasetDict":
        \"\"\"
        Remove one or several column(s) in the dataset and the features associated to them.
        The removal is done on-the-fly on the examples when iterating over the dataset.
        The removal is applied to all the datasets of the dataset dictionary.
        \"\"\"
        return IterableDatasetDict({k: dataset.remove_columns(column_names) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, List[str]]) -> "IterableDatasetDict":
        \"\"\"Select one or several column(s) in the dataset and the features
        associated to them. The selection is done on-the-fly on the examples
        when iterating over the dataset. The selection is applied to all the
        datasets of the dataset dictionary.
        Args:
            column_names (`Union[str, list[str]]`)
        \"\"\"
        return IterableDatasetDict({k: dataset.select_columns(column_names) for k, dataset in self.items()})

    def cast(
        self,
        features: Features,
    ) -> "IterableDatasetDict":
        return IterableDatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})

    def shuffle(
        self, seed=None, generator: Optional[np.random.Generator] = None, buffer_size: int = 1000
    ) -> "IterableDatasetDict":
        \"\"\"
        Shufles values of inner datasets
        
        seed (`int`, *optional*, defaults to `None`):
            Random seed that will be used to shuffle the dataset.
            It is used to sample from the shuffle buffer and also to shuffle the data shards.
        generator (`numpy.random.Generator`, *optional*):
            Numpy random Generator to use to compute the permutation of the dataset rows.
            If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
        buffer_size (`int`, defaults to `1000`):
            Size of the buffer.
        \"\"\"
        return IterableDatasetDict(
            {
                k: dataset.shuffle(seed=seed, generator=generator, buffer_size=buffer_size)
                for k, dataset in self.items()
            }
        )
```

Json schema:
```
{'properties': {'class_IterableDatasetDict': {'description': 'Docstring for the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict', 'type': 'string'}, 'class_IterableDatasetDict_method___repr__': {'description': 'Docstring for the method `__repr__` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method   Repr  ', 'type': 'string'}, 'class_IterableDatasetDict_method___enter__': {'description': 'Docstring for the method `__enter__` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method   Enter  ', 'type': 'string'}, 'class_IterableDatasetDict_method___exit__': {'description': 'Docstring for the method `__exit__` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method   Exit  ', 'type': 'string'}, 'class_IterableDatasetDict_method_data': {'description': 'Docstring for the method `data` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Data', 'type': 'string'}, 'class_IterableDatasetDict_method_rename_column': {'description': 'Docstring for the method `rename_column` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Rename Column', 'type': 'string'}, 'class_IterableDatasetDict_method_rename_columns': {'description': 'Docstring for the method `rename_columns` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Rename Columns', 'type': 'string'}, 'class_IterableDatasetDict_method_remove_columns': {'description': 'Docstring for the method `remove_columns` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Remove Columns', 'type': 'string'}, 'class_IterableDatasetDict_method_select_columns': {'description': 'Docstring for the method `select_columns` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Select Columns', 'type': 'string'}, 'class_IterableDatasetDict_method_cast': {'description': 'Docstring for the method `cast` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Cast', 'type': 'string'}, 'class_IterableDatasetDict_method_shuffle': {'description': 'Docstring for the method `shuffle` of the class `IterableDatasetDict`', 'title': 'Class Iterabledatasetdict Method Shuffle', 'type': 'string'}}, 'required': ['class_IterableDatasetDict', 'class_IterableDatasetDict_method___repr__', 'class_IterableDatasetDict_method___enter__', 'class_IterableDatasetDict_method___exit__', 'class_IterableDatasetDict_method_data', 'class_IterableDatasetDict_method_rename_column', 'class_IterableDatasetDict_method_rename_columns', 'class_IterableDatasetDict_method_remove_columns', 'class_IterableDatasetDict_method_select_columns', 'class_IterableDatasetDict_method_cast', 'class_IterableDatasetDict_method_shuffle'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_IterableDatasetDict": "A dictionary-like class that holds multiple `IterableDataset` objects. This class provides methods to manipulate the datasets collectively, such as renaming columns, removing columns, casting features, and shuffling the data.\\\\n\\\\nMethods:\\\\n\\\\t`__repr__()`: Returns a string representation of the `IterableDatasetDict`.\\\\n\\\\t`rename_column(original_column_name: str, new_column_name: str)`: Renames a column in all datasets.\\\\n\\\\t`rename_columns(column_mapping: Dict[str, str])`: Renames multiple columns in all datasets.\\\\n\\\\t`remove_columns(column_names: Union[str, List[str]])`: Removes specified columns from all datasets.\\\\n\\\\t`select_columns(column_names: Union[str, List[str]])`: Selects specified columns in all datasets.\\\\n\\\\t`cast(features: Features)`: Casts the features of all datasets.\\\\n\\\\t`shuffle(seed=None, generator: Optional[np.random.Generator] = None, buffer_size: int = 1000)`: Shuffles the data in all datasets.", "class_IterableDatasetDict_method___repr__": "", "class_IterableDatasetDict_method___enter__": "", "class_IterableDatasetDict_method___exit__": "", "class_IterableDatasetDict_method_data": "The Apache Arrow tables backing each split.\\\\n\\\\nReturns:\\\\n\\\\t`dict[str, Table]`: A dictionary mapping split names to their corresponding Arrow tables.", "class_IterableDatasetDict_method_rename_column": "Rename a column in the dataset, and move the features associated to the original column under the new column\\\\nname.\\\\nThe renaming is applied to all the datasets of the dataset dictionary.\\\\n\\\\nArgs:\\\\n\\\\toriginal_column_name (`str`):\\\\n\\\\t\\\\tName of the column to rename.\\\\n\\\\tnew_column_name (`str`):\\\\n\\\\t\\\\tNew name for the column.\\\\n\\\\nReturns:\\\\n\\\\t`IterableDatasetDict`: A copy of the dataset with a renamed column.", "class_IterableDatasetDict_method_rename_columns": "Rename several columns in the dataset, and move the features associated to the original columns under\\\\nthe new column names.\\\\nThe renaming is applied to all the datasets of the dataset dictionary.\\\\n\\\\nArgs:\\\\n\\\\tcolumn_mapping (`dict[str, str]`):\\\\n\\\\t\\\\tA mapping of columns to rename to their new names.\\\\n\\\\nReturns:\\\\n\\\\t`IterableDatasetDict`: A copy of the dataset with renamed columns", "class_IterableDatasetDict_method_remove_columns": "Remove one or several column(s) in the dataset and the features associated to them.\\\\nThe removal is done on-the-fly on the examples when iterating over the dataset.\\\\nThe removal is applied to all the datasets of the dataset dictionary.\\\\n\\\\nArgs:\\\\n\\\\tcolumn_names (`Union[str, list[str]]`):\\\\n\\\\t\\\\tName of the column(s) to remove.\\\\n\\\\nReturns:\\\\n\\\\t`IterableDatasetDict`: A copy of the dataset object without the columns to remove.", "class_IterableDatasetDict_method_select_columns": "Select one or several column(s) in the dataset and the features\\\\n\\\\tassociated to them. The selection is done on-the-fly on the examples\\\\n\\\\twhen iterating over the dataset. The selection is applied to all the\\\\n\\\\tdatasets of the dataset dictionary.\\\\n\\\\n\\\\tArgs:\\\\n\\\\t\\\\tcolumn_names (`Union[str, list[str]]`):\\\\n\\\\t\\\\t\\\\tName of the column(s) to keep.\\\\n\\\\n\\\\tReturns:\\\\n\\\\t\\\\t`IterableDatasetDict`: A copy of the dataset object with only selected columns.", "class_IterableDatasetDict_method_cast": "Cast the dataset to a new set of features.\\\\nThe type casting is applied to all the datasets of the dataset dictionary.\\\\n\\\\nArgs:\\\\n\\\\tfeatures (`Features`):\\\\n\\\\t\\\\tNew features to cast the dataset to.\\\\n\\\\t\\\\tThe name of the fields in the features must match the current column names.\\\\n\\\\t\\\\tThe type of the data must also be convertible from one type to the other.\\\\n\\\\t\\\\tFor non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`map`] to update the Dataset.\\\\n\\\\nReturns:\\\\n\\\\t`IterableDatasetDict`: A copy of the dataset with casted features.", "class_IterableDatasetDict_method_shuffle": "Randomly shuffles the elements of this dataset.\\\\nThe shuffling is applied to all the datasets of the dataset dictionary.\\\\n\\\\nThis dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer,\\\\nreplacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or\\\\nequal to the full size of the dataset is required.\\\\n\\\\nFor instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1000, then `shuffle` will\\\\ninitially select a random element from only the first 1000 elements in the buffer. Once an element is\\\\nselected, its space in the buffer is replaced by the next (i.e. 1,001-st) element,\\\\nmaintaining the 1000 element buffer.\\\\n\\\\nArgs:\\\\n\\\\tseed (`int`, *optional*, defaults to `None`):\\\\n\\\\t\\\\tRandom seed that will be used to shuffle the dataset.\\\\n\\\\t\\\\tIt is used to sample from the shuffle buffer and also to shuffle the data shards.\\\\n\\\\tgenerator (`numpy.random.Generator`, *optional*):\\\\n\\\\t\\\\tNumpy random Generator to use to compute the permutation of the dataset rows.\\\\n\\\\t\\\\tIf `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).\\\\n\\\\tbuffer_size (`int`, defaults to `1000`):\\\\n\\\\t\\\\tSize of the buffer.\\\\n\\\\t\\\\t\\\\nReturns:\\\\n\\\\t`IterableDatasetDict`: A copy of the dataset dictionary with the same datasets with shuffled rows."}
\"\"\"
'''

Example 9:
'''
User:
Input code:
```
import warnings
from math import ceil
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .image_utils import (
    ImageInput,
    get_channel_dimension_axis,
    get_image_size,
    infer_channel_dimension_format,
)
import torch

class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"

def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    rescaled_image = image.astype(np.float64) * scale  # Numpy type promotion has changed, so always upcast first
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    rescaled_image = rescaled_image.astype(dtype)  # Finally downcast to the desired dtype at the end

    return rescaled_image


def _rescale_for_pil_conversion(image):
    if image.dtype == np.uint8:
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(0 <= image) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale
```

Json schema:
```
{'properties': {'class_ChannelDimension': {'description': 'Docstring for the class `ChannelDimension`', 'title': 'Class Channeldimension', 'type': 'string'}, 'function_to_channel_dimension_format': {'description': 'Docstring for the function `to_channel_dimension_format`', 'title': 'Function To Channel Dimension Format', 'type': 'string'}, 'function_rescale': {'description': 'Docstring for the function `rescale`', 'title': 'Function Rescale', 'type': 'string'}, 'function__rescale_for_pil_conversion': {'description': 'Docstring for the function `_rescale_for_pil_conversion`', 'title': 'Function  Rescale For Pil Conversion', 'type': 'string'}}, 'required': ['class_ChannelDimension', 'function_to_channel_dimension_format', 'function_rescale', 'function__rescale_for_pil_conversion'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"class_ChannelDimension": "", "function_to_channel_dimension_format": "Converts `image` to the channel dimension format specified by `channel_dim`.\\\\n\\\\nArgs:\\\\n\\\\timage (`numpy.ndarray`):\\\\n\\\\t\\\\tThe image to have its channel dimension set.\\\\n\\\\tchannel_dim (`ChannelDimension`):\\\\n\\\\t\\\\tThe channel dimension format to use.\\\\n\\\\tinput_channel_dim (`ChannelDimension`, *optional*):\\\\n\\\\t\\\\tThe channel dimension format of the input image. If not provided, it will be inferred from the input image.\\\\n\\\\nReturns:\\\\n\\\\t`np.ndarray`: The image with the channel dimension set to `channel_dim`.", "function_rescale": "Rescales `image` by `scale`.\\\\n\\\\nArgs:\\\\n\\\\timage (`np.ndarray`):\\\\n\\\\t\\\\tThe image to rescale.\\\\n\\\\tscale (`float`):\\\\n\\\\t\\\\tThe scale to use for rescaling the image.\\\\n\\\\tdata_format (`ChannelDimension`, *optional*):\\\\n\\\\t\\\\tThe channel dimension format of the image. If not provided, it will be the same as the input image.\\\\n\\\\tdtype (`np.dtype`, *optional*, defaults to `np.float32`):\\\\n\\\\t\\\\tThe dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature\\\\n\\\\t\\\\textractors.\\\\n\\\\tinput_data_format (`ChannelDimension`, *optional*):\\\\n\\\\t\\\\tThe channel dimension format of the input image. If not provided, it will be inferred from the input image.\\\\n\\\\nReturns:\\\\n\\\\t`np.ndarray`: The rescaled image.", "function__rescale_for_pil_conversion": "Detects whether or not the image needs to be rescaled before being converted to a PIL image.\\\\n\\\\nThe assumption is that if `image` is of type `np.float` and all values are between 0 and 1, it needs to be\\\\nrescaled.\\\\n\\\\nArgs:\\\\n\\\\timage (`np.ndarray`):\\\\n\\\\t\\\\tThe image, which potentially needs to be rescaled.\\\\n\\\\nReturns:\\\\n\\\\t`bool`: True if the image needs to be rescaled, else False"}
\"\"\"
'''

Example 10:
'''
User:
Input code:
```
from typing import Optional, Union

import torch
from ..utils import add_start_docstrings, logging
from .stopping_criteria import StoppingCriteria

logger = logging.get_logger(__name__)

def is_within_length_limits(current_length: int, max_length: int) -> bool:
    return current_length < max_length and current_length >= 0

class MaxLengthCriteria(StoppingCriteria):

    max_length: int
    max_position_embeddings: Optional[int] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
        
class DynamicStateHandler:
    def __init__(self, payload: Any = None, mode: str = "auto", **kwargs):
        self._internal_state = {}
        self.payload = payload
        self._mode = mode
        self._temporary_flags = kwargs.get('flags', {})
    
    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._mode == "auto":
            return {**data, "processed": True}
        return None
```

Json schema:
```
{'properties': {'function_is_within_length_limits': {'description': 'Docstring for the function `is_within_length_limits`', 'title': 'Function Is Within Length Limits', 'type': 'string'}, 'class_MaxLengthCriteria': {'description': 'Docstring for the class `MaxLengthCriteria`', 'title': 'Class Maxlengthcriteria', 'type': 'string'}, 'class_MaxLengthCriteria_method___init__': {'description': 'Docstring for the method `__init__` of the class `MaxLengthCriteria`', 'title': 'Class Maxlengthcriteria Method   Init  ', 'type': 'string'}, 'class_MaxLengthCriteria_method___call__': {'description': 'Docstring for the method `__call__` of the class `MaxLengthCriteria`', 'title': 'Class Maxlengthcriteria Method   Call  ', 'type': 'string'}, 'class_DynamicStateHandler': {'description': 'Docstring for the class `DynamicStateHandler`', 'title': 'Class Dynamicstatehandler', 'type': 'string'}, 'class_DynamicStateHandler_method___init__': {'description': 'Docstring for the method `__init__` of the class `DynamicStateHandler`', 'title': 'Class Dynamicstatehandler Method   Init  ', 'type': 'string'}, 'class_DynamicStateHandler_method_process': {'description': 'Docstring for the method `process` of the class `DynamicStateHandler`', 'title': 'Class Dynamicstatehandler Method Process', 'type': 'string'}}, 'required': ['function_is_within_length_limits', 'class_MaxLengthCriteria', 'class_MaxLengthCriteria_method___init__', 'class_MaxLengthCriteria_method___call__', 'class_DynamicStateHandler', 'class_DynamicStateHandler_method___init__', 'class_DynamicStateHandler_method_process'], 'title': 'DocstringModel', 'type': 'object'}
```

Assistant:
\"\"\"
{"function_is_within_length_limits": "", "class_MaxLengthCriteria": "This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep\\\\nin mind for decoder-only type of transformers, this will include the initial prompted tokens.\\\\n\\\\nArgs:\\\\n\\\\tmax_length (`int`):\\\\n\\\\t\\\\tThe maximum length that the output sequence can have in number of tokens.\\\\n\\\\tmax_position_embeddings (`int`, *optional*):\\\\n\\\\t\\\\tThe maximum model length, as defined by the model\'s `config.max_position_embeddings` attribute.", "class_MaxLengthCriteria_method___init__": "", "class_MaxLengthCriteria_method___call__": "Args:\\\\n\\\\t\\\\tinput_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\\\\n\\\\t\\\\t\\\\tIndices of input sequence tokens in the vocabulary.\\\\n\\\\t\\\\tscores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):\\\\n\\\\t\\\\t\\\\tPrediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax\\\\n\\\\t\\\\t\\\\tor scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,\\\\n\\\\t\\\\t\\\\tmake sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.\\\\n\\\\t\\\\tkwargs (`Dict[str, Any]`, *optional*):\\\\n\\\\t\\\\t\\\\tAdditional stopping criteria specific kwargs.\\\\n\\\\n\\\\tReturn:\\\\n\\\\t\\\\t`torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`), where `True` indicates we stop generation\\\\n\\\\t\\\\t\\\\tfor a particular row, `True` indicates we should continue.", "class_DynamicStateHandler": "", "class_DynamicStateHandler_method___init__": "", "class_DynamicStateHandler_method_process": ""}
\"\"\"
'''
""".rstrip()

SYSTEM_PROMPT_COMMENTS_EXTENSIVE = '''
Act as an experienced Python specialist.

Write comment strings (or multistrings) for the code following the provided json schema. \
Write comments only if you are confident that your comment is correct, valid, and required \
for better understanding of the code. \
If there is a chance that your comment may be incorrect or is a hallucination, leave it empty. \
It is much more dangerous to write an incorrect comment than to leave it empty.

Example 1:
\'\'\'
Input code:
```
import numpy as np


class PortfolioAnalyzer:
    """
    A class for analyzing a portfolio of stocks. It provides methods to calculate volatility, Sharpe ratio, and optimize portfolio weights based on mean-variance optimization.
    """

    def __init__(self, portfolio_data: dict[str, list[float]]):
        """
        Initialize the PortfolioAnalyzer with portfolio data.

        Args:
            portfolio_data (`dict[str, list[float]]`):
                A dictionary with stock symbols as keys and lists of historical prices as values.
        """
        self.portfolio = portfolio_data
        self.metrics = {}

    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate the annualized volatility for a given stock.

        Args:
            symbol (`str`):
                The stock symbol for which to calculate the volatility.

        Returns:
            `float`: The annualized volatility of the stock.
        """
        prices = np.array(self.portfolio[symbol])
        returns = np.diff(prices) / prices[:-1]

        annual_volatility = np.std(returns) * np.sqrt(252)
        return annual_volatility

    def sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio for a given stock.

        Args:
            symbol (`str`):
                The stock symbol for which to calculate the Sharpe ratio.
            risk_free_rate (`float`, *optional*, defaults to `0.02`):
                The risk-free rate to use in the Sharpe ratio calculation.

        Returns:
            `float`: The Sharpe ratio of the stock.
        """
        prices = self.portfolio[symbol]
        total_return = (prices[-1] - prices[0]) / prices[0]

        annual_return = total_return * (252 / len(prices))
        volatility = self.calculate_volatility(symbol)

        return (annual_return - risk_free_rate) / volatility

    def optimize_weights(self, target_return: float = 0.15) -> np.ndarray:
        """
        Use mean-variance optimization to find optimal portfolio weights.

        Args:
            target_return (`float`, *optional*, defaults to `0.15`):
                The target return for the portfolio.

        Returns:
            `np.ndarray`: The covariance matrix of the returns, which is a step in the optimization process.

        Note:
            This method currently returns the covariance matrix instead of the optimal weights. This behavior should be corrected to return the optimal weights.
        """
        ## find optimal portfolio weights
        returns_matrix = []
        for symbol in self.portfolio:
            prices = np.array(self.portfolio[symbol])
            returns = np.diff(prices) / prices[:-1]
            returns_matrix.append(returns)

        cov_matrix = np.cov(returns_matrix)

        #  for later use
        self.metrics['covariance_matrix'] = cov_matrix
        self.metrics['mean_returns'] = np.mean(returns_matrix, axis=1)

        return cov_matrix
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: import numpy as np', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: class PortfolioAnalyzer:', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3:     def __init__(self, portfolio_data: dict[str, list[float]]):', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4:         self.portfolio = portfolio_data', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5:         self.metrics = {}', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6:     def calculate_volatility(self, symbol: str) -> float:', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': 'Comment for the line n. 7:         prices = np.array(self.portfolio[symbol])', 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8:         returns = np.diff(prices) / prices[:-1]', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9:         annual_volatility = np.std(returns) * np.sqrt(252)', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10:         return annual_volatility', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11:     def sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:', 'title': 'Line11', 'type': 'string'}, 'line12': {'description': 'Comment for the line n. 12:         prices = self.portfolio[symbol]', 'title': 'Line12', 'type': 'string'}, 'line13': {'description': 'Comment for the line n. 13:         total_return = (prices[-1] - prices[0]) / prices[0]', 'title': 'Line13', 'type': 'string'}, 'line14': {'description': 'Comment for the line n. 14:         annual_return = total_return * (252 / len(prices))', 'title': 'Line14', 'type': 'string'}, 'line15': {'description': 'Comment for the line n. 15:         volatility = self.calculate_volatility(symbol)', 'title': 'Line15', 'type': 'string'}, 'line16': {'description': 'Comment for the line n. 16:         return (annual_return - risk_free_rate) / volatility', 'title': 'Line16', 'type': 'string'}, 'line17': {'description': 'Comment for the line n. 17:     def optimize_weights(self, target_return: float = 0.15) -> np.ndarray:', 'title': 'Line17', 'type': 'string'}, 'line18': {'default': '# find optimal portfolio weights', 'description': 'Comment for the line n. 18:         returns_matrix = []', 'title': 'Line18', 'type': 'string'}, 'line19': {'description': 'Comment for the line n. 19:         for symbol in self.portfolio:', 'title': 'Line19', 'type': 'string'}, 'line20': {'description': 'Comment for the line n. 20:             prices = np.array(self.portfolio[symbol])', 'title': 'Line20', 'type': 'string'}, 'line21': {'description': 'Comment for the line n. 21:             returns = np.diff(prices) / prices[:-1]', 'title': 'Line21', 'type': 'string'}, 'line22': {'description': 'Comment for the line n. 22:             returns_matrix.append(returns)', 'title': 'Line22', 'type': 'string'}, 'line23': {'description': 'Comment for the line n. 23:         cov_matrix = np.cov(returns_matrix)', 'title': 'Line23', 'type': 'string'}, 'line24': {'default': 'for later use', 'description': "Comment for the line n. 24:         self.metrics['covariance_matrix'] = cov_matrix", 'title': 'Line24', 'type': 'string'}, 'line25': {'description': "Comment for the line n. 25:         self.metrics['mean_returns'] = np.mean(returns_matrix, axis=1)", 'title': 'Line25', 'type': 'string'}, 'line26': {'description': 'Comment for the line n. 26:         return cov_matrix', 'title': 'Line26', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11', 'line12', 'line13', 'line14', 'line15', 'line16', 'line17', 'line19', 'line20', 'line21', 'line22', 'line23', 'line25', 'line26'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "Initialize a dictionary to store calculated metrics", "line6": "", "line7": "Convert price data to percentage returns", "line8": "", "line9": "Annualize volatility (assuming daily data, multiply by sqrt(252))", "line10": "", "line11": "", "line12": "Retrieve the list of prices for the given stock symbol", "line13": "Calculate Sharpe ratio: (Return - Risk_Free_Rate) / Volatility", "line14": "Annualize the return (assuming one year of data)", "line15": "", "line16": "", "line17": "", "line18": "Create a matrix to store the returns of each stock", "line19": "Calculate the percentage returns from the prices for each stock in the portfolio", "line20": "", "line21": "", "line22": "", "line23": "Calculate the covariance matrix of the returns", "line24": "Store the results in metrics dictionary for later use", "line25": "", "line26": ""}
\'\'\'

Example 2:
\'\'\'
Input code:
```
def create_pydantic_model_class(name: str) -> dict[str, tuple[Any, Field]]:
	"""
	Creates a dictionary entry for a Pydantic model field representing a class docstring.
	
	Args:
		name (`str`):
			The name of the class.
	
	Returns:
		`dict[str, tuple[Any, Field]]`: A dictionary with a single key-value pair where the key is the formatted class name and the value is a tuple containing the type `str` and a `Field` object with a description.
	"""

	return {
		"class_"
		+ name: (str, Field(..., description=f"Docstring for the class `{name}`"))
	}
def create_pydantic_model_function(name: str) -> dict[str, tuple[Any, Field]]:
	"""
	Creates a dictionary entry for a Pydantic model field representing a function docstring.
	
	Args:
		name (`str`):
			The name of the function.
	
	Returns:
		`dict[str, tuple[Any, Field]]`: A dictionary with a single key-value pair where the key is the formatted function name and the value is a tuple containing the type `str` and a `Field` object with a description.
	"""

	return {
		"function_"
		+ name: (str, Field(..., description=f"Docstring for the function `{name}`"))
	}
def get_system_prompt(use_long_system_prompt: bool = True) -> str:
	"""
	Returns a system prompt string, optionally extended with additional content.
	
	Args:
		use_long_system_prompt (`bool`, *optional*, defaults to `True`):
			If `True`, the returned prompt includes additional content.
	
	Returns:
		`str`: The system prompt string, possibly extended with additional content.
	"""

	if use_long_system_prompt:
		return SYSTEM_PROMPT_DOCSTRING + SYSTEM_PROMPT_DOCSTRING_ADDITION
	return SYSTEM_PROMPT_DOCSTRING
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: def create_pydantic_model_class(name: str) -> dict[str, tuple[Any, Field]]:', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: \treturn {', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3: \t\t"class_"', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4: \t\t+ name: (str, Field(..., description=f"Docstring for the class `{name}`"))', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5: \t}', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6: def create_pydantic_model_function(name: str) -> dict[str, tuple[Any, Field]]:', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': 'Comment for the line n. 7: \treturn {', 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8: \t\t"function_"', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9: \t\t+ name: (str, Field(..., description=f"Docstring for the function `{name}`"))', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10: \t}', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11: def get_system_prompt(use_long_system_prompt: bool = True) -> str:', 'title': 'Line11', 'type': 'string'}, 'line12': {'description': 'Comment for the line n. 12: \tif use_long_system_prompt:', 'title': 'Line12', 'type': 'string'}, 'line13': {'description': 'Comment for the line n. 13: \t\treturn SYSTEM_PROMPT_DOCSTRING + SYSTEM_PROMPT_DOCSTRING_ADDITION', 'title': 'Line13', 'type': 'string'}, 'line14': {'description': 'Comment for the line n. 14: \treturn SYSTEM_PROMPT_DOCSTRING', 'title': 'Line14', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11', 'line12', 'line13', 'line14'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": ""}
\'\'\'

Example 3:
\'\'\'
Input code:
```
def get_docstring_position_in_code(
	node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
	code: str,
	lines_shift: int,
) -> Union[tuple[int, int], None]:
	"""
	Determines the start and end positions of a docstring within the source code of a function, async function, or class. If no docstring exists, it returns the position where a docstring should be inserted.

	Args:
		node (`Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]`):
			The AST node representing the function, async function, or class.
		code (`str`):
			The source code as a string.
		lines_shift (`int`):
			An integer shift to adjust line numbers (useful if the code snippet is part of a larger file).

	Returns:
		`Union[tuple[int, int], None]`: A tuple containing the start and end positions of the docstring in the source code, or `None` if the input node is not a function, async function, or class.
	"""

	if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
		return None

	if (
		node.body
		and isinstance(node.body[0], ast.Expr)
		and isinstance(node.body[0].value, (ast.Str, ast.Constant))
	):
		doc_node = node.body[0]
		lines = code.splitlines()

		start = sum(
			len(line) + 1 for line in lines[: doc_node.lineno - 1 + lines_shift]
		)
		start += doc_node.col_offset

        # in this case docstring is single-line
		if doc_node.lineno == doc_node.end_lineno:
			line = lines[doc_node.lineno - 1 + lines_shift]
			quote_type = line[doc_node.col_offset : doc_node.col_offset + 3]
			if quote_type in ('"""', "\'\'\'"):
				end = line.find(quote_type, doc_node.col_offset + 3) + 3
			else:
				quote_type = line[doc_node.col_offset]
				end = line.find(quote_type, doc_node.col_offset + 1) + 1
			end = start + (end - doc_node.col_offset)
		else:
			last_line = lines[doc_node.end_lineno - 1 + lines_shift].rstrip()
			end = sum(
				len(line) + 1 for line in lines[: doc_node.end_lineno - 1 + lines_shift]
			)
			end += len(last_line)

		return start, end

	pos = _get_docstring_position_with_no_docstring(node=node, code=code, lines_shift=lines_shift)
	# return twice to match the format
	return pos, pos
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: def get_docstring_position_in_code(', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: \tnode: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3: \tcode: str,', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4: \tlines_shift: int,', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5: ) -> Union[tuple[int, int], None]:', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6: \tif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': 'Comment for the line n. 7: \t\treturn None', 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8: \tif (', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9: \t\tnode.body', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10: \t\tand isinstance(node.body[0], ast.Expr)', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11: \t\tand isinstance(node.body[0].value, (ast.Str, ast.Constant))', 'title': 'Line11', 'type': 'string'}, 'line12': {'description': 'Comment for the line n. 12: \t):', 'title': 'Line12', 'type': 'string'}, 'line13': {'description': 'Comment for the line n. 13: \t\tdoc_node = node.body[0]', 'title': 'Line13', 'type': 'string'}, 'line14': {'description': 'Comment for the line n. 14: \t\tlines = code.splitlines()', 'title': 'Line14', 'type': 'string'}, 'line15': {'description': 'Comment for the line n. 15: \t\tstart = sum(', 'title': 'Line15', 'type': 'string'}, 'line16': {'description': 'Comment for the line n. 16: \t\t\tlen(line) + 1 for line in lines[: doc_node.lineno - 1 + lines_shift]', 'title': 'Line16', 'type': 'string'}, 'line17': {'description': 'Comment for the line n. 17: \t\t)', 'title': 'Line17', 'type': 'string'}, 'line18': {'description': 'Comment for the line n. 18: \t\tstart += doc_node.col_offset', 'title': 'Line18', 'type': 'string'}, 'line19': {'default': 'in this case docstring is single-line', 'description': 'Comment for the line n. 19: \t\tif doc_node.lineno == doc_node.end_lineno:', 'title': 'Line19', 'type': 'string'}, 'line20': {'description': 'Comment for the line n. 20: \t\t\tline = lines[doc_node.lineno - 1 + lines_shift]', 'title': 'Line20', 'type': 'string'}, 'line21': {'description': 'Comment for the line n. 21: \t\t\tquote_type = line[doc_node.col_offset : doc_node.col_offset + 3]', 'title': 'Line21', 'type': 'string'}, 'line22': {'description': 'Comment for the line n. 22: \t\t\tif quote_type in (\'"""\', "\'\'\'"):', 'title': 'Line22', 'type': 'string'}, 'line23': {'description': 'Comment for the line n. 23: \t\t\t\tend = line.find(quote_type, doc_node.col_offset + 3) + 3', 'title': 'Line23', 'type': 'string'}, 'line24': {'description': 'Comment for the line n. 24: \t\t\telse:', 'title': 'Line24', 'type': 'string'}, 'line25': {'description': 'Comment for the line n. 25: \t\t\t\tquote_type = line[doc_node.col_offset]', 'title': 'Line25', 'type': 'string'}, 'line26': {'description': 'Comment for the line n. 26: \t\t\t\tend = line.find(quote_type, doc_node.col_offset + 1) + 1', 'title': 'Line26', 'type': 'string'}, 'line27': {'description': 'Comment for the line n. 27: \t\t\tend = start + (end - doc_node.col_offset)', 'title': 'Line27', 'type': 'string'}, 'line28': {'description': 'Comment for the line n. 28: \t\telse:', 'title': 'Line28', 'type': 'string'}, 'line29': {'description': 'Comment for the line n. 29: \t\t\tlast_line = lines[doc_node.end_lineno - 1 + lines_shift].rstrip()', 'title': 'Line29', 'type': 'string'}, 'line30': {'description': 'Comment for the line n. 30: \t\t\tend = sum(', 'title': 'Line30', 'type': 'string'}, 'line31': {'description': 'Comment for the line n. 31: \t\t\t\tlen(line) + 1 for line in lines[: doc_node.end_lineno - 1 + lines_shift]', 'title': 'Line31', 'type': 'string'}, 'line32': {'description': 'Comment for the line n. 32: \t\t\t)', 'title': 'Line32', 'type': 'string'}, 'line33': {'description': 'Comment for the line n. 33: \t\t\tend += len(last_line)', 'title': 'Line33', 'type': 'string'}, 'line34': {'description': 'Comment for the line n. 34: \t\treturn start, end', 'title': 'Line34', 'type': 'string'}, 'line35': {'description': 'Comment for the line n. 35: \tpos = _get_docstring_position_with_no_docstring(node=node, code=code, lines_shift=lines_shift)', 'title': 'Line35', 'type': 'string'}, 'line36': {'default': 'return twice to match the format', 'description': 'Comment for the line n. 36: \treturn pos, pos', 'title': 'Line36', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11', 'line12', 'line13', 'line14', 'line15', 'line16', 'line17', 'line18', 'line20', 'line21', 'line22', 'line23', 'line24', 'line25', 'line26', 'line27', 'line28', 'line29', 'line30', 'line31', 'line32', 'line33', 'line34', 'line35'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "Check if the docstring already exists", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "Get the AST node of the docstring", "line14": "", "line15": "Calculate docstring's start position by:\\n1. Summing lengths of all previous lines (including newline chars)\\n2. Adding the column offset of the current line (number of preceding chars in the current line)", "line16": "", "line17": "", "line18": "", "line19": "Check if the docstring is single-line", "line20": "Get the line containing the docstring", "line21": "Determine the type of the quote used for the docstring", "line22": "Check if the docstring is enclosed in triple quotes", "line23": "Find the position of the closing triple quote", "line24": "Handle single quotes", "line25": "Determine the type of the quote used for the docstring", "line26": "Find the position of the closing single quote", "line27": "Calculate the end position of the docstring", "line28": "Handle multi-line docstrings", "line29": "Get the last line of the docstring", "line30": "Calculate the starting position of the last line of the docstring", "line31": "", "line32": "", "line33": "Add the length of the last line to the end position", "line34": "Return the start and end positions of the docstring", "line35": "Get the position where a docstring should be inserted if it doesn't exist", "line36": "Return the position twice since the start and end positions are the same"}
\'\'\'

Example 4:
\'\'\'
Input code:
```
from typing import Union, Optional, Any

class DynamicStateHandler:
    def __init__(self, payload: Any = None, mode: str = "auto", **kwargs):
        self._internal_state = {}
        self.payload = payload
        self._mode = mode
        self._temporary_flags = kwargs.get('flags', {})
    
    def process(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        if self._mode == "auto":
            return {**data, "processed": True}
        return None
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: from typing import Union, Optional, Any', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: class DynamicStateHandler:', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3:     def __init__(self, payload: Any = None, mode: str = "auto", **kwargs):', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4:         self._internal_state = {}', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5:         self.payload = payload', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6:         self._mode = mode', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': "Comment for the line n. 7:         self._temporary_flags = kwargs.get('flags', {})", 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8:     def process(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9:         if self._mode == "auto":', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10:             return {**data, "processed": True}', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11:         return None', 'title': 'Line11', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": ""}
\'\'\'

Example 5:
\'\'\'
Input code:
```
import time

async def fetch_user_profile(user_id: int) -> dict:
	"""
   	Asynchronously fetches a user profile from an API and returns it as a dictionary.
	
   	Args:
   		user_id (`int`):
   			The ID of the user whose profile is to be fetched.
	
   	Returns:
   		`dict`: A dictionary containing the user's profile information, including:
   			- `id` (`int`): The user's ID.
   			- `name` (`str`): The user's name.
   			- `email` (`str`): The user's email address.
   			- `created_at` (`str`): The timestamp of when the user profile was created.
   			- `request_time` (`float`): The time taken to fetch the user profile in seconds.
	"""
	start_time = time.time()
	async with aiohttp.ClientSession() as session:
		url = f"https://api.example.com/users/{user_id}"
		response = await session.get(url)
		data = await response.json()
		
	profile = {
		"id": user_id,
		"name": data["name"],
		"email": data["email"],
		"created_at": data["created_at"],
	}
	profile["request_time"] = time.time() - start_time
	return profile
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: import time', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: async def fetch_user_profile(user_id: int) -> dict:', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3: \tstart_time = time.time()', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4: \tasync with aiohttp.ClientSession() as session:', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5: \t\turl = f"https://api.example.com/users/{user_id}"', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6: \t\tresponse = await session.get(url)', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': 'Comment for the line n. 7: \t\tdata = await response.json()', 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8: \tprofile = {', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9: \t\t"id": user_id,', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10: \t\t"name": data["name"],', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11: \t\t"email": data["email"],', 'title': 'Line11', 'type': 'string'}, 'line12': {'description': 'Comment for the line n. 12: \t\t"created_at": data["created_at"],', 'title': 'Line12', 'type': 'string'}, 'line13': {'description': 'Comment for the line n. 13: \t}', 'title': 'Line13', 'type': 'string'}, 'line14': {'description': 'Comment for the line n. 14: \tprofile["request_time"] = time.time() - start_time', 'title': 'Line14', 'type': 'string'}, 'line15': {'description': 'Comment for the line n. 15: \treturn profile', 'title': 'Line15', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11', 'line12', 'line13', 'line14', 'line15'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "Create an asynchronous HTTP session", "line5": "", "line6": "Fetch the user profile with a GET request to the API.", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": "", "line15": ""}
\'\'\'

Example 6:
\'\'\'
Input code:
```
import ast
from openai import Client
from .components import add_docstrings, annotate_arguments, potentially_add_class_to_typing_import
from .utils.utils import _get_nodes_dict_with_functions_classes_methods

def _check_no_duplicating_methods(nodes: list[ast.AST]) -> None:
	"""
   	Checks for duplicate function and class method names within a list of AST nodes. Raises a `ValueError` if any duplicates are found.
	
   	Args:
   		nodes (`list[ast.AST]`):
   			A list of AST nodes to check for duplicate function and class method names.
	
   	Returns:
   		None
	
   	Raises:
   		ValueError: If any function or class method names are duplicated.
	"""
	functions_names = []
	classes_names = []
	for node in nodes:
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			functions_names.append(node.name)
		elif isinstance(node, ast.ClassDef):
			classes_names.append(node.name)
			class_methods_names = []
			for subnode in node.body:
				if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
					class_methods_names.append(subnode.name)
			#methods
			if len(set(class_methods_names)) != len(class_methods_names):
				for method in class_methods_names:
					if class_methods_names.count(method) != 1:
						raise ValueError(
							f"Method {method} is duplicated for class {node.name}."
							+ "Please fix this before running annotation."
						)

	#functions
	if len(set(functions_names)) != len(functions_names):
		for func in functions_names:
			if functions_names.count(func) != 1:
				raise ValueError(
					f"Function {func} is duplicated."
					+ "Please fix this before running annotation."
				)
    #classes
	if len(set(classes_names)) != len(classes_names):
		for class_ in classes_names:
			if classes_names.count(class_) != 1:
				raise ValueError(
					f"Class {class_} is duplicated."
					+ "Please fix this before running annotation."
				)

def document_python_code(
	code: str,
	client: Client,
	modify_existing_annotation: bool = False,
	model_checkpoint: str = "Qwen/Qwen2.5-Coder-32B-Instruct-fast",
) -> Generator[str, None, None]:
	"""
	Generates and adds docstrings to Python code using an OpenAI client. The function processes the code to annotate arguments and add docstrings to functions, classes, and methods. It also ensures that necessary typing imports are included.
	
	Args:
		code (`str`):
			The Python code as a string to which docstrings and argument annotations will be added.
		client (`Client`):
			An instance of the OpenAI client used to generate docstrings and annotations.
		modify_existing_annotation (`bool`, *optional*, defaults to `False`):
			If `True`, existing annotations and docstrings will be modified. If `False`, existing annotations and docstrings will be preserved unless they are empty.
		model_checkpoint (`str`, *optional*, defaults to `'Qwen/Qwen2.5-Coder-32B-Instruct-fast'`):
			The model checkpoint to use for generating docstrings and annotations.
	
	Yields:
		`str`: The updated code with added or modified docstrings and argument annotations.
	
	Raises:
		`SyntaxError`: If the generated code is not valid Python syntax.
	"""
	code = rf"{code}" # read it raw to avoid ast errors
	required_typing_imports = set()
	body = ast.parse(code).body

	_check_no_duplicating_classes_methods_functions(body)
	target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(body)

	output = None
	for output in annotate_arguments(
		target_nodes_dict=target_nodes_dict,
		code=code,
		client=client,
		modify_existing_annotation=modify_existing_annotation,
		model_checkpoint=model_checkpoint,
	):
		if isinstance(output, str):
			yield output
		code = output
	if isinstance(output, tuple):
		code, func_required_typing_imports = output
		required_typing_imports = required_typing_imports.union(
			func_required_typing_imports
		)

	code = code.replace(" " * 4, "	")
	tree = ast.parse(code)
	target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(tree.body)

	for output in add_docstrings(
		target_nodes_dict=target_nodes_dict,
		code=code,
		client=client,
		modify_existing_annotation=modify_existing_annotation,
		model_checkpoint=model_checkpoint,
	):
		yield output
		code = output
	# Add classes from typing that were not imported
	for typing_class in required_typing_imports:
		code = potentially_add_class_to_typing_import(code, tree, typing_class)
		yield code
	code = code.replace(" " * 4, "	")
	ast.parse(code) # to check it is valid
	yield code
```

Json schema:
```
{'properties': {'line1': {'description': 'Comment for the line n. 1: import ast', 'title': 'Line1', 'type': 'string'}, 'line2': {'description': 'Comment for the line n. 2: from openai import Client', 'title': 'Line2', 'type': 'string'}, 'line3': {'description': 'Comment for the line n. 3: from .components import add_docstrings, annotate_arguments, potentially_add_class_to_typing_import', 'title': 'Line3', 'type': 'string'}, 'line4': {'description': 'Comment for the line n. 4: from .utils.utils import _get_nodes_dict_with_functions_classes_methods', 'title': 'Line4', 'type': 'string'}, 'line5': {'description': 'Comment for the line n. 5: def _check_no_duplicating_methods(nodes: list[ast.AST]) -> None:', 'title': 'Line5', 'type': 'string'}, 'line6': {'description': 'Comment for the line n. 6: \tfunctions_names = []', 'title': 'Line6', 'type': 'string'}, 'line7': {'description': 'Comment for the line n. 7: \tclasses_names = []', 'title': 'Line7', 'type': 'string'}, 'line8': {'description': 'Comment for the line n. 8: \tfor node in nodes:', 'title': 'Line8', 'type': 'string'}, 'line9': {'description': 'Comment for the line n. 9: \t\tif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):', 'title': 'Line9', 'type': 'string'}, 'line10': {'description': 'Comment for the line n. 10: \t\t\tfunctions_names.append(node.name)', 'title': 'Line10', 'type': 'string'}, 'line11': {'description': 'Comment for the line n. 11: \t\telif isinstance(node, ast.ClassDef):', 'title': 'Line11', 'type': 'string'}, 'line12': {'description': 'Comment for the line n. 12: \t\t\tclasses_names.append(node.name)', 'title': 'Line12', 'type': 'string'}, 'line13': {'description': 'Comment for the line n. 13: \t\t\tclass_methods_names = []', 'title': 'Line13', 'type': 'string'}, 'line14': {'description': 'Comment for the line n. 14: \t\t\tfor subnode in node.body:', 'title': 'Line14', 'type': 'string'}, 'line15': {'description': 'Comment for the line n. 15: \t\t\t\tif isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):', 'title': 'Line15', 'type': 'string'}, 'line16': {'description': 'Comment for the line n. 16: \t\t\t\t\tclass_methods_names.append(subnode.name)', 'title': 'Line16', 'type': 'string'}, 'line17': {'default': 'methods', 'description': 'Comment for the line n. 17: \t\t\tif len(set(class_methods_names)) != len(class_methods_names):', 'title': 'Line17', 'type': 'string'}, 'line18': {'description': 'Comment for the line n. 18: \t\t\t\tfor method in class_methods_names:', 'title': 'Line18', 'type': 'string'}, 'line19': {'description': 'Comment for the line n. 19: \t\t\t\t\tif class_methods_names.count(method) != 1:', 'title': 'Line19', 'type': 'string'}, 'line20': {'description': 'Comment for the line n. 20: \t\t\t\t\t\traise ValueError(', 'title': 'Line20', 'type': 'string'}, 'line21': {'description': 'Comment for the line n. 21: \t\t\t\t\t\t\tf"Method {method} is duplicated for class {node.name}."', 'title': 'Line21', 'type': 'string'}, 'line22': {'description': 'Comment for the line n. 22: \t\t\t\t\t\t\t+ "Please fix this before running annotation."', 'title': 'Line22', 'type': 'string'}, 'line23': {'description': 'Comment for the line n. 23: \t\t\t\t\t\t)', 'title': 'Line23', 'type': 'string'}, 'line24': {'default': 'functions', 'description': 'Comment for the line n. 24: \tif len(set(functions_names)) != len(functions_names):', 'title': 'Line24', 'type': 'string'}, 'line25': {'description': 'Comment for the line n. 25: \t\tfor func in functions_names:', 'title': 'Line25', 'type': 'string'}, 'line26': {'description': 'Comment for the line n. 26: \t\t\tif functions_names.count(func) != 1:', 'title': 'Line26', 'type': 'string'}, 'line27': {'description': 'Comment for the line n. 27: \t\t\t\traise ValueError(', 'title': 'Line27', 'type': 'string'}, 'line28': {'description': 'Comment for the line n. 28: \t\t\t\t\tf"Function {func} is duplicated."', 'title': 'Line28', 'type': 'string'}, 'line29': {'description': 'Comment for the line n. 29: \t\t\t\t\t+ "Please fix this before running annotation."', 'title': 'Line29', 'type': 'string'}, 'line30': {'description': 'Comment for the line n. 30: \t\t\t\t)', 'title': 'Line30', 'type': 'string'}, 'line31': {'default': 'classes', 'description': 'Comment for the line n. 31: \tif len(set(classes_names)) != len(classes_names):', 'title': 'Line31', 'type': 'string'}, 'line32': {'description': 'Comment for the line n. 32: \t\tfor class_ in classes_names:', 'title': 'Line32', 'type': 'string'}, 'line33': {'description': 'Comment for the line n. 33: \t\t\tif classes_names.count(class_) != 1:', 'title': 'Line33', 'type': 'string'}, 'line34': {'description': 'Comment for the line n. 34: \t\t\t\traise ValueError(', 'title': 'Line34', 'type': 'string'}, 'line35': {'description': 'Comment for the line n. 35: \t\t\t\t\tf"Class {class_} is duplicated."', 'title': 'Line35', 'type': 'string'}, 'line36': {'description': 'Comment for the line n. 36: \t\t\t\t\t+ "Please fix this before running annotation."', 'title': 'Line36', 'type': 'string'}, 'line37': {'description': 'Comment for the line n. 37: \t\t\t\t)', 'title': 'Line37', 'type': 'string'}, 'line38': {'description': 'Comment for the line n. 38: def document_python_code(', 'title': 'Line38', 'type': 'string'}, 'line39': {'description': 'Comment for the line n. 39: \tcode: str,', 'title': 'Line39', 'type': 'string'}, 'line40': {'description': 'Comment for the line n. 40: \tclient: Client,', 'title': 'Line40', 'type': 'string'}, 'line41': {'description': 'Comment for the line n. 41: \tmodify_existing_annotation: bool = False,', 'title': 'Line41', 'type': 'string'}, 'line42': {'description': 'Comment for the line n. 42: \tmodel_checkpoint: str = "Qwen/Qwen2.5-Coder-32B-Instruct-fast",', 'title': 'Line42', 'type': 'string'}, 'line43': {'description': 'Comment for the line n. 43: ) -> Generator[str, None, None]:', 'title': 'Line43', 'type': 'string'}, 'line44': {'default': 'read it raw to avoid ast errors', 'description': 'Comment for the line n. 44: \tcode = rf"{code}" # read it raw to avoid ast errors', 'title': 'Line44', 'type': 'string'}, 'line45': {'description': 'Comment for the line n. 45: \trequired_typing_imports = set()', 'title': 'Line45', 'type': 'string'}, 'line46': {'description': 'Comment for the line n. 46: \tbody = ast.parse(code).body', 'title': 'Line46', 'type': 'string'}, 'line47': {'description': 'Comment for the line n. 47: \t_check_no_duplicating_classes_methods_functions(body)', 'title': 'Line47', 'type': 'string'}, 'line48': {'description': 'Comment for the line n. 48: \ttarget_nodes_dict = _get_nodes_dict_with_functions_classes_methods(body)', 'title': 'Line48', 'type': 'string'}, 'line49': {'description': 'Comment for the line n. 49: \toutput = None', 'title': 'Line49', 'type': 'string'}, 'line50': {'description': 'Comment for the line n. 50: \tfor output in annotate_arguments(', 'title': 'Line50', 'type': 'string'}, 'line51': {'description': 'Comment for the line n. 51: \t\ttarget_nodes_dict=target_nodes_dict,', 'title': 'Line51', 'type': 'string'}, 'line52': {'description': 'Comment for the line n. 52: \t\tcode=code,', 'title': 'Line52', 'type': 'string'}, 'line53': {'description': 'Comment for the line n. 53: \t\tclient=client,', 'title': 'Line53', 'type': 'string'}, 'line54': {'description': 'Comment for the line n. 54: \t\tmodify_existing_annotation=modify_existing_annotation,', 'title': 'Line54', 'type': 'string'}, 'line55': {'description': 'Comment for the line n. 55: \t\tmodel_checkpoint=model_checkpoint,', 'title': 'Line55', 'type': 'string'}, 'line56': {'description': 'Comment for the line n. 56: \t):', 'title': 'Line56', 'type': 'string'}, 'line57': {'description': 'Comment for the line n. 57: \t\tif isinstance(output, str):', 'title': 'Line57', 'type': 'string'}, 'line58': {'description': 'Comment for the line n. 58: \t\t\tyield output', 'title': 'Line58', 'type': 'string'}, 'line59': {'description': 'Comment for the line n. 59: \t\tcode = output', 'title': 'Line59', 'type': 'string'}, 'line60': {'description': 'Comment for the line n. 60: \tif isinstance(output, tuple):', 'title': 'Line60', 'type': 'string'}, 'line61': {'description': 'Comment for the line n. 61: \t\tcode, func_required_typing_imports = output', 'title': 'Line61', 'type': 'string'}, 'line62': {'description': 'Comment for the line n. 62: \t\trequired_typing_imports = required_typing_imports.union(', 'title': 'Line62', 'type': 'string'}, 'line63': {'description': 'Comment for the line n. 63: \t\t\tfunc_required_typing_imports', 'title': 'Line63', 'type': 'string'}, 'line64': {'description': 'Comment for the line n. 64: \t\t)', 'title': 'Line64', 'type': 'string'}, 'line65': {'description': 'Comment for the line n. 65: \tcode = code.replace(" " * 4, "\t")', 'title': 'Line65', 'type': 'string'}, 'line66': {'description': 'Comment for the line n. 66: \ttree = ast.parse(code)', 'title': 'Line66', 'type': 'string'}, 'line67': {'description': 'Comment for the line n. 67: \ttarget_nodes_dict = _get_nodes_dict_with_functions_classes_methods(tree.body)', 'title': 'Line67', 'type': 'string'}, 'line68': {'description': 'Comment for the line n. 68: \tfor output in add_docstrings(', 'title': 'Line68', 'type': 'string'}, 'line69': {'description': 'Comment for the line n. 69: \t\ttarget_nodes_dict=target_nodes_dict,', 'title': 'Line69', 'type': 'string'}, 'line70': {'description': 'Comment for the line n. 70: \t\tcode=code,', 'title': 'Line70', 'type': 'string'}, 'line71': {'description': 'Comment for the line n. 71: \t\tclient=client,', 'title': 'Line71', 'type': 'string'}, 'line72': {'description': 'Comment for the line n. 72: \t\tmodify_existing_annotation=modify_existing_annotation,', 'title': 'Line72', 'type': 'string'}, 'line73': {'description': 'Comment for the line n. 73: \t\tmodel_checkpoint=model_checkpoint,', 'title': 'Line73', 'type': 'string'}, 'line74': {'description': 'Comment for the line n. 74: \t):', 'title': 'Line74', 'type': 'string'}, 'line75': {'description': 'Comment for the line n. 75: \t\tyield output', 'title': 'Line75', 'type': 'string'}, 'line76': {'description': 'Comment for the line n. 76: \t\tcode = output', 'title': 'Line76', 'type': 'string'}, 'line77': {'default': 'Add classes from typing that were not imported', 'description': 'Comment for the line n. 77: \tfor typing_class in required_typing_imports:', 'title': 'Line77', 'type': 'string'}, 'line78': {'description': 'Comment for the line n. 78: \t\tcode = potentially_add_class_to_typing_import(code, tree, typing_class)', 'title': 'Line78', 'type': 'string'}, 'line79': {'description': 'Comment for the line n. 79: \t\tyield code', 'title': 'Line79', 'type': 'string'}, 'line80': {'description': 'Comment for the line n. 80: \tcode = code.replace(" " * 4, "\t")', 'title': 'Line80', 'type': 'string'}, 'line81': {'default': 'to check it is valid', 'description': 'Comment for the line n. 81: \tast.parse(code) # to check it is valid', 'title': 'Line81', 'type': 'string'}, 'line82': {'description': 'Comment for the line n. 82: \tyield code', 'title': 'Line82', 'type': 'string'}}, 'required': ['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10', 'line11', 'line12', 'line13', 'line14', 'line15', 'line16', 'line18', 'line19', 'line20', 'line21', 'line22', 'line23', 'line25', 'line26', 'line27', 'line28', 'line29', 'line30', 'line32', 'line33', 'line34', 'line35', 'line36', 'line37', 'line38', 'line39', 'line40', 'line41', 'line42', 'line43', 'line45', 'line46', 'line47', 'line48', 'line49', 'line50', 'line51', 'line52', 'line53', 'line54', 'line55', 'line56', 'line57', 'line58', 'line59', 'line60', 'line61', 'line62', 'line63', 'line64', 'line65', 'line66', 'line67', 'line68', 'line69', 'line70', 'line71', 'line72', 'line73', 'line74', 'line75', 'line76', 'line78', 'line79', 'line80', 'line82'], 'title': 'CodeCommentsModel', 'type': 'object'}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": "", "line15": "", "line16": "", "line17": "Check if there are duplicate method names in the current class", "line18": "", "line19": "Check if the method name appears more than once", "line20": "", "line21": "", "line22": "", "line23": "", "line24": "Check if there are duplicate function names", "line25": "", "line26": "Check if the function name appears more than once", "line27": "", "line28": "", "line29": "", "line30": "", "line31": "Check if there are duplicate class names", "line32": "", "line33": "Check if the class name appears more than once", "line34": "", "line35": "", "line36": "", "line37": "", "line38": "", "line39": "", "line40": "", "line41": "", "line42": "", "line43": "", "line44": "Read the code as a raw string to avoid AST parsing errors", "line45": "", "line46": "Parse the code into an AST", "line47": "", "line48": "Get a dictionary of nodes that need to be annotated or documented", "line49": "Annotate the arguments of functions, classes, and methods", "line50": "", "line51": "", "line52": "", "line53": "", "line54": "", "line55": "", "line56": "", "line57": "", "line58": "", "line59": "", "line60": "", "line61": "Get the required imports from the `typing` package that will need to be added in the end", "line62": "", "line63": "", "line64": "", "line65": "Replace spaces with tabs for consistent formatting", "line66": "Lines may have changed, so it's easier to rerun `ast.parse` which takes < 1ms than track\\nthis throughout the code", "line67": "Get dictionary with target nodes with the updated AST code", "line68": "Add docstrings to functions, classes, and methods", "line69": "", "line70": "", "line71": "", "line72": "", "line73": "", "line74": "", "line75": "", "line76": "", "line77": "If there are classes from the `typing` package that were used for annotation but not imported,\\nadd them to the imports", "line78": "", "line79": "", "line80": "Replace spaces with tabs for consistent formatting", "line81": "Make sure the generated code has valid Python syntax", "line82": ""}
\'\'\'
'''.strip()

SYSTEM_PROMPT_COMMENTS = '''
Act as an experienced Python specialist.

Write comment strings (or multistrings) for the code where necessary following the provided json schema. \
Write comments only if you are confident that your comment is correct, valid, and required in this place \
for better understanding of the code. \
If the line is easy to understand and doesn't require a comment, leave the comment empty. \
If there is a chance that your comment may be incorrect or is a hallucination, leave it empty as well. \
It is much more dangerous to write an incorrect comment than to leave it empty.

Example 1:
\'\'\'
Input code:
```
import numpy as np


class PortfolioAnalyzer:
    """
    A class for analyzing a portfolio of stocks. It provides methods to calculate volatility, Sharpe ratio, and optimize portfolio weights based on mean-variance optimization.
    """

    def __init__(self, portfolio_data: dict[str, list[float]]):
        """
        Initialize the PortfolioAnalyzer with portfolio data.

        Args:
            portfolio_data (`dict[str, list[float]]`):
                A dictionary with stock symbols as keys and lists of historical prices as values.
        """
        self.portfolio = portfolio_data
        self.metrics = {}

    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate the annualized volatility for a given stock.

        Args:
            symbol (`str`):
                The stock symbol for which to calculate the volatility.

        Returns:
            `float`: The annualized volatility of the stock.
        """
        prices = np.array(self.portfolio[symbol])
        returns = np.diff(prices) / prices[:-1]

        annual_volatility = np.std(returns) * np.sqrt(252)
        return annual_volatility

    def sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio for a given stock.

        Args:
            symbol (`str`):
                The stock symbol for which to calculate the Sharpe ratio.
            risk_free_rate (`float`, *optional*, defaults to `0.02`):
                The risk-free rate to use in the Sharpe ratio calculation.

        Returns:
            `float`: The Sharpe ratio of the stock.
        """
        prices = self.portfolio[symbol]
        total_return = (prices[-1] - prices[0]) / prices[0]

        annual_return = total_return * (252 / len(prices))
        volatility = self.calculate_volatility(symbol)

        return (annual_return - risk_free_rate) / volatility

    def optimize_weights(self, target_return: float = 0.15) -> np.ndarray:
        """
        Use mean-variance optimization to find optimal portfolio weights.

        Args:
            target_return (`float`, *optional*, defaults to `0.15`):
                The target return for the portfolio.

        Returns:
            `np.ndarray`: The covariance matrix of the returns, which is a step in the optimization process.

        Note:
            This method currently returns the covariance matrix instead of the optimal weights. This behavior should be corrected to return the optimal weights.
        """
        ## find optimal portfolio weights
        returns_matrix = []
        for symbol in self.portfolio:
            prices = np.array(self.portfolio[symbol])
            returns = np.diff(prices) / prices[:-1]
            returns_matrix.append(returns)

        cov_matrix = np.cov(returns_matrix)

        #  for later use
        self.metrics['covariance_matrix'] = cov_matrix
        self.metrics['mean_returns'] = np.mean(returns_matrix, axis=1)

        return cov_matrix
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: import numpy as np", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: class PortfolioAnalyzer:", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3:     def __init__(self, portfolio_data: dict[str, list[float]]):", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4:         self.portfolio = portfolio_data", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5:         self.metrics = {}", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6:     def calculate_volatility(self, symbol: str) -> float:", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7:         prices = np.array(self.portfolio[symbol])", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8:         returns = np.diff(prices) / prices[:-1]", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9:         annual_volatility = np.std(returns) * np.sqrt(252)", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10:         return annual_volatility", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11:     def sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.02) -> float:", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12:         prices = self.portfolio[symbol]", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13:         total_return = (prices[-1] - prices[0]) / prices[0]", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14:         annual_return = total_return * (252 / len(prices))", "title": "Line14"}, "line15": {"description": "Comment for the line n. 15:         volatility = self.calculate_volatility(symbol)", "title": "Line15"}, "line16": {"description": "Comment for the line n. 16:         return (annual_return - risk_free_rate) / volatility", "title": "Line16"}, "line17": {"description": "Comment for the line n. 17:     def optimize_weights(self, target_return: float = 0.15) -> np.ndarray:", "title": "Line17"}, "line18": {"default": "# find optimal portfolio weights", "description": "Comment for the line n. 18:         returns_matrix = []", "title": "Line18"}, "line19": {"description": "Comment for the line n. 19:         for symbol in self.portfolio:", "title": "Line19"}, "line20": {"description": "Comment for the line n. 20:             prices = np.array(self.portfolio[symbol])", "title": "Line20"}, "line21": {"description": "Comment for the line n. 21:             returns = np.diff(prices) / prices[:-1]", "title": "Line21"}, "line22": {"description": "Comment for the line n. 22:             returns_matrix.append(returns)", "title": "Line22"}, "line23": {"description": "Comment for the line n. 23:         cov_matrix = np.cov(returns_matrix)", "title": "Line23"}, "line24": {"default": "for later use", "description": "Comment for the line n. 24:         self.metrics[\'covariance_matrix\'] = cov_matrix", "title": "Line24"}, "line25": {"description": "Comment for the line n. 25:         self.metrics[\'mean_returns\'] = np.mean(returns_matrix, axis=1)", "title": "Line25"}, "line26": {"description": "Comment for the line n. 26:         return cov_matrix", "title": "Line26"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "Convert price data to percentage returns", "line8": "", "line9": "Annualize volatility (assuming daily data, multiply by sqrt(252))", "line10": "", "line11": "", "line12": "", "line13": "Calculate Sharpe ratio: (Return - Risk_Free_Rate) / Volatility", "line14": "Annualize the return (assuming one year of data)", "line15": "", "line16": "", "line17": "", "line18": "Create a matrix to store the returns of each stock", "line19": "Calculate the percentage returns from the prices for each stock in the portfolio", "line20": "", "line21": "", "line22": "", "line23": "Calculate the covariance matrix of the returns", "line24": "Store the results in metrics dictionary for later use", "line25": "", "line26": ""}
\'\'\'

Example 2:
\'\'\'
Input code:
```
def create_pydantic_model_class(name: str) -> dict[str, tuple[Any, Field]]:
	"""
	Creates a dictionary entry for a Pydantic model field representing a class docstring.

	Args:
		name (`str`):
			The name of the class.

	Returns:
		`dict[str, tuple[Any, Field]]`: A dictionary with a single key-value pair where the key is the formatted class name and the value is a tuple containing the type `str` and a `Field` object with a description.
	"""

	return {
		"class_"
		+ name: (str, Field(..., description=f"Docstring for the class `{name}`"))
	}
def create_pydantic_model_function(name: str) -> dict[str, tuple[Any, Field]]:
	"""
	Creates a dictionary entry for a Pydantic model field representing a function docstring.

	Args:
		name (`str`):
			The name of the function.

	Returns:
		`dict[str, tuple[Any, Field]]`: A dictionary with a single key-value pair where the key is the formatted function name and the value is a tuple containing the type `str` and a `Field` object with a description.
	"""

	return {
		"function_"
		+ name: (str, Field(..., description=f"Docstring for the function `{name}`"))
	}
def get_system_prompt(use_long_system_prompt: bool = True) -> str:
	"""
	Returns a system prompt string, optionally extended with additional content.

	Args:
		use_long_system_prompt (`bool`, *optional*, defaults to `True`):
			If `True`, the returned prompt includes additional content.

	Returns:
		`str`: The system prompt string, possibly extended with additional content.
	"""

	if use_long_system_prompt:
		return SYSTEM_PROMPT_DOCSTRING + SYSTEM_PROMPT_DOCSTRING_ADDITION
	return SYSTEM_PROMPT_DOCSTRING
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: def create_pydantic_model_class(name: str) -> dict[str, tuple[Any, Field]]:", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: \\treturn {", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3: \\t\\t\\"class_\\"", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4: \\t\\t+ name: (str, Field(..., description=f\\"Docstring for the class `{name}`\\"))", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5: \\t}", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6: def create_pydantic_model_function(name: str) -> dict[str, tuple[Any, Field]]:", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7: \\treturn {", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8: \\t\\t\\"function_\\"", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9: \\t\\t+ name: (str, Field(..., description=f\\"Docstring for the function `{name}`\\"))", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10: \\t}", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11: def get_system_prompt(use_long_system_prompt: bool = True) -> str:", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12: \\tif use_long_system_prompt:", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13: \\t\\treturn SYSTEM_PROMPT_DOCSTRING + SYSTEM_PROMPT_DOCSTRING_ADDITION", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14: \\treturn SYSTEM_PROMPT_DOCSTRING", "title": "Line14"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": ""}
\'\'\'

Example 3:
\'\'\'
Input code:
```
def get_docstring_position_in_code(
	node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
	code: str,
	lines_shift: int,
) -> Union[tuple[int, int], None]:
	"""
	Determines the start and end positions of a docstring within the source code of a function, async function, or class. If no docstring exists, it returns the position where a docstring should be inserted.

	Args:
		node (`Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]`):
			The AST node representing the function, async function, or class.
		code (`str`):
			The source code as a string.
		lines_shift (`int`):
			An integer shift to adjust line numbers (useful if the code snippet is part of a larger file).

	Returns:
		`Union[tuple[int, int], None]`: A tuple containing the start and end positions of the docstring in the source code, or `None` if the input node is not a function, async function, or class.
	"""

	if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
		return None

	if (
		node.body
		and isinstance(node.body[0], ast.Expr)
		and isinstance(node.body[0].value, (ast.Str, ast.Constant))
	):
		doc_node = node.body[0]
		lines = code.splitlines()

		start = sum(
			len(line) + 1 for line in lines[: doc_node.lineno - 1 + lines_shift]
		)
		start += doc_node.col_offset

        # in this case docstring is single-line
		if doc_node.lineno == doc_node.end_lineno:
			line = lines[doc_node.lineno - 1 + lines_shift]
			quote_type = line[doc_node.col_offset : doc_node.col_offset + 3]
			if quote_type in ('"""', "\'\'\'"):
				end = line.find(quote_type, doc_node.col_offset + 3) + 3
			else:
				quote_type = line[doc_node.col_offset]
				end = line.find(quote_type, doc_node.col_offset + 1) + 1
			end = start + (end - doc_node.col_offset)
		else:
			last_line = lines[doc_node.end_lineno - 1 + lines_shift].rstrip()
			end = sum(
				len(line) + 1 for line in lines[: doc_node.end_lineno - 1 + lines_shift]
			)
			end += len(last_line)

		return start, end

	pos = _get_docstring_position_with_no_docstring(node=node, code=code, lines_shift=lines_shift)
	# return twice to match the format
	return pos, pos
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: def get_docstring_position_in_code(", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: \\tnode: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3: \\tcode: str,", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4: \\tlines_shift: int,", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5: ) -> Union[tuple[int, int], None]:", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6: \\tif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7: \\t\\treturn None", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8: \\tif (", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9: \\t\\tnode.body", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10: \\t\\tand isinstance(node.body[0], ast.Expr)", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11: \\t\\tand isinstance(node.body[0].value, (ast.Str, ast.Constant))", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12: \\t):", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13: \\t\\tdoc_node = node.body[0]", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14: \\t\\tlines = code.splitlines()", "title": "Line14"}, "line15": {"description": "Comment for the line n. 15: \\t\\tstart = sum(", "title": "Line15"}, "line16": {"description": "Comment for the line n. 16: \\t\\t\\tlen(line) + 1 for line in lines[: doc_node.lineno - 1 + lines_shift]", "title": "Line16"}, "line17": {"description": "Comment for the line n. 17: \\t\\t)", "title": "Line17"}, "line18": {"description": "Comment for the line n. 18: \\t\\tstart += doc_node.col_offset", "title": "Line18"}, "line19": {"default": "in this case docstring is single-line", "description": "Comment for the line n. 19: \\t\\tif doc_node.lineno == doc_node.end_lineno:", "title": "Line19"}, "line20": {"description": "Comment for the line n. 20: \\t\\t\\tline = lines[doc_node.lineno - 1 + lines_shift]", "title": "Line20"}, "line21": {"description": "Comment for the line n. 21: \\t\\t\\tquote_type = line[doc_node.col_offset : doc_node.col_offset + 3]", "title": "Line21"}, "line22": {"description": "Comment for the line n. 22: \\t\\t\\tif quote_type in (\'\\"\\"\\"\', \\"\'\'\'\\"):", "title": "Line22"}, "line23": {"description": "Comment for the line n. 23: \\t\\t\\t\\tend = line.find(quote_type, doc_node.col_offset + 3) + 3", "title": "Line23"}, "line24": {"description": "Comment for the line n. 24: \\t\\t\\telse:", "title": "Line24"}, "line25": {"description": "Comment for the line n. 25: \\t\\t\\t\\tquote_type = line[doc_node.col_offset]", "title": "Line25"}, "line26": {"description": "Comment for the line n. 26: \\t\\t\\t\\tend = line.find(quote_type, doc_node.col_offset + 1) + 1", "title": "Line26"}, "line27": {"description": "Comment for the line n. 27: \\t\\t\\tend = start + (end - doc_node.col_offset)", "title": "Line27"}, "line28": {"description": "Comment for the line n. 28: \\t\\telse:", "title": "Line28"}, "line29": {"description": "Comment for the line n. 29: \\t\\t\\tlast_line = lines[doc_node.end_lineno - 1 + lines_shift].rstrip()", "title": "Line29"}, "line30": {"description": "Comment for the line n. 30: \\t\\t\\tend = sum(", "title": "Line30"}, "line31": {"description": "Comment for the line n. 31: \\t\\t\\t\\tlen(line) + 1 for line in lines[: doc_node.end_lineno - 1 + lines_shift]", "title": "Line31"}, "line32": {"description": "Comment for the line n. 32: \\t\\t\\t)", "title": "Line32"}, "line33": {"description": "Comment for the line n. 33: \\t\\t\\tend += len(last_line)", "title": "Line33"}, "line34": {"description": "Comment for the line n. 34: \\t\\treturn start, end", "title": "Line34"}, "line35": {"description": "Comment for the line n. 35: \\tpos = _get_docstring_position_with_no_docstring(node=node, code=code, lines_shift=lines_shift)", "title": "Line35"}, "line36": {"default": "return twice to match the format", "description": "Comment for the line n. 36: \\treturn pos, pos", "title": "Line36"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "Check if the docstring already exists", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "Get the AST node of the docstring", "line14": "", "line15": "Calculate docstring's start position by:\\n1. Summing lengths of all previous lines (including newline chars)\\n2. Adding the column offset of the current line (number of preceding chars in the current line)", "line16": "", "line17": "", "line18": "", "line19": "Check if the docstring is single-line", "line20": "Get the line containing the docstring", "line21": "Determine the type of the quote used for the docstring", "line22": "Check if the docstring is enclosed in triple quotes", "line23": "Find the position of the closing triple quote", "line24": "Handle single quotes", "line25": "Determine the type of the quote used for the docstring", "line26": "Find the position of the closing single quote", "line27": "Calculate the end position of the docstring", "line28": "Handle multi-line docstrings", "line29": "Get the last line of the docstring", "line30": "Calculate the starting position of the last line of the docstring", "line31": "", "line32": "", "line33": "Add the length of the last line to the end position", "line34": "Return the start and end positions of the docstring", "line35": "Get the position where a docstring should be inserted if it doesn't exist", "line36": "Return the position twice since the start and end positions are the same"}
\'\'\'

Example 4:
\'\'\'
Input code:
```
from typing import Union, Optional, Any

class DynamicStateHandler:
    def __init__(self, payload: Any = None, mode: str = "auto", **kwargs):
        self._internal_state = {}
        self.payload = payload
        self._mode = mode
        self._temporary_flags = kwargs.get('flags', {})

    def process(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        if self._mode == "auto":
            return {**data, "processed": True}
        return None
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: from typing import Union, Optional, Any", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: class DynamicStateHandler:", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3:     def __init__(self, payload: Any = None, mode: str = \\"auto\\", **kwargs):", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4:         self._internal_state = {}", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5:         self.payload = payload", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6:         self._mode = mode", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7:         self._temporary_flags = kwargs.get(\'flags\', {})", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8:     def process(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9:         if self._mode == \\"auto\\":", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10:             return {**data, \\"processed\\": True}", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11:         return None", "title": "Line11"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": ""}
\'\'\'

Example 5:
\'\'\'
Input code:
```
import math
from typing import List, Optional, Tuple, Union
import torch
from pos_embed import _get_3d_sincos_pos_embed_np, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid


def get_timestep_embedding(
	timesteps: torch.Tensor,
	embedding_dim: int,
	flip_sin_to_cos: bool = False,
	downscale_freq_shift: float = 1,
	scale: float = 1,
	max_period: int = 10000,
) -> torch.Tensor:
	"""
	Creates sinusoidal timestep embeddings, matching the implementation in Denoising Diffusion Probabilistic Models.
	
	Args:
		timesteps (`torch.Tensor`):
			A 1-D Tensor of N indices, one per batch element. These may be fractional.
		embedding_dim (`int`):
			The dimension of the output embeddings.
		flip_sin_to_cos (`bool`, defaults to `False`):
			Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False).
		downscale_freq_shift (`float`, defaults to `1`):
			Controls the delta between frequencies between dimensions.
		scale (`float`, defaults to `1`):
			Scaling factor applied to the embeddings.
		max_period (`int`, defaults to `10000`):
			Controls the maximum frequency of the embeddings.
	
	Returns:
		`torch.Tensor`: An [N x dim] Tensor of positional embeddings.
	"""
	assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_3d_sincos_pos_embed(
	embed_dim: int,
	spatial_size: Union[int, Tuple[int, int]],
	temporal_size: int,
	spatial_interpolation_scale: float = 1.0,
	temporal_interpolation_scale: float = 1.0,
	device: Optional[torch.device] = None,
	output_type: str = "np",
) -> torch.Tensor:
	"""
	Creates 3D sinusoidal positional embeddings.
	
	Args:
		embed_dim (`int`):
			The embedding dimension of inputs. It must be divisible by 4.
		spatial_size (`int` or `Tuple[int, int]`):
			The spatial dimension of positional embeddings. If an integer is provided, the same size is applied to both
			spatial dimensions (height and width).
		temporal_size (`int`):
			The temporal dimension of positional embeddings (number of frames).
		spatial_interpolation_scale (`float`, defaults to `1.0`):
			Scale factor for spatial grid interpolation.
		temporal_interpolation_scale (`float`, defaults to `1.0`):
			Scale factor for temporal grid interpolation.
		device (`torch.device`, *optional*):
			Device to place the embeddings on.
		output_type (`str`, defaults to `'np'`):
			Type of the output embeddings. Can be `'np'` for NumPy arrays or `'pt'` for PyTorch tensors.
	
	Returns:
		`torch.Tensor`: The 3D sinusoidal positional embeddings of shape `[temporal_size, spatial_size[0] * spatial_size[1], embed_dim]`.
	"""
	if output_type == "np":
        return _get_3d_sincos_pos_embed_np(
            embed_dim=embed_dim,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
        )
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # spatoal
    grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, output_type="pt")

    # temporal
    grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type="pt")

    # concat
    pos_embed_spatial = pos_embed_spatial[None, :, :]
    pos_embed_spatial = pos_embed_spatial.repeat_interleave(temporal_size, dim=0)  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = pos_embed_temporal.repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )

    pos_embed = torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)
    return pos_embed
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: import math", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: from typing import List, Optional, Tuple, Union", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3: import torch", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4: from pos_embed import _get_3d_sincos_pos_embed_np, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5: def get_timestep_embedding(", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6: \\ttimesteps: torch.Tensor,", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7: \\tembedding_dim: int,", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8: \\tflip_sin_to_cos: bool = False,", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9: \\tdownscale_freq_shift: float = 1,", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10: \\tscale: float = 1,", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11: \\tmax_period: int = 10000,", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12: ) -> torch.Tensor:", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13: \\tassert len(timesteps.shape) == 1, \\"Timesteps should be a 1d-array\\"", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14:     half_dim = embedding_dim // 2", "title": "Line14"}, "line15": {"default": "Calculate the exponent for the sinusoidal embedding", "description": "Comment for the line n. 15:     exponent = -math.log(max_period) * torch.arange(", "title": "Line15"}, "line16": {"description": "Comment for the line n. 16:         start=0, end=half_dim, dtype=torch.float32, device=timesteps.device", "title": "Line16"}, "line17": {"description": "Comment for the line n. 17:     )", "title": "Line17"}, "line18": {"default": "Adjust the exponent based on the downscale frequency shift", "description": "Comment for the line n. 18:     exponent = exponent / (half_dim - downscale_freq_shift)", "title": "Line18"}, "line19": {"default": "Calculate the embedding using the exponent", "description": "Comment for the line n. 19:     emb = torch.exp(exponent)", "title": "Line19"}, "line20": {"default": "Multiply timesteps with the embedding", "description": "Comment for the line n. 20:     emb = timesteps[:, None].float() * emb[None, :]", "title": "Line20"}, "line21": {"default": "Scale the embedding", "description": "Comment for the line n. 21:     emb = scale * emb", "title": "Line21"}, "line22": {"default": "Concatenate sine and cosine of the embedding", "description": "Comment for the line n. 22:     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)", "title": "Line22"}, "line23": {"default": "Flip the order of sine and cosine if required", "description": "Comment for the line n. 23:     if flip_sin_to_cos:", "title": "Line23"}, "line24": {"description": "Comment for the line n. 24:         emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)", "title": "Line24"}, "line25": {"default": "If embedding dimension is odd, pad the embedding", "description": "Comment for the line n. 25:     if embedding_dim % 2 == 1:", "title": "Line25"}, "line26": {"description": "Comment for the line n. 26:         emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))", "title": "Line26"}, "line27": {"description": "Comment for the line n. 27:     return emb", "title": "Line27"}, "line28": {"description": "Comment for the line n. 28: def get_3d_sincos_pos_embed(", "title": "Line28"}, "line29": {"description": "Comment for the line n. 29: \\tembed_dim: int,", "title": "Line29"}, "line30": {"description": "Comment for the line n. 30: \\tspatial_size: Union[int, Tuple[int, int]],", "title": "Line30"}, "line31": {"description": "Comment for the line n. 31: \\ttemporal_size: int,", "title": "Line31"}, "line32": {"description": "Comment for the line n. 32: \\tspatial_interpolation_scale: float = 1.0,", "title": "Line32"}, "line33": {"description": "Comment for the line n. 33: \\ttemporal_interpolation_scale: float = 1.0,", "title": "Line33"}, "line34": {"description": "Comment for the line n. 34: \\tdevice: Optional[torch.device] = None,", "title": "Line34"}, "line35": {"description": "Comment for the line n. 35: \\toutput_type: str = \\"np\\",", "title": "Line35"}, "line36": {"description": "Comment for the line n. 36: ) -> torch.Tensor:", "title": "Line36"}, "line37": {"description": "Comment for the line n. 37: \\tif output_type == \\"np\\":", "title": "Line37"}, "line38": {"description": "Comment for the line n. 38:         return _get_3d_sincos_pos_embed_np(", "title": "Line38"}, "line39": {"description": "Comment for the line n. 39:             embed_dim=embed_dim,", "title": "Line39"}, "line40": {"description": "Comment for the line n. 40:             spatial_size=spatial_size,", "title": "Line40"}, "line41": {"description": "Comment for the line n. 41:             temporal_size=temporal_size,", "title": "Line41"}, "line42": {"description": "Comment for the line n. 42:             spatial_interpolation_scale=spatial_interpolation_scale,", "title": "Line42"}, "line43": {"description": "Comment for the line n. 43:             temporal_interpolation_scale=temporal_interpolation_scale,", "title": "Line43"}, "line44": {"description": "Comment for the line n. 44:         )", "title": "Line44"}, "line45": {"description": "Comment for the line n. 45:     if embed_dim % 4 != 0:", "title": "Line45"}, "line46": {"description": "Comment for the line n. 46:         raise ValueError(\\"`embed_dim` must be divisible by 4\\")", "title": "Line46"}, "line47": {"description": "Comment for the line n. 47:     if isinstance(spatial_size, int):", "title": "Line47"}, "line48": {"description": "Comment for the line n. 48:         spatial_size = (spatial_size, spatial_size)", "title": "Line48"}, "line49": {"description": "Comment for the line n. 49:     embed_dim_spatial = 3 * embed_dim // 4", "title": "Line49"}, "line50": {"description": "Comment for the line n. 50:     embed_dim_temporal = embed_dim // 4", "title": "Line50"}, "line51": {"default": "## 1. Spacial\\\\nCreate a grid for the height dimension", "description": "Comment for the line n. 51:     grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32) / spatial_interpolation_scale", "title": "Line51"}, "line52": {"default": "Create a grid for the width dimension", "description": "Comment for the line n. 52:     grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32) / spatial_interpolation_scale", "title": "Line52"}, "line53": {"default": "Create a meshgrid for spatial dimensions. Here x goes first.", "description": "Comment for the line n. 53:     grid = torch.meshgrid(grid_w, grid_h, indexing=\\"xy\\")", "title": "Line53"}, "line54": {"description": "Comment for the line n. 54:     grid = torch.stack(grid, dim=0)", "title": "Line54"}, "line55": {"description": "Comment for the line n. 55:     grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])", "title": "Line55"}, "line56": {"default": "Get the spatial positional embedding", "description": "Comment for the line n. 56:     pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, output_type=\\"pt\\")", "title": "Line56"}, "line57": {"default": "## 2. Temporal\\\\nCreate a grid for the temporal dimension", "description": "Comment for the line n. 57:     grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32) / temporal_interpolation_scale", "title": "Line57"}, "line58": {"default": "Get the temporal positional embedding", "description": "Comment for the line n. 58:     pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type=\\"pt\\")", "title": "Line58"}, "line59": {"default": "## 3. Concatenate spacial and temporal embeddings\\\\nAdd a batch dimension to the spatial embedding", "description": "Comment for the line n. 59:     pos_embed_spatial = pos_embed_spatial[None, :, :]", "title": "Line59"}, "line60": {"default": "Repeat the spatial embedding along the temporal dimension\\\\n[T, H*W, D // 4 * 3]", "description": "Comment for the line n. 60:     pos_embed_spatial = pos_embed_spatial.repeat_interleave(temporal_size, dim=0)", "title": "Line60"}, "line61": {"default": "Add a batch dimension to the temporal embedding", "description": "Comment for the line n. 61:     pos_embed_temporal = pos_embed_temporal[:, None, :]", "title": "Line61"}, "line62": {"default": "Repeat the temporal embedding along the spatial dimension\\\\n[T, H*W, D // 4]", "description": "Comment for the line n. 62:     pos_embed_temporal = pos_embed_temporal.repeat_interleave(", "title": "Line62"}, "line63": {"description": "Comment for the line n. 63:         spatial_size[0] * spatial_size[1], dim=1", "title": "Line63"}, "line64": {"description": "Comment for the line n. 64:     )", "title": "Line64"}, "line65": {"default": "[T, H*W, D]", "description": "Comment for the line n. 65:     pos_embed = torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)", "title": "Line65"}, "line66": {"description": "Comment for the line n. 66:     return pos_embed", "title": "Line66"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": "", "line15": "Calculate the exponent for the sinusoidal embedding", "line16": "", "line17": "", "line18": "Adjust the exponent based on the downscale frequency shift", "line19": "Calculate the embedding using the exponent", "line20": "Multiply timesteps with the embedding", "line21": "Scale the embedding", "line22": "Concatenate sine and cosine of the embedding", "line23": "Flip the order of sine and cosine if required", "line24": "", "line25": "If embedding dimension is odd, pad the embedding", "line26": "", "line27": "", "line28": "", "line29": "", "line30": "", "line31": "", "line32": "", "line33": "", "line34": "", "line35": "", "line36": "", "line37": "", "line38": "", "line39": "", "line40": "", "line41": "", "line42": "", "line43": "", "line44": "", "line45": "", "line46": "", "line47": "", "line48": "", "line49": "", "line50": "", "line51": "## 1. Spacial\\nCreate a grid for the height dimension", "line52": "Create a grid for the width dimension", "line53": "Create a meshgrid for spatial dimensions", "line54": "", "line55": "", "line56": "Get the spatial positional embedding", "line57": "## 2. Temporal\\nCreate a grid for the temporal dimension", "line58": "Get the temporal positional embedding", "line59": "## 3. Concatenate spacial and temporal embeddings\\nAdd a batch dimension to the spatial embedding", "line60": "Repeat the spatial embedding along the temporal dimension\\n[T, H*W, D // 4 * 3]", "line61": "Add a batch dimension to the temporal embedding", "line62": "Repeat the temporal embedding along the spatial dimension\\n[T, H*W, D // 4]", "line63": "", "line64": "", "line65": "[T, H*W, D]", "line66": ""}
\'\'\'

Example 6:
\'\'\'
Input code:
```
import time

async def fetch_user_profile(user_id: int) -> dict:
	"""
   	Asynchronously fetches a user profile from an API and returns it as a dictionary.

   	Args:
   		user_id (`int`):
   			The ID of the user whose profile is to be fetched.

   	Returns:
   		`dict`: A dictionary containing the user's profile information, including:
   			- `id` (`int`): The user's ID.
   			- `name` (`str`): The user's name.
   			- `email` (`str`): The user's email address.
   			- `created_at` (`str`): The timestamp of when the user profile was created.
   			- `request_time` (`float`): The time taken to fetch the user profile in seconds.
	"""
	start_time = time.time()
	async with aiohttp.ClientSession() as session:
		url = f"https://api.example.com/users/{user_id}"
		response = await session.get(url)
		data = await response.json()

	profile = {
		"id": user_id,
		"name": data["name"],
		"email": data["email"],
		"created_at": data["created_at"],
	}
	profile["request_time"] = time.time() - start_time
	return profile
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: import time", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: async def fetch_user_profile(user_id: int) -> dict:", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3: \\tstart_time = time.time()", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4: \\tasync with aiohttp.ClientSession() as session:", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5: \\t\\turl = f\\"https://api.example.com/users/{user_id}\\"", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6: \\t\\tresponse = await session.get(url)", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7: \\t\\tdata = await response.json()", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8: \\tprofile = {", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9: \\t\\t\\"id\\": user_id,", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10: \\t\\t\\"name\\": data[\\"name\\"],", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11: \\t\\t\\"email\\": data[\\"email\\"],", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12: \\t\\t\\"created_at\\": data[\\"created_at\\"],", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13: \\t}", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14: \\tprofile[\\"request_time\\"] = time.time() - start_time", "title": "Line14"}, "line15": {"description": "Comment for the line n. 15: \\treturn profile", "title": "Line15"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "Create an asynchronous HTTP session", "line5": "", "line6": "Make a GET request to the API to fetch the user profile", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": "", "line15": ""}
\'\'\'

Example 7:
\'\'\'
Input code:
```
import ast
from openai import Client
from .components import add_docstrings, annotate_arguments, potentially_add_class_to_typing_import
from .utils.utils import _get_nodes_dict_with_functions_classes_methods

def _check_no_duplicating_methods(nodes: list[ast.AST]) -> None:
	"""
   	Checks for duplicate function and class method names within a list of AST nodes. Raises a `ValueError` if any duplicates are found.

   	Args:
   		nodes (`list[ast.AST]`):
   			A list of AST nodes to check for duplicate function and class method names.

   	Returns:
   		None

   	Raises:
   		ValueError: If any function or class method names are duplicated.
	"""
	functions_names = []
	classes_names = []
	for node in nodes:
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			functions_names.append(node.name)
		elif isinstance(node, ast.ClassDef):
			classes_names.append(node.name)
			class_methods_names = []
			for subnode in node.body:
				if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
					class_methods_names.append(subnode.name)
			#methods
			if len(set(class_methods_names)) != len(class_methods_names):
				for method in class_methods_names:
					if class_methods_names.count(method) != 1:
						raise ValueError(
							f"Method {method} is duplicated for class {node.name}."
							+ "Please fix this before running annotation."
						)

	#functions
	if len(set(functions_names)) != len(functions_names):
		for func in functions_names:
			if functions_names.count(func) != 1:
				raise ValueError(
					f"Function {func} is duplicated."
					+ "Please fix this before running annotation."
				)
    #classes
	if len(set(classes_names)) != len(classes_names):
		for class_ in classes_names:
			if classes_names.count(class_) != 1:
				raise ValueError(
					f"Class {class_} is duplicated."
					+ "Please fix this before running annotation."
				)

def document_python_code(
	code: str,
	client: Client,
	modify_existing_annotation: bool = False,
	model_checkpoint: str = "Qwen/Qwen2.5-Coder-32B-Instruct-fast",
) -> Generator[str, None, None]:
	"""
	Generates and adds docstrings to Python code using an OpenAI client. The function processes the code to annotate arguments and add docstrings to functions, classes, and methods. It also ensures that necessary typing imports are included.

	Args:
		code (`str`):
			The Python code as a string to which docstrings and argument annotations will be added.
		client (`Client`):
			An instance of the OpenAI client used to generate docstrings and annotations.
		modify_existing_annotation (`bool`, *optional*, defaults to `False`):
			If `True`, existing annotations and docstrings will be modified. If `False`, existing annotations and docstrings will be preserved unless they are empty.
		model_checkpoint (`str`, *optional*, defaults to `'Qwen/Qwen2.5-Coder-32B-Instruct-fast'`):
			The model checkpoint to use for generating docstrings and annotations.

	Yields:
		`str`: The updated code with added or modified docstrings and argument annotations.

	Raises:
		`SyntaxError`: If the generated code is not valid Python syntax.
	"""
	code = rf"{code}" # read it raw to avoid ast errors
	required_typing_imports = set()
	body = ast.parse(code).body

	_check_no_duplicating_classes_methods_functions(body)
	target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(body)

	output = None
	for output in annotate_arguments(
		target_nodes_dict=target_nodes_dict,
		code=code,
		client=client,
		modify_existing_annotation=modify_existing_annotation,
		model_checkpoint=model_checkpoint,
	):
		if isinstance(output, str):
			yield output
		code = output
	if isinstance(output, tuple):
		code, func_required_typing_imports = output
		required_typing_imports = required_typing_imports.union(
			func_required_typing_imports
		)

	code = code.replace(" " * 4, "	")
	tree = ast.parse(code)
	target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(tree.body)

	for output in add_docstrings(
		target_nodes_dict=target_nodes_dict,
		code=code,
		client=client,
		modify_existing_annotation=modify_existing_annotation,
		model_checkpoint=model_checkpoint,
	):
		yield output
		code = output
	# Add classes from typing that were not imported
	for typing_class in required_typing_imports:
		code = potentially_add_class_to_typing_import(code, tree, typing_class)
		yield code
	code = code.replace(" " * 4, "	")
	ast.parse(code) # to check it is valid
	yield code
```

Json schema:
```
{"line1": {"description": "Comment for the line n. 1: import ast", "title": "Line1"}, "line2": {"description": "Comment for the line n. 2: from openai import Client", "title": "Line2"}, "line3": {"description": "Comment for the line n. 3: from .components import add_docstrings, annotate_arguments, potentially_add_class_to_typing_import", "title": "Line3"}, "line4": {"description": "Comment for the line n. 4: from .utils.utils import _get_nodes_dict_with_functions_classes_methods", "title": "Line4"}, "line5": {"description": "Comment for the line n. 5: def _check_no_duplicating_methods(nodes: list[ast.AST]) -> None:", "title": "Line5"}, "line6": {"description": "Comment for the line n. 6: \\tfunctions_names = []", "title": "Line6"}, "line7": {"description": "Comment for the line n. 7: \\tclasses_names = []", "title": "Line7"}, "line8": {"description": "Comment for the line n. 8: \\tfor node in nodes:", "title": "Line8"}, "line9": {"description": "Comment for the line n. 9: \\t\\tif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):", "title": "Line9"}, "line10": {"description": "Comment for the line n. 10: \\t\\t\\tfunctions_names.append(node.name)", "title": "Line10"}, "line11": {"description": "Comment for the line n. 11: \\t\\telif isinstance(node, ast.ClassDef):", "title": "Line11"}, "line12": {"description": "Comment for the line n. 12: \\t\\t\\tclasses_names.append(node.name)", "title": "Line12"}, "line13": {"description": "Comment for the line n. 13: \\t\\t\\tclass_methods_names = []", "title": "Line13"}, "line14": {"description": "Comment for the line n. 14: \\t\\t\\tfor subnode in node.body:", "title": "Line14"}, "line15": {"description": "Comment for the line n. 15: \\t\\t\\t\\tif isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):", "title": "Line15"}, "line16": {"description": "Comment for the line n. 16: \\t\\t\\t\\t\\tclass_methods_names.append(subnode.name)", "title": "Line16"}, "line17": {"default": "methods", "description": "Comment for the line n. 17: \\t\\t\\tif len(set(class_methods_names)) != len(class_methods_names):", "title": "Line17"}, "line18": {"description": "Comment for the line n. 18: \\t\\t\\t\\tfor method in class_methods_names:", "title": "Line18"}, "line19": {"description": "Comment for the line n. 19: \\t\\t\\t\\t\\tif class_methods_names.count(method) != 1:", "title": "Line19"}, "line20": {"description": "Comment for the line n. 20: \\t\\t\\t\\t\\t\\traise ValueError(", "title": "Line20"}, "line21": {"description": "Comment for the line n. 21: \\t\\t\\t\\t\\t\\t\\tf\\"Method {method} is duplicated for class {node.name}.\\"", "title": "Line21"}, "line22": {"description": "Comment for the line n. 22: \\t\\t\\t\\t\\t\\t\\t+ \\"Please fix this before running annotation.\\"", "title": "Line22"}, "line23": {"description": "Comment for the line n. 23: \\t\\t\\t\\t\\t\\t)", "title": "Line23"}, "line24": {"default": "functions", "description": "Comment for the line n. 24: \\tif len(set(functions_names)) != len(functions_names):", "title": "Line24"}, "line25": {"description": "Comment for the line n. 25: \\t\\tfor func in functions_names:", "title": "Line25"}, "line26": {"description": "Comment for the line n. 26: \\t\\t\\tif functions_names.count(func) != 1:", "title": "Line26"}, "line27": {"description": "Comment for the line n. 27: \\t\\t\\t\\traise ValueError(", "title": "Line27"}, "line28": {"description": "Comment for the line n. 28: \\t\\t\\t\\t\\tf\\"Function {func} is duplicated.\\"", "title": "Line28"}, "line29": {"description": "Comment for the line n. 29: \\t\\t\\t\\t\\t+ \\"Please fix this before running annotation.\\"", "title": "Line29"}, "line30": {"description": "Comment for the line n. 30: \\t\\t\\t\\t)", "title": "Line30"}, "line31": {"default": "classes", "description": "Comment for the line n. 31: \\tif len(set(classes_names)) != len(classes_names):", "title": "Line31"}, "line32": {"description": "Comment for the line n. 32: \\t\\tfor class_ in classes_names:", "title": "Line32"}, "line33": {"description": "Comment for the line n. 33: \\t\\t\\tif classes_names.count(class_) != 1:", "title": "Line33"}, "line34": {"description": "Comment for the line n. 34: \\t\\t\\t\\traise ValueError(", "title": "Line34"}, "line35": {"description": "Comment for the line n. 35: \\t\\t\\t\\t\\tf\\"Class {class_} is duplicated.\\"", "title": "Line35"}, "line36": {"description": "Comment for the line n. 36: \\t\\t\\t\\t\\t+ \\"Please fix this before running annotation.\\"", "title": "Line36"}, "line37": {"description": "Comment for the line n. 37: \\t\\t\\t\\t)", "title": "Line37"}, "line38": {"description": "Comment for the line n. 38: def document_python_code(", "title": "Line38"}, "line39": {"description": "Comment for the line n. 39: \\tcode: str,", "title": "Line39"}, "line40": {"description": "Comment for the line n. 40: \\tclient: Client,", "title": "Line40"}, "line41": {"description": "Comment for the line n. 41: \\tmodify_existing_annotation: bool = False,", "title": "Line41"}, "line42": {"description": "Comment for the line n. 42: \\tmodel_checkpoint: str = \\"Qwen/Qwen2.5-Coder-32B-Instruct-fast\\",", "title": "Line42"}, "line43": {"description": "Comment for the line n. 43: ) -> Generator[str, None, None]:", "title": "Line43"}, "line44": {"default": "read it raw to avoid ast errors", "description": "Comment for the line n. 44: \\tcode = rf\\"{code}\\" # read it raw to avoid ast errors", "title": "Line44"}, "line45": {"description": "Comment for the line n. 45: \\trequired_typing_imports = set()", "title": "Line45"}, "line46": {"description": "Comment for the line n. 46: \\tbody = ast.parse(code).body", "title": "Line46"}, "line47": {"description": "Comment for the line n. 47: \\t_check_no_duplicating_classes_methods_functions(body)", "title": "Line47"}, "line48": {"description": "Comment for the line n. 48: \\ttarget_nodes_dict = _get_nodes_dict_with_functions_classes_methods(body)", "title": "Line48"}, "line49": {"description": "Comment for the line n. 49: \\toutput = None", "title": "Line49"}, "line50": {"description": "Comment for the line n. 50: \\tfor output in annotate_arguments(", "title": "Line50"}, "line51": {"description": "Comment for the line n. 51: \\t\\ttarget_nodes_dict=target_nodes_dict,", "title": "Line51"}, "line52": {"description": "Comment for the line n. 52: \\t\\tcode=code,", "title": "Line52"}, "line53": {"description": "Comment for the line n. 53: \\t\\tclient=client,", "title": "Line53"}, "line54": {"description": "Comment for the line n. 54: \\t\\tmodify_existing_annotation=modify_existing_annotation,", "title": "Line54"}, "line55": {"description": "Comment for the line n. 55: \\t\\tmodel_checkpoint=model_checkpoint,", "title": "Line55"}, "line56": {"description": "Comment for the line n. 56: \\t):", "title": "Line56"}, "line57": {"description": "Comment for the line n. 57: \\t\\tif isinstance(output, str):", "title": "Line57"}, "line58": {"description": "Comment for the line n. 58: \\t\\t\\tyield output", "title": "Line58"}, "line59": {"description": "Comment for the line n. 59: \\t\\tcode = output", "title": "Line59"}, "line60": {"description": "Comment for the line n. 60: \\tif isinstance(output, tuple):", "title": "Line60"}, "line61": {"description": "Comment for the line n. 61: \\t\\tcode, func_required_typing_imports = output", "title": "Line61"}, "line62": {"description": "Comment for the line n. 62: \\t\\trequired_typing_imports = required_typing_imports.union(", "title": "Line62"}, "line63": {"description": "Comment for the line n. 63: \\t\\t\\tfunc_required_typing_imports", "title": "Line63"}, "line64": {"description": "Comment for the line n. 64: \\t\\t)", "title": "Line64"}, "line65": {"description": "Comment for the line n. 65: \\tcode = code.replace(\\" \\" * 4, \\"\\t\\")", "title": "Line65"}, "line66": {"description": "Comment for the line n. 66: \\ttree = ast.parse(code)", "title": "Line66"}, "line67": {"description": "Comment for the line n. 67: \\ttarget_nodes_dict = _get_nodes_dict_with_functions_classes_methods(tree.body)", "title": "Line67"}, "line68": {"description": "Comment for the line n. 68: \\tfor output in add_docstrings(", "title": "Line68"}, "line69": {"description": "Comment for the line n. 69: \\t\\ttarget_nodes_dict=target_nodes_dict,", "title": "Line69"}, "line70": {"description": "Comment for the line n. 70: \\t\\tcode=code,", "title": "Line70"}, "line71": {"description": "Comment for the line n. 71: \\t\\tclient=client,", "title": "Line71"}, "line72": {"description": "Comment for the line n. 72: \\t\\tmodify_existing_annotation=modify_existing_annotation,", "title": "Line72"}, "line73": {"description": "Comment for the line n. 73: \\t\\tmodel_checkpoint=model_checkpoint,", "title": "Line73"}, "line74": {"description": "Comment for the line n. 74: \\t):", "title": "Line74"}, "line75": {"description": "Comment for the line n. 75: \\t\\tyield output", "title": "Line75"}, "line76": {"description": "Comment for the line n. 76: \\t\\tcode = output", "title": "Line76"}, "line77": {"default": "Add classes from typing that were not imported", "description": "Comment for the line n. 77: \\tfor typing_class in required_typing_imports:", "title": "Line77"}, "line78": {"description": "Comment for the line n. 78: \\t\\tcode = potentially_add_class_to_typing_import(code, tree, typing_class)", "title": "Line78"}, "line79": {"description": "Comment for the line n. 79: \\t\\tyield code", "title": "Line79"}, "line80": {"description": "Comment for the line n. 80: \\tcode = code.replace(\\" \\" * 4, \\"\\t\\")", "title": "Line80"}, "line81": {"default": "to check it is valid", "description": "Comment for the line n. 81: \\tast.parse(code) # to check it is valid", "title": "Line81"}, "line82": {"description": "Comment for the line n. 82: \\tyield code", "title": "Line82"}}
```

Assistant:
{"line1": "", "line2": "", "line3": "", "line4": "", "line5": "", "line6": "", "line7": "", "line8": "", "line9": "", "line10": "", "line11": "", "line12": "", "line13": "", "line14": "", "line15": "", "line16": "", "line17": "Check if there are duplicate method names in the current class", "line18": "", "line19": "Check if the method name appears more than once", "line20": "", "line21": "", "line22": "", "line23": "", "line24": "Check if there are duplicate function names", "line25": "", "line26": "Check if the function name appears more than once", "line27": "", "line28": "", "line29": "", "line30": "", "line31": "Check if there are duplicate class names", "line32": "", "line33": "Check if the class name appears more than once", "line34": "", "line35": "", "line36": "", "line37": "", "line38": "", "line39": "", "line40": "", "line41": "", "line42": "", "line43": "", "line44": "Read the code as a raw string to avoid AST parsing errors", "line45": "", "line46": "Parse the code into an AST", "line47": "", "line48": "Get a dictionary of nodes that need to be annotated or documented", "line49": "Annotate the arguments of functions, classes, and methods", "line50": "", "line51": "", "line52": "", "line53": "", "line54": "", "line55": "", "line56": "", "line57": "", "line58": "", "line59": "", "line60": "", "line61": "Get the required imports from the `typing` package that will need to be added in the end", "line62": "", "line63": "", "line64": "", "line65": "Replace spaces with tabs for consistent formatting", "line66": "Lines may have changed, so it's easier to rerun `ast.parse` which takes < 1ms than track\\nthis throughout the code", "line67": "Get dictionary with target nodes with the updated AST code", "line68": "Add docstrings to functions, classes, and methods", "line69": "", "line70": "", "line71": "", "line72": "", "line73": "", "line74": "", "line75": "", "line76": "", "line77": "If there are classes from the `typing` package that were used for annotation but not imported,\\nadd them to the imports", "line78": "", "line79": "", "line80": "Replace spaces with tabs for consistent formatting", "line81": "Make sure the generated code has valid Python syntax", "line82": ""}
\'\'\'
'''.strip()

SYSTEM_PROMPT_FIX_ANNOTATION = """
Act as an experienced Python specialist.

You will be provided with a "broken" argument annotation in Python, which was written by our junior developer. \
Kindly fix this annotation. \
Output only the fixed annotation and nothing more. \
Your output will be directly inserted into the code and parsed with `ast`, \
so don't output anything else besides tha fixed annotation.

Examples for few-shot learning:
Example 1:
'''
User:
Union[str, List[Union[int, str, List[Any]]]

Assistant:
Union[str, List[Union[int, str, List[Any]]]]
'''

Example 2:
'''
User:
tuple[ type[BaseModel], dict[ str, tuple[ Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef], dict[ str, tuple[ Union[ast.arg, ast.AnnAssign, ast.Assign, ast.Constant, None], str | None, ] | tuple[Union[ast.arg, ast.AnnAssign, ast.Assign], str | None, str]

Assistant:
tuple[ type[BaseModel], dict[ str, tuple[ Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef], dict[ str, tuple[ Union[ast.arg, ast.AnnAssign, ast.Assign, ast.Constant, None], str | None, ] | tuple[Union[ast.arg, ast.AnnAssign, ast.Assign], str | None, str]]]]]
'''

Example 3:
'''
User:
integer

Assistant:
int
'''
""".strip()
