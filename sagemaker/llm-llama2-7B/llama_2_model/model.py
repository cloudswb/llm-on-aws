from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
##
from transformers import LlamaTokenizer, LlamaForCausalLM
import json

model = None
tokenizer = None


def get_model(properties):
    model_name = properties["model_id"]
    tensor_parallel_degree = properties["tensor_parallel_degree"]
    max_tokens = int(properties.get("max_tokens", "1024"))
    dtype = torch.float16

    model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=dtype, device_map='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# system_prompt = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#             If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
#             """

system_prompt = ""


def get_prompt(message: str, chat_history: list[tuple[str, str]]) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("ask", input_map)
        
        # if data.startswith("[INST]"):
        #     data = data
        # else:
        #     data = get_prompt(data, [])
        
        parameters = input_map.pop("parameters", {})
        outputs = Output()

        enable_streaming = inputs.get_properties().get("enable_streaming",
                            "false").lower() == "true"
        if enable_streaming:
            stream_generator = StreamingUtils.get_stream_generator(
                "DeepSpeed")
            outputs.add_stream_content(
                stream_generator(model, tokenizer, data,
                                 **parameters))
            return outputs

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        input_tokens = tokenizer(data, padding=True,
                                return_tensors="pt").to(
                                torch.cuda.current_device())
        # with torch.no_grad():
        #     output_tokens = model.generate(input_tokens.input_ids, **parameters)
        # output_tokens = model.generate(input_tokens.input_ids, **parameters)
        
        # input_tokens = tokenizer(data, return_tensors='pt')
        output_tokens = model.generate(input_tokens.input_ids, **parameters)

        
        # print("output_tokens", json.dumps(output_tokens))
        generated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        answer = [{"generated_text": s} for s in generated_text]
        answer_text = ''
        for item in answer:  
            if '[/INST]' in item['generated_text']:  
                answer_text += item['generated_text'].split('[/INST]')[1]
        
        outputs.add_as_json({"answer": answer_text})
        return outputs
    
        # outputs.add_as_json([{"generated_text": s} for s in generated_text])
        # return outputs
    except Exception as e:
        logging.exception("Huggingface inference failed")
        # error handling
        outputs = Output().error(str(e))


def handle(inputs: Input) -> None:
    global model, tokenizer
    if not model:
        model, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return inference(inputs)