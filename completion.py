import os
import json
import time
import yaml
import random
import shortuuid

import requests
from typing import Optional
import boto3

from glob import glob
from tqdm import tqdm

# API setting constants
API_MAX_RETRY = 50
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = None

registered_api_completion = {}
registered_engine_completion = {}


def register_api(api_type):
    def decorator(func):
        registered_api_completion[api_type] = func
        return func

    return decorator


def register_engine(engine_type):
    def decorator(func):
        registered_engine_completion[engine_type] = func
        return func

    return decorator


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    if not os.path.exists(answer_dir):
        return {}
    
    filenames = []
    for folder in os.listdir(answer_dir):
        if not os.path.isdir(os.path.join(answer_dir, folder)):
            continue
        if not os.path.exists(os.path.join(answer_dir, folder, "generation.jsonl")):
            print(f"WARNING: {folder} does not have generation.jsonl, skip it.")
            continue
        filenames.append(os.path.join(answer_dir, folder, "generation.jsonl"))

    filenames.sort()
    model_answers = {}
    for filename in filenames:
        # Use parent directory name as model name
        model_name = os.path.basename(os.path.dirname(filename))
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["uid"]] = line
        model_answers[model_name] = answer
    return model_answers


def load_model_judgements(answer_dir: str):
    """Load model judgements.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["uid"]] = line
        model_answers[model_name] = answer

    return model_answers


def load_model_answers_and_execution_results(data_dir: str):
    """Load model answers and execution results.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = []
    for folder in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, folder)):
            continue
        if not os.path.exists(os.path.join(data_dir, folder, "execution_results.jsonl")):
            print(f"WARNING: {folder} does not have execution_results.jsonl, skip it.")
            continue
        filenames.append(os.path.join(data_dir, folder, "execution_results.jsonl"))

    filenames.sort()
    model_answers = {}

    for filename in filenames:
        # Use parent directory name as model name
        model_name = os.path.basename(os.path.dirname(filename))
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["uid"]] = line
        model_answers[model_name] = answer

    return model_answers



def load_id_to_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                
                if line["uid"] in model_answers:
                    model_answers[line["uid"]][model_name] = line
                else:
                    model_answers[line["uid"]] = {model_name: line}
                
    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


@register_api("openai")
def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
        
    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = {
                "answer": completion.choices[0].message.content
            }
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print("=== DEBUG: OpenAI BadRequestError ===")
            print("Error type:", type(e))
            print("Error message:", str(e))
            print("=== Analyzing messages for image issues ===")
            for i, msg in enumerate(messages):
                print(f"Message {i} role: {msg.get('role', 'unknown')}")
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        for j, item in enumerate(content):
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if url.startswith("data:image/png;base64,"):
                                    base64_part = url[22:]  # Remove "data:image/png;base64," prefix
                                    print(f"  Image {j}: base64 length = {len(base64_part)}")
                                    if len(base64_part) < 50:
                                        print(f"  *** ISSUE: Image {j} has very short/empty base64: '{url}'")
                                elif url.startswith("data:image/"):
                                    print(f"  Image {j}: Non-PNG data URL: {url[:50]}...")
                                else:
                                    print(f"  Image {j}: Unexpected URL format: {url[:50]}...")
                            elif isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                print(f"  Text {j}: {len(text_content)} chars")
                            else:
                                print(f"  Content {j}: {type(item)} - {str(item)[:50]}...")
                    else:
                        print(f"  Content: {type(content)} - {str(content)[:100]}...")
            print("=== End debug info ===")
            break
        except KeyError:
            print(type(e), e)
            break
    
    return output


@register_api("openai_streaming")
def chat_completion_openai_streaming(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    """Streaming version of OpenAI completion that yields tokens as they arrive"""
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
        
    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Error in streaming completion: {e}")
        yield f"Error: {str(e)}"


@register_api("openai_thinking")
def chat_completion_openai_thinking(model, messages, api_dict=None, **kwargs):
    import openai
    
    if api_dict:
        client = openai.OpenAI(
            api_key=api_dict["api_key"],
            base_url=api_dict["api_base"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else 'medium',
                
            )
            output = {
                "answer": completion.choices[0].message.content
            }
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print("=== DEBUG: OpenAI BadRequestError ===")
            print("Error type:", type(e))
            print("Error message:", str(e))
            print("=== Analyzing messages for image issues ===")
            for i, msg in enumerate(messages):
                print(f"Message {i} role: {msg.get('role', 'unknown')}")
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        for j, item in enumerate(content):
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if url.startswith("data:image/png;base64,"):
                                    base64_part = url[22:]  # Remove "data:image/png;base64," prefix
                                    print(f"  Image {j}: base64 length = {len(base64_part)}")
                                    if len(base64_part) < 50:
                                        print(f"  *** ISSUE: Image {j} has very short/empty base64: '{url}'")
                                elif url.startswith("data:image/"):
                                    print(f"  Image {j}: Non-PNG data URL: {url[:50]}...")
                                else:
                                    print(f"  Image {j}: Unexpected URL format: {url[:50]}...")
                            elif isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                print(f"  Text {j}: {len(text_content)} chars")
                            else:
                                print(f"  Content {j}: {type(item)} - {str(item)[:50]}...")
                    else:
                        print(f"  Content: {type(content)} - {str(content)[:100]}...")
            print("=== End debug info ===")
            break
        except KeyError:
            print(type(e), e)
            break
    
    return output


@register_api("deepseek_reasoner")
def chat_completion_deepseek_reasoner(messages, api_dict, **kwargs):
    import urllib.request

    chat_endpoint_headers = {
        "User-Agent": "curl/8.7.1",
        "Authorization": "Bearer {}".format(api_dict['api_key']),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    chat_endpoint_url = "https://api.deepseek.com/chat/completions"
    
    req_body = {
        "messages": messages,
        "model": "deepseek-reasoner",
        "stream": False,
    }
    req_data = json.dumps(req_body).encode("utf-8")
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            req = urllib.request.Request(
                chat_endpoint_url,
                headers = chat_endpoint_headers.copy(),
                data = req_data,
            )
            
            with urllib.request.urlopen(req) as res:
                res_data = res.read()
            res_body = json.loads(res_data.decode("utf-8"))
            
            output = {
                "thought": res_body["choices"][0]["message"]["reasoning_content"],
                "answer": res_body["choices"][0]["message"]["content"],
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output


@register_api("deepseek")
def chat_completion_deepseek(messages, max_tokens, api_dict, **kwargs):
    import urllib.request

    chat_endpoint_headers = {
        "User-Agent": "curl/8.7.1",
        "Authorization": "Bearer {}".format(api_dict['api_key']),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    chat_endpoint_url = "https://api.deepseek.com/chat/completions"
    
    req_body = {
        "messages": messages,
        "model": "deepseek-chat",
        "stream": False,
        "max_tokens": max_tokens,
    }
    req_data = json.dumps(req_body).encode("utf-8")
    
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            req = urllib.request.Request(
                chat_endpoint_url,
                headers = chat_endpoint_headers.copy(),
                data = req_data,
            )
            
            with urllib.request.urlopen(req) as res:
                res_data = res.read()
            res_body = json.loads(res_data.decode("utf-8"))
            
            output = {
                "answer": res_body["choices"][0]["message"]["content"],
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output


@register_api("anthropic")
def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = {
                "answer": response.content[0].text
            }
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


@register_api("anthropic_thinking")
def chat_completion_anthropic_thinking(model, messages, max_tokens, budget_tokens, **kwargs):
    import anthropic

    client = anthropic.Anthropic(
        timeout=1200,
    )
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                },
                messages=messages,
            )
            output = {
                "thought": response.content[0].thinking,
                "answer": response.content[1].text,
            }
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


@register_api("mistral")
def chat_completion_mistral(model, messages, temperature, max_tokens, **kwargs):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = {
                "answer": chat_response.choices[0].message.content
            }
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


@register_api("xai")
def chat_completion_xai(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import xai_sdk

    client = xai_sdk.Client(api_key=api_dict['api_key'], api_host=api_dict['api_base']).compat
    output = API_ERROR_OUTPUT
    
    for _ in range(API_MAX_RETRY):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            output_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    output_text += chunk.choices[0].delta.content
            
            output = {
                "answer": output_text
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    
    return output


@register_api("litellm")
def chat_completion_litellm(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    import litellm

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = {
                "answer": response.choices[0].message.content
            }
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    
    return output


@register_api("litellm_streaming")
def chat_completion_litellm_streaming(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    """Streaming version of litellm completion"""
    import litellm

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Error in litellm streaming completion: {e}")
        yield f"Error: {str(e)}"


@register_api("anthropic_streaming")
def chat_completion_anthropic_streaming(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    """Streaming version of Anthropic completion"""
    import anthropic
    
    if api_dict:
        client = anthropic.Anthropic(api_key=api_dict["api_key"])
    else:
        client = anthropic.Anthropic()
    
    try:
        # Convert messages to Anthropic format
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append(msg)
        
        stream = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message if system_message else None,
            messages=conversation_messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.text:
                yield chunk.delta.text
                
    except Exception as e:
        print(f"Error in Anthropic streaming completion: {e}")
        yield f"Error: {str(e)}"

@register_api("gemini")
def http_completion_gemini(model, messages, **kwargs):
    import requests
    
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    sys_prompt = None
    if messages[0]["role"] == "system":
        sys_prompt = {
            "parts":[
                {"text": messages[0]["content"]}
                ]
            }
        messages = messages[1:]
        
    role_map = {"user": "user",
                "assistant": "model"}
    
    conv = [{"parts":[{"text":turn["content"]}], "role":role_map[turn["role"]]} for turn in messages]
    
    json_request = {
            "contents": conv,
            "safetySettings": safety_settings,
            "systemInstruction": sys_prompt,
    }

    if "temperature" in kwargs and "max_tokens" in kwargs:
        gen_config = {
            "temperature": kwargs["temperature"],
            "maxOutputTokens": kwargs["max_tokens"],
        }
        json_request["generationConfig"] = gen_config
    elif "temperature" in kwargs:
        gen_config = {
            "temperature": kwargs["temperature"],
        }
        json_request["generationConfig"] = gen_config
    elif "max_tokens" in kwargs:
        gen_config = {
            "maxOutputTokens": kwargs["max_tokens"],
        }
        json_request["generationConfig"] = gen_config
        
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json=json_request,
            )
        except Exception as e:
            print(f"**API REQUEST ERROR** Reason: {e}.")
            time.sleep(API_RETRY_SLEEP)
        if response.status_code != 200:
            print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")
            time.sleep(API_RETRY_SLEEP)
        try:
            output = {
                "answer": response.json()["candidates"][0]["content"]["parts"][0]["text"],
            }
        except KeyError as e:
            print(type(e), e)
            print(response.json())
    return output
    

@register_api("vertex")
def vertex_completion_gemini(model, messages, project_id, regions, **kwargs):
    import requests
    import subprocess
    
    output = API_ERROR_OUTPUT
    
    # Obtain the access token using gcloud CLI
    access_token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"], 
        text=True
    ).strip()

    if messages[0]["role"] == "system":
        data = {
            "systemInstruction": {
                "role": "system", # ignored by vertexi api (04/18/2025)
                "parts": [{
                    "text": messages[0]["content"]
                }]
            },
        }
        messages = messages[1:]
    else:
        data = {}
        
    role_map = {
        "user": "user",
        "assistant": "model"
    }
    
    messages = [{"parts":[{"text":turn["content"]}], "role":role_map[turn["role"]]} for turn in messages]

    url = (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{regions}/publishers/google/models/"
        f"{model}:generateContent"
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    data = data | {
        "contents": messages,
    } 
    
    if "temperature" in kwargs or "max_tokens" in kwargs:
        gen_config = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            gen_config["maxOutputTokens"] = kwargs["max_tokens"]
        data["generationConfig"] = gen_config

    response = requests.post(url, json=data, headers=headers)
    
    try:
        output = {
            "answer": response.json()["candidates"][0]["content"]["parts"][0]["text"],
        }
    except KeyError as e:
        print(type(e), e)
        print(response.json())
        
    return output


@register_api("cohere")
def chat_completion_cohere(model, messages, temperature, max_tokens, **kwargs):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = {
                "answer": response.text
            }
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


@register_api("meta")
def chat_completion_meta(model, messages, temperature, max_tokens, api_dict, **kwargs):
    assert api_dict
    texts = [{"role": m["role"],
              "text": m["content"]} for m in messages]
            
    output = ""
    for _ in range(API_MAX_RETRY):
        try:
            res = requests.post(
                f"{api_dict['api_base']}/chat_stream_completions?access_token={api_dict['api_key']}",
                stream=True,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "chunks_delimited": True,
                    "messages": texts,
                    "options": {
                        "max_tokens": max_tokens,
                        "generation_algorithm": "top_p",
                        "top_p": 1,
                        "temperature": temperature,
                    },
                },
                timeout=30,
            )
            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        part = json.loads(line.decode("utf-8"))
                        if "text" in part:
                            output += part["text"]
                break
            else:
                print(f"**API REQUEST ERROR** Code: {res.status_code}")
                time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print("**API REQUEST ERROR** Reason: Unknown.")
            time.sleep(API_RETRY_SLEEP)
            continue
        
    return {
        "answer": output
    }


def batch_submit_sglang(
    executor, 
    tokenizer, 
    temperature, 
    max_tokens, 
    all_context,
    max_context_length=None,
    end_think_token=None,
):
    print(f"DEBUG: sglang_completion_qwq: max_context_length: {max_context_length}")
    
    sampling_params = {
        "temperature": temperature,
        "skip_special_tokens": False,
        "max_new_tokens": max_tokens - 1,
        "no_stop_trim": True,
    }
        
    batch_prompt_token_ids = []
    batch_uids =[]
    uid_to_prompt = {}
    uid_to_response = {}
    
    for context in all_context:
        prompt_token_ids = tokenizer.apply_chat_template(
            context['turns'],
            add_generation_prompt=True,
            tokenize=True,
        )
        
        if max_context_length and (len(prompt_token_ids) + max_tokens) > max_context_length:
            print(f"DEBUG: sglang_completion_qwq: context length ({len(prompt_token_ids) + max_tokens}) > max_context_length ({max_context_length}), skip this context")
            continue
        
        batch_prompt_token_ids.append(prompt_token_ids)
        batch_uids.append(context['uid'])
        
        uid_to_prompt[context['uid']] = context['turns']
        
    err_msg = f"ERROR: len(batch_prompt_token_ids): {len(batch_prompt_token_ids)} != len(batch_uids): {len(batch_uids)}"
    assert len(batch_prompt_token_ids) == len(batch_uids), err_msg
    
    _ = executor.submit(
        prompt_token_ids=batch_prompt_token_ids,
        sampling_params=[sampling_params] * len(batch_uids),
        keys=batch_uids,
    )
    
    for request in tqdm(executor.as_completed(), total=len(batch_uids)):
        uid = request.key()
        result = request.result()
        raw_response = tokenizer.decode(
            result['output_ids'],
            skip_special_tokens=True,
        )
        
        if end_think_token:
            thought, _, ans = raw_response.partition(end_think_token)
            if ans == "":
                uid_to_response[uid] = {"thought": thought, "answer": raw_response}
            else:
                uid_to_response[uid] = {"thought": thought, "answer": ans}
        else:
            uid_to_response[uid] = {"answer": raw_response}
    
    # assert len(uid_to_response) == len(all_context), f"ERROR: len output ({len(uid_to_response)}) != len input ({len(all_context)})"
    return uid_to_response


def _infer_cuda_tp_world_size():
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is None:
        tp_world_size = 8
    else:
        tp_world_size = len(cuda_devices.split(","))
    return tp_world_size


def download_model(model: str, max_workers: int = 64):
    import subprocess
    
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    cmd = [
        "huggingface-cli", 
        "download", 
        f"--max-workers={max_workers}", 
        model
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:


@register_engine("sglang")
def sglang_completion(
    model, 
    batch_context,
    answer_file,
    temperature,
    max_tokens=32768,
    end_think_token=None,
    **kwargs,
):
    from transformers import AutoTokenizer
    from utils.sglang_server import SGLangServerExecutor
    import re

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    uids = [context['uid'] for context in batch_context]
    prompts = [context['instruction'] for context in batch_context]
    code_envs = [context['environment'] for context in batch_context]
    processed_context = [
        {
            "uid": uids[i], 
            "turns": [{
                "content": prompts[i],
                "role": "user",
            }]
        } 
        for i in tqdm(range(len(uids)))
    ]
    download_model(model=model)
    
    server_args = {
        "model_path": model,
        "dtype": "auto",
        "tp_size": _infer_cuda_tp_world_size(),
        "mem_fraction_static": 0.7,
        "max_prefill_tokens": max_tokens,
        "max_workers": 256,
        "server_port": 30000,
    }
    
    executor = SGLangServerExecutor(
        **server_args,
    )
    
    
    uid_to_response = batch_submit_sglang(
        executor=executor, 
        tokenizer=tokenizer,
        temperature=temperature,
        max_tokens=max_tokens,
        all_context=processed_context,
        end_think_token=end_think_token,
    )
    
    executor.join()
    time.sleep(10)
        
    num_null = sum(
        [uid_to_response[uid]['answer'] is None for uid in uids if uid in uid_to_response]
    )
    
    records = []
    for i, context in enumerate(processed_context):
        uid = context['uid']
        if uid not in uid_to_response:
            continue

        answer_data = uid_to_response[uid]

        record = {
            "uid": uid,
            "ans_id": shortuuid.uuid(),
            "model": kwargs.get("model_display_name", model),
            "messages": context['turns'] + [
                {"content": answer_data, "role": "assistant"}
            ],
            "environment": code_envs[i],
            "tstamp": time.time(),
            "metadata": {},
        }

        records.append(record)

    with open(answer_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + '\n')


@register_api("aws_claude")
def chat_completion_aws_bedrock_claude(messages, api_dict=None, aws_region="us-west-2", **kwargs):
    """
    Call AWS Bedrock API for chat completion
    
    Args:
        model (str): Model ID
        conv (object): Conversation object containing messages
        temperature (float): Temperature parameter for response generation
        max_tokens (int): Maximum tokens in response
        api_dict (dict, optional): API configuration dictionary
        aws_region (str, optional): AWS region, defaults to "us-west-2"
        
    Returns:
        str: Generated response text or error message
    """
        
    # Configure AWS client if api_dict provided
    if api_dict is not None:
        bedrock_rt_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=api_dict.get('aws_access_key_id'),
            aws_secret_access_key=api_dict.get('aws_secret_access_key')
        )
    else:
        bedrock_rt_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,)
    
    output = API_ERROR_OUTPUT
    
    #get kwargs from settings
    temperature= kwargs["temperature"]
    max_tokens= kwargs["max_tokens"]
    model = kwargs["model_id"]
    
    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        prompt = messages[0]['content']

    
    # Retry logic for API calls
    for _ in range(API_MAX_RETRY):
        try:
            # Prepare request body
            prompt_json = {
                "system": sys_msg,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31",
                "stop_sequences": ["Human"]
            }

            # Call Bedrock API
            response = bedrock_rt_client.invoke_model(
                body=json.dumps(prompt_json),
                modelId=model,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            output = {"answer":response_body.get("content")[0].get("text")}
            break
            
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output

@register_api("aws_mistral")
def chat_completion_aws_bedrock_mistral(messages, api_dict=None, aws_region="us-west-2", **kwargs):
    """
    Call AWS Bedrock API for chat completion
    
    Args:
        model (str): Model ID
        conv (object): Conversation object containing messages
        temperature (float): Temperature parameter for response generation
        max_tokens (int): Maximum tokens in response
        api_dict (dict, optional): API configuration dictionary
        aws_region (str, optional): AWS region, defaults to "us-west-2"
        
    Returns:
        str: Generated response text or error message
    """
        
    # Configure AWS client if api_dict provided
    if api_dict is not None:
        bedrock_rt_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=api_dict.get('aws_access_key_id'),
            aws_secret_access_key=api_dict.get('aws_secret_access_key')
        )
    else:
        bedrock_rt_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,)
    
    output = API_ERROR_OUTPUT

    #get kwargs from settings
    temperature= kwargs["temperature"]
    max_tokens= kwargs["max_tokens"]
    model = kwargs["model_id"]
    
    # Retry logic for API calls
    for _ in range(API_MAX_RETRY):
        try:
            ## =============== Format prompt ================
            prompt = "\n".join([content for message in messages for content in message["content"]])
            formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"            
            body = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "stop": ["Human:"],
                "temperature": temperature,
            }

            # Call Bedrock API
            response = bedrock_rt_client.invoke_model(
                body=json.dumps(body),
                modelId=model,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            
            if "pixtral-large" in model: #us.mistral.pixtral-large-2502-v1:0
                output = {"answer": response_body.get("choices")[0].get("message").get("content")}
            else:
                output = {"answer": response_body.get("outputs")[0].get("text")}
      
            break
            
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
            
    return output


@register_api("mistral_streaming")
def chat_completion_mistral_streaming(model, messages, temperature, max_tokens, api_dict=None, **kwargs):
    """Streaming version of Mistral completion"""
    import openai
    
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error: {str(e)}"


@register_api("gemini_streaming")
def chat_completion_gemini_streaming(model, messages, **kwargs):
    """Streaming version of Gemini completion"""
    import google.generativeai as genai
    
    try:
        # Configure the API
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Create model
        model_genai = genai.GenerativeModel(model)
        
        # Convert messages to Gemini format
        conversation = model_genai.start_chat(history=[])
        
        # Get the last user message
        last_user_message = None
        for msg in messages:
            if msg["role"] == "user":
                last_user_message = msg["content"]
        
        if not last_user_message:
            yield "Error: No user message found"
            return
        
        # Stream the response
        response = conversation.send_message(last_user_message, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"Error: {str(e)}"
