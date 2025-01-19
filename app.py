import contextlib
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os, ast, time, random, openai, torch, json
from typing import Literal

load_dotenv()

torch.manual_seed(42)

MODEL = Literal["openai", "mistral", "merged", 'phi']

Model_MAP = {
    "openai": "gpt-4o-mini",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "merged": "yvelos/phi-merge-arithmetic",
    "phi": "microsoft/Phi-3-mini-128k-instruct"
}

generation_args = {
    "max_new_tokens": 1024,
    "top_p": 0.95,
    "return_full_text": False,
    "temperature": 0.1,
    "do_sample": True,
}
def login_hub(token=None):
    return login(token=token) if token else login(token=os.environ['HUB_TOKEN'])

login_hub()

openai.api_key = os.environ["OPENAI_API_KEY"]

def bnbConfig(bnb_4bits_compute_dtype="float16"):
    bnb_4bits_compute_dtype = getattr(torch, bnb_4bits_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bits_compute_dtype,
    )
def load_model(
    model:MODEL
    ):
    
    return AutoModelForCausalLM.from_pretrained(
        Model_MAP[model],
        quantization_config=bnbConfig(),
        torch_dtype=torch.float16,
        use_cache=False,
        low_cpu_mem_usage=True,
        return_dict=True,
        device_map={"": 0},
    )

def load_tokenizer(model_path: MODEL):
    tokenizer = AutoTokenizer.from_pretrained(Model_MAP[model_path])
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_json_file(input_path: str):
    data = []
    with open("test.jsonl", 'r') as file:
        data.extend(json.loads(line) for line in file)
    return data


def zero_shot_prompting(
    input_path, 
    output_path,
    model_type: MODEL,
):
    data = load_json_file(input_path)
    system_prompt = """Generate a comprehensive answer to the given question solely based on the content provided.
Provide the title and summary in the english language. Your output should look like this {"title": "title", "summary": "summary"}"""
    with open(f"results/zero_shot{model_type}_{output_path}", 'w') as file:
        start_time = time.time()
        answer_data = []
        if model_type == 'openai':    
            for item in data:
                chat_response = apply_chat_template(
                    Model_MAP[model_type],
                    system_prompt,
                    item['instruction']
                
                )
                with contextlib.suppress(Exception):
                    output = json.loads(chat_response.strip())
                _extracted_from_prompting(item, output, file, answer_data)
        
        else:
            pipe = pipeline(
                "text-generation", 
                model=load_model(model_type), 
                tokenizer=load_tokenizer(model_type),
                # model=Model_MAP[model_type],
                # tokenizer=Model_MAP[model_type],
                # torch_dtype=torch.float16,
                # device="cuda"
            )
            for i, item in enumerate(data[:]):       
                messages = [
                    {"role": 'system', "content": system_prompt},
                    {"role": "user", "content": item['instruction']},
                ]
                with torch.no_grad():
                    print(f"==== Start gen {i}======")
                    output = pipe(messages, **generation_args)
                    output = output[0]['generated_text']
                    print(output)
                    # output = ast.literal_eval(output)
                    with contextlib.suppress(Exception):
                        output = json.loads(output.strip())
                    _extracted_from_prompting(item, output, file, answer_data)
                    print(f"End gen {i}")
        end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Inference time: {execution_time:.4f} secondes")
            
    return answer_data

def few_shot_prompting(
    input_path, 
    output_path,
    model_type: MODEL,
    tokenizer=False, 
    ):
    data = load_json_file(input_path)
    answer_data = []
    start_time = time.time()
    if tokenizer and model_type != 'openai':
        pipe = pipeline(
            "text-generation", 
            model=load_model(model_type), 
            tokenizer=load_tokenizer(model_type),
            # device="cuda", 
            # torch_dtype=torch.float16,
        )
    with open(f"results/few_shot_1_{model_type}_{output_path}", 'w') as file:
        for item in data[48:]:
            sample = random.choice(data)
            system_prompt = f"""Generate a comprehensive answer to the given question solely based on the content provided.
Provide the title and description in the english language. Your output should be a json output like this {{"title": "title", "summary": "summary"}}".\nHere is an example: {sample['instruction']} \n#answer\n {{"title":"{sample['answer']['title']}", "summary": "{sample['answer']['summary']}"}}"""
            # print(system_prompt)
            if model_type == 'openai':
                chat_response = apply_chat_template(
                    Model_MAP[model_type],
                    system_prompt,
                    item['instruction']

                )
                with contextlib.suppress(Exception):
                    output = json.loads(chat_response.strip())
                _extracted_from_prompting(item, output, file, answer_data)
            else:
                messages = [
                    {"role": 'system', "content": system_prompt},
                    {"role": "user", "content": item['instruction']},
                ]
                with torch.no_grad():
                    output = pipe(messages, **generation_args)
                    output = output[0]['generated_text']
                    print(output)
                    with contextlib.suppress(Exception):
                        output = json.loads(output.strip())
                    _extracted_from_prompting(item, output, file, answer_data)
        end_time = time.time()

    execution_time = end_time - start_time
    print(f"Inference time: {execution_time:.4f} secondes")

    return answer_data

# TODO Rename this here and in `few_shot_prompting`
def _extracted_from_prompting(item, output, file, answer_data):
    output = parse_json_response(output)
    output['id'] = item['id']
    json.dump(output, file)
    file.write('\n')
    answer_data.append(output)
    
def parse_json_response(response: str):
    """
    correct and parse the json string to convert then in python dictionary.

    Args:
        response (str): The raw json response under the string format.

    Returns:
        dict: A python valid dictionary or error if the parsing failed.
    """
    if isinstance(response, dict): return response
    try:
        # case 1 : remove `json`  tag or other non relevant spaces
        response = response.strip()

        # case 2 : Check if the string use the simple '' for the key/values
        if response.startswith("{") and response.endswith("}"):
            if "'" in response and '"' not in response:
                # Replace ''by ""
                response = response.replace("'", '"')

            # case 3 : If the string has '' and "", correct the errors.
            if re.search(r':\s*\'', response):
                # Replace '' around of values by ""
                response = re.sub(r':\s*\'(.*?)\'', r': "\1"', response)

            # case 4 : Parser the json string
            return json.loads(response)
        elif "Title:" in response and "Summary:" in response:
            lines = response.strip().split("\n\n")
            return {
                "title": lines[0].replace("Title: ", "").strip(),
                "summary": lines[1].replace("Summary: ", "").strip(),
            }
    except json.JSONDecodeError as e:
        response = ast.literal_eval(response)
        return response
        # print(" Error parsing du JSON :", e)


def apply_chat_template(
    model: str,
    system_prompt: str, 
    user_prompt: str
    ):
    conversation = [{"role": "user", "content": user_prompt}]
    message_input = conversation.copy()
    prompt = [{"role": "system", "content": system_prompt}]
    
    message_input.insert(0, prompt[0])
    
    try:
        chat_completion = openai.chat.completions.create(
            model=model,
            temperature=0.82,
            frequency_penalty=0,
            max_tokens=512,
            top_p=1,
            presence_penalty=0,
            seed=42,
            messages=message_input
        )
        chat_response = chat_completion.choices[0].message.content
        
        conversation.append({"role": "assistant", "content": chat_response})
        print(chat_response)
        return chat_response
    except Exception as e:
        raise e


if __name__ == '__main__':
    few_shot_prompting(
        'dataset/test.jsonl', 
        "output.jsonl",
        'phi',
        True
    )
    # zero_shot_prompting(
    #     'dataset/test.jsonl', 
    #     "output.jsonl",
    #     'merged',
    # )
