import requests
import json
import yaml
from statistics import mean
import os
import logging

# Load test prompts from YAML file
def load_test_prompts(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to set up a unique logger for each prompt
def setup_logger(model_name, prompt_index):
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(f"{model_name}_prompt{prompt_index}")
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, f"{model_name}_prompt{prompt_index}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger

# Function to pull a model
def pull_model(model_name):
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name}
    
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Model {model_name} pulled successfully.")
        else:
            print(f"Failed to pull model {model_name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error pulling model {model_name}: {e}")

# Function to unload a model from memory
def unload_model(model_name):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "keep_alive": 0}
    
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Model {model_name} unloaded from memory.")
        else:
            print(f"Failed to unload model {model_name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error unloading model {model_name}: {e}")

# Function to delete a model
def delete_model(model_name):
    url = "http://localhost:11434/api/delete"
    payload = {"name": model_name}
    
    try:
        response = requests.delete(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Model {model_name} deleted successfully.")
        else:
            print(f"Failed to delete model {model_name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error deleting model {model_name}: {e}")

# Function to call the LLM API and collect metrics
def call_llm_api(prompt, model="llama3.1"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            api_response = response.json()
            total_duration = api_response.get("total_duration", 0) / 1e9  # Convert to seconds
            load_duration = api_response.get("load_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_duration = api_response.get("prompt_eval_duration", 0) / 1e9  # Convert to seconds
            eval_duration = api_response.get("eval_duration", 0) / 1e9  # Convert to seconds
            eval_count = api_response.get("eval_count", 0)

            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0

            return {
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_duration": eval_duration,
                "tokens_per_second": tokens_per_second,
                "response": api_response.get("response", "")
            }
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

# Function to run the test for a single prompt multiple times and return averaged metrics
def run_tests_for_prompt(prompt, model="llama3.1", test_runs=3):
    metrics_list = []
    
    for _ in range(test_runs):
        metrics = call_llm_api(prompt, model)
        if metrics:
            metrics_list.append(metrics)
    
    if metrics_list:
        average_metrics = {
            "average_total_duration": mean([m["total_duration"] for m in metrics_list]),
            "average_load_duration": mean([m["load_duration"] for m in metrics_list]),
            "average_prompt_eval_duration": mean([m["prompt_eval_duration"] for m in metrics_list]),
            "average_eval_duration": mean([m["eval_duration"] for m in metrics_list]),
            "average_tokens_per_second": mean([m["tokens_per_second"] for m in metrics_list]),
            "response_example": metrics_list[0]["response"]  # Example response from the first run
        }
        return average_metrics
    return None

# Function to run all tests for multiple models and log results
def run_all_tests(test_file, model_list, test_runs=3):
    test_data = load_test_prompts(test_file)['tests']

    for model_name in model_list:
        pull_model(model_name)  # Pull model before running tests
        
        for index, test in enumerate(test_data, start=1):
            prompt = test["prompt"]
            logger = setup_logger(model_name, index)  # Setup a unique logger for each prompt

            logger.info(f"Running tests for model {model_name}, prompt {index}: '{prompt}'")
            logger.info("-" * 50)  # Vertical separator
            avg_metrics = run_tests_for_prompt(prompt, model_name, test_runs)
            
            if avg_metrics:
                logger.info(f"--- Average Metrics for Prompt {index} ---")
                logger.info(f"Total Duration: {avg_metrics['average_total_duration']:.2f} seconds")
                logger.info(f"Load Duration: {avg_metrics['average_load_duration']:.2f} seconds")
                logger.info(f"Prompt Eval Duration: {avg_metrics['average_prompt_eval_duration']:.2f} seconds")
                logger.info(f"Response Eval Duration: {avg_metrics['average_eval_duration']:.2f} seconds")
                logger.info(f"Tokens per Second: {avg_metrics['average_tokens_per_second']:.2f} tokens/s")

                logger.info("-" * 50)
                logger.info("Example Response:\n")
                logger.info(f"{avg_metrics['response_example']}\n")
            else:
                logger.warning(f"No data collected for prompt {index}.\n")

        unload_model(model_name)  # Unload model after running tests
        delete_model(model_name)  # Delete model after running tests

if __name__ == "__main__":
    test_file = "test_prompts.yaml"
    model_list = ['8b-instruct-q4_0', '8b-instruct-q5_0', '8b-instruct-q8_0']
    test_runs = 5
    run_all_tests(test_file, model_list, test_runs)
