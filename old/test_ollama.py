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
    # Create the logs directory if it does not exist
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logger for this prompt
    logger = logging.getLogger(f"{model_name}_prompt{prompt_index}")
    logger.setLevel(logging.INFO)

    # Create a file handler for this prompt
    log_file = os.path.join(log_dir, f"{model_name}_prompt{prompt_index}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a logging format and add it to the file handler
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger

# Function to call the LLM API and collect metrics
def call_llm_api(prompt, model="llama3.1"):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        # "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            api_response = response.json()

            # Extract relevant metrics from the response
            total_duration = api_response.get("total_duration", 0) / 1e9  # Convert to seconds
            load_duration = api_response.get("load_duration", 0) / 1e9  # Convert to seconds
            prompt_eval_duration = api_response.get("prompt_eval_duration", 0) / 1e9  # Convert to seconds
            eval_duration = api_response.get("eval_duration", 0) / 1e9  # Convert to seconds
            eval_count = api_response.get("eval_count", 0)

            # Calculate tokens per second
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
        # Calculate averages for relevant metrics
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

# Function to run all tests from the YAML file and log results
def run_all_tests(test_file, model="llama3.1", test_runs=3):
    test_data = load_test_prompts(test_file)['tests']


    for index, test in enumerate(test_data, start=1):
        prompt = test["prompt"]
        logger = setup_logger(model, index)  # Setup a unique logger for each prompt

        # Add separators and line breaks between sections
        logger.info(f"Running tests for prompt {index}: '{prompt}'")
        logger.info("-" * 50)  # Vertical separator
        avg_metrics = run_tests_for_prompt(prompt, model, test_runs)
        
        if avg_metrics:
            logger.info(f"--- Average Metrics for Prompt {index} ---")
            logger.info(f"Total Duration: {avg_metrics['average_total_duration']:.2f} seconds")
            logger.info(f"Load Duration: {avg_metrics['average_load_duration']:.2f} seconds")
            logger.info(f"Prompt Eval Duration: {avg_metrics['average_prompt_eval_duration']:.2f} seconds")
            logger.info(f"Response Eval Duration: {avg_metrics['average_eval_duration']:.2f} seconds")
            logger.info(f"Tokens per Second: {avg_metrics['average_tokens_per_second']:.2f} tokens/s")

            logger.info("-" * 50)  # Vertical separator
            logger.info("Example Response:\n")
            logger.info(f"{avg_metrics['response_example']}\n")
        else:
            logger.warning(f"No data collected for prompt {index}.\n")

if __name__ == "__main__":
    # File containing test prompts (YAML format)
    test_file = "test_prompts.yaml"
    
    # Model name
    model_name = "llama3.1"
    
    # Number of test runs per prompt
    test_runs = 5
    
    # Run all tests
    run_all_tests(test_file, model=model_name, test_runs=test_runs)
