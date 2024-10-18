import yaml
import subprocess
import os
import time
import requests

# Function to check if the model is accessible
def check_model_loaded(port=11434):
    try:
        response = requests.get(f"http://localhost:{port}/api/ps")
        if response.status_code == 200:
            data = response.json()
            if "models" in data and len(data["models"]) > 0:
                return True
    except requests.exceptions.RequestException:
        return False
    return False

# Function to pull the required model using Ollama API
def pull_model(model_name, port=11434):
    url = f"http://localhost:{port}/api/pull"
    payload = {
        "name": model_name,
    }
    
    try:
        response = requests.post(url, json=payload, stream=False)
        if response.status_code == 200:
            print(f"Model {model_name} pulled successfully!")
        else:
            print(f"Failed to pull model {model_name}. Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while pulling model {model_name}: {e}")
        return False
    return True

# Function to stop and remove Docker container
def stop_and_remove_container(cont_name):
    subprocess.run(["docker", "stop", cont_name])  # Use cont_name
    subprocess.run(["docker", "rm", cont_name])
    print(f"Container {cont_name} removed successfully!")

# Function to run Docker container for each model and GPU combination
def run_docker_container(model_name, cont_name, gpu_ids, port=11434):
    gpu_devices = ','.join([str(gpu) for gpu in gpu_ids])

    # Start the container
    docker_run_command = [
        "docker", "run", "-d", f"--gpus=device={gpu_devices}", "-p", f"{port}:{port}", "--name", cont_name, "ollama/ollama"
    ]
    subprocess.run(docker_run_command)

    # Wait for the container to be fully running
    container_status_command = f"docker inspect -f '{{{{.State.Running}}}}' {cont_name}"
    container_running = False
    while not container_running:
        status = subprocess.check_output(container_status_command, shell=True).strip().decode('utf-8')
        if status == "true":
            container_running = True
        else:
            time.sleep(1)

    # Pull the model using the Ollama API, handle failure gracefully
    if not pull_model(model_name, port):
        print(f"Skipping model {model_name} due to pull failure.")
        return False  # Stop here if the model pull failed
    
    # Proceed with model loading and testing if pull is successful
    docker_exec_command = [
        "docker", "exec", "-dit", cont_name,
        "ollama", "run", model_name  # Adjust based on correct API usage
    ]
    
    try:
        # Execute the model loading, and capture any memory overflow errors
        subprocess.run(docker_exec_command, check=True)
    except subprocess.CalledProcessError as e:
        # Handle the error if model loading fails due to memory overflow
        if "CUDA out of memory" in str(e):
            print(f"Memory overflow occurred when loading model {model_name}. Skipping remaining models.")
            return False  # Signal to skip all remaining models
        else:
            print(f"An error occurred when loading model {model_name}: {e}")
            return False
    
    print(f"Waiting for model {model_name} to be accessible...")
    while not check_model_loaded(port=port):
        time.sleep(5)
    print(f"Model {model_name} is now accessible!")
    return True

# Import the test function from test_model_prompts
from test_model_prompts import test_all_prompts

# Main execution block
if __name__ == "__main__":
    
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test_file = config['test_file']
    test_runs = config['test_runs']
    model_list = config['model_list']  # Models in ascending size
    gpu_id_lists = config['gpu_id_lists']
    
    # Loop over GPU and model combinations
    for gpu_id_list in gpu_id_lists:
        skip_remaining_models = False  # Flag to determine if we should skip remaining models
        
        for model_name in model_list:
            cont_name = f"ollama_{model_name.split(':')[-1]}"
            
            if skip_remaining_models:
                print(f"Skipping model {model_name} due to previous memory overflow.")
                continue  # Skip all remaining models once a memory overflow occurs
            
            try:
                # Run Docker container and load model
                if not run_docker_container(model_name, cont_name, gpu_id_list):
                    skip_remaining_models = True  # Set the flag to skip remaining models
                    break  # Exit the loop for current GPU, no need to test larger models

                # Run tests after the model is loaded
                print("Testing prompts...")
                test_all_prompts(gpu_id_list, test_file, model=model_name, test_runs=test_runs)
                print("Testing finished successfully!")
                
            except Exception as e:
                # Log the error and clean up
                print(f"Error occurred during testing for model {model_name}: {e}")
            
            finally:
                # Clean up container
                stop_and_remove_container(cont_name)
                print(f"Cleaned up container for model {model_name}")
