import requests
import json

def test_llm_api(prompt, model="llama3.1"):
    url = "http://localhost:11434/api/generate"  # The API URL
    headers = {
        "Content-Type": "application/json"  # Set content type to JSON
    }
    
    # The payload with the prompt and potentially a model
    payload = {
        "model": model,  # Model name might be required
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Send a POST request to the API with the prompt
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("API responded successfully!")
            print("Response (raw):")
            print(response.text)  # Print the raw response text instead of parsing it as JSON
        else:
            print(f"API returned an error: Status code {response.status_code}")
            print(f"Response content: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while trying to reach the API: {e}")

if __name__ == "__main__":
    # Example prompt to send to the LLM
    example_prompt = "Explain what artificial intelligence is."
    
    # Model name (if necessary for API)
    model_name = "llama3.1"
    
    # Call the function to test the API with the prompt
    test_llm_api(example_prompt, model=model_name)
