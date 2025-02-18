import random
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"  # Default Ollama endpoint


def generate_with_ollama(prompt: str, model:str):
    """Calls Ollama's local API to generate text from the model."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model, #"deepseek-r1:1.5b" #hf.co/ubiqtuitin/deepseek_r1_medical-ft_q4_k_m
            "prompt": prompt,
            "stream": True,  # Streaming for real-time output
        },
        stream=True,
    )
    if response.status_code != 200:
        print(f"Error: Failed to call Ollama API. Status code {response.status_code}. Response: {response.text}")
        return  # Exit early if there is an error
    for line in response.iter_lines():
        if line:
            yield json.loads(line)["response"]


def reasoning_effort(question: str, model: str, min_thinking_tokens: int):
    """Forces the LLM to continue reasoning by replacing stop tokens."""
    response = ""
    if model == "deepseek-r1:1.5b":
        prompt = (
            "You are a student taking the USMLE exam. Your task is to answer the following "
            "question with one of the multiple choices. Answer the following question, using "
            "a chain of thought process to reason to get to the answer. "
            "Encapsulate your reasoning within <think> and </think> tokens. "
            "Encapsulate your answer in <answer> and </answer> tokens. "
            "Respond in the following format: \n"
            "<think> {Put your thought process here} </think>\n"
            "<answer> {Put your final answer here} </answer>\n\n"
            "Here is the question:\n"
            f"{question}"
        )
    else:
        prompt = f"{question}"

    n_thinking_tokens = 0
    num_swaps = 0

    # print(prompt, end="", flush=True)
    for chunk in generate_with_ollama(prompt, model):
        # print("chunk: ", chunk)
        # If chunk is None or empty, handle it properly
        try:
            if not chunk or chunk.strip() == "":  
                if n_thinking_tokens < min_thinking_tokens:
                    replacement = random.choice(["\nWait,", "\nHmm,", "\nAlternatively,"])
                    response += replacement
                    num_swaps += 1
                    n_thinking_tokens += len(replacement.split())  # Approximate token count
            elif ("</" in chunk.strip() or chunk.strip() == "</think>" or "answer" in chunk.strip().lower()) and n_thinking_tokens < min_thinking_tokens:
                replacement = random.choice(["\nWait,", "\nHmm,", "\nAlternatively,"])
                response += replacement
                num_swaps += 1
                n_thinking_tokens += len(replacement.split())  # Approximate token count
            else:
                # Normal processing of LLM output
                response += chunk
                n_thinking_tokens += len(chunk.split())
                
        except Exception as e:
            print(f"Error processing chunk: {e}. Skipping this chunk.")
            continue

    return response, n_thinking_tokens, num_swaps

# if __name__ == "__main__":
#     answer, thoughts, num_think_tokens, num_swaps = reasoning_effort("what is 1+1?", min_thinking_tokens=0)