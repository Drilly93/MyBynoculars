from openai import OpenAI
import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm
import yaml
import concurrent.futures 

# Configuration
INPUT_GENERATED_FILE = "data/datasets/drcat_human.jsonl"
OUTPUT_BALANCED_FILE = "data/datasets/our_drcat_humane_imitation.json"
API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL_NAME = "gpt-4o-mini"
PROMPT_FILE = "data/code/prompt_base.yaml" 
MAX_WORKERS = 20 
print(API_KEY)

client = OpenAI(api_key=API_KEY)

def call_api_and_generate_text(amorce_prompt, base_template): 
    prompt = base_template.replace("{amorce}", amorce_prompt) 
    
    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You send back the rest of the article without anything else."},
        {"role": "user", "content": prompt}
        ],
        temperature=0.0, 
        max_tokens=400
        )
    generated_part = response.choices[0].message.content.strip()
    return generated_part
        

def main():
    dataset = pd.read_json(INPUT_GENERATED_FILE, lines=True)
    dataset_final = pd.DataFrame(columns=["title", "text", "generate"]) 
    
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f) 
    
    base_template = prompt_config['generation_config']['humane_template'] 
    
    tasks = [] 
    
    data_to_process = dataset 
    
    for i in tqdm(range(1000)): 
        ligne_i = data_to_process.iloc[i]
        full_original = ligne_i['text']
        words = full_original.split()
        amorce = " ".join(words[:30])
        
        tasks.append((ligne_i, amorce, base_template)) 
        

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: 
        results = executor.map(lambda t: (t[0], t[1], call_api_and_generate_text(t[1], t[2])), tasks)

    
    for ligne_i, amorce, new_generation in results: 
        
        dataset_final = pd.concat([dataset_final, pd.DataFrame({
            "title": [ligne_i['title']], 
            "text": [full_original], 
            "generate": [new_generation],
        })], ignore_index=True)
    
    dataset_final.to_json(OUTPUT_BALANCED_FILE, orient="records", lines=True)
    

if __name__ == "__main__":
    main()