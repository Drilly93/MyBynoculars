import os
import argparse
import torch
import pandas as pd
import json

from data_score import eval_dataset

def find_generated_key(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            for k in data.keys():
                if "generate" in k.lower():
                    return k  
    return "generate"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Precise it
    parser.add_argument("--datasets_dir", type=str)
    parser.add_argument("--pretrained_model", type=str) # pretrained model 
    parser.add_argument("--instruct_model", type=str) # instruct model 
    parser.add_argument("--results_dir", type=str,default="outputs")

    # Hyper parameters 
    parser.add_argument("--max_token", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="low-fpr")

    args = parser.parse_args()

    # Use of GPU or not 
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")

    # Get all files names jsonl

    dataset_names = [f for f in os.listdir(args.datasets_dir) if f.endswith(".jsonl")]
    output_names = [ os.path.splitext(f)[0] for f in dataset_names ]
    os.makedirs(f"{args.results_dir}", exist_ok=True)

    print("=" * 5, "start", "=" * 5)


    for i in range(len(dataset_names)):
        data_path = os.path.join(args.datasets_dir, dataset_names[i])
        output_file = output_names[i]

        # Extract h_keys and m_keys from file name if possible

        print(f"Processing dataset: {data_path}")

        key_generated = find_generated_key(data_path)
        print(f"Identified generated text key: {key_generated}")
        
        logs = eval_dataset(
            data_path=data_path,
            observer=args.pretrained_model,
            performer=args.instruct_model,
            h_key="text",
            m_key= "generate",
            max_token=args.max_token,
            batch_size=args.batch_size,
            mode=args.mode
        )

        # Create a #results csv file
        df = pd.DataFrame(logs)

        # Put it inside results folder keep dataset file name
        output_csv = os.path.join(args.results_dir, f"{output_file}_results.csv")
        df.to_csv(output_csv, index=False)
        print(f"Logs saved to {output_csv}")

