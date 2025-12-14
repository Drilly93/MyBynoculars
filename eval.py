import os
import argparse
import torch
import pandas as pd

from data_score import eval_dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Precise it
    parser.add_argument("--data_path", type=str, help="Path to the jsonl file")
    parser.add_argument("--pretrained_model", type=str) # pretrained model 
    parser.add_argument("--instruct_model", type=str) # instruct model 
    parser.add_argument("--human_text_key", type=str, help="key for the human-generated text")
    parser.add_argument("--machine_text_key", type=str,help="key for the machine-generated text")
    parser.add_argument("--output_file", type=str,default="output")

    # Hyper parameters 
    parser.add_argument("--max_token", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, default="low-fpr", help="Mode of the detector : low-fpr / high-tpr")

    args = parser.parse_args()

    # Use of GPU or not 
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")

    # os.makedirs(f"{args.experiment_path}", exist_ok=True)

    print("=" * 5, "start", "=" * 5)

    logs = eval_dataset(
        data_path=args.data_path,
        observer=args.pretrained_model,
        performer=args.instruct_model,
        h_key=args.human_text_key,
        m_key=args.machine_text_key,
        max_token=args.max_token,
        batch_size=args.batch_size,
        mode=args.mode
    )

    # Create a #results csv file
    df = pd.DataFrame(logs)

    # Put it inside results folder keep dataset file name
    output_csv = os.path.join("results", f"{args.output_file}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Logs saved to {output_csv}")

    print("=" * 5, "END", "=" * 5)

    df_logs = pd.DataFrame(logs)
    df_scores = df_logs[["score", "true_class", "predicted_class"] ]
    df_scores = df_scores.sample(n=30, random_state=42)
    print(df_scores.head(30))

# Make the command line call example
# 
# python main.py 
#   --data_path data/my_cc_news.jsonl 
#   --pretrained_model google/gemma-3-270m 
#   --instruct_model google/gemma-3-270m-it 
#   --human_text_key text 
#   --machine_text_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt
