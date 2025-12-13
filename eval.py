
from data_score import eval_dataset
import argparse

if __name__ == "__main__":

    # Precise it
    parser.add_argument("--data_path", type=str, help="Path to the jsonl file")
    parser.add_argument("--pretrained_model", type=str, default="gpt2") # pretrained model 
    parser.add_argument("--instruct_model", type=str) # instruct model 
    parser.add_argument("--human_text_key", type=str, help="key for the human-generated text")
    parser.add_argument("--machine_text_key", type=str,help="key for the machine-generated text")

    # Hyper parameters 
    parser.add_argument("--tokens_seen", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="low-fpr", help="Mode of the detector : low-fpr / high-tpr")

    args = parser.parse_args()

    # Use of GPU or not 
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")

    logs = eval_dataset(
        data_path=args.data_path,
        observer=args.pretrained_model,
        performer=args.instruct_model,
        h_key=args.human_text_key,
        m_key=args.machine_text_key,
        max_token=args.tokens_seen,
        batch_size=args.batch_size,
        mode=args.mode
    )

    # Create a #results csv file
    df = pd.DataFrame(logs)
    output_csv = args.data_path.replace(".jsonl", "_detection_logs.csv")
    df.to_csv(output_csv, index=False)
    print(f"Logs saved to {output_csv}")

    print("=" * 60, "END", "=" * 60)