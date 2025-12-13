python main.py      # Quick test script for MyBinoculars

python eval.py \
  --data_path data/my_cc_news.jsonl \
  --pretrained_model google/gemma-3-270m \
  --instruct_model google/gemma-3-270m-it \
  --human_text_key text \
  --machine_text_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt