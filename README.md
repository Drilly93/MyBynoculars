# Binoculars - AI generataed text detector 

This repository implements and evaluates the Binoculars method for detecting machine-generated text using cross-perplexity between observer and performer models.

Google gemma is the model used by default. You can change the model at any time using parser arguments. It should donwload the pretrained model when first executing.

Necessity of a Huggings Face token to have access to pretrained model.

**Recomended execution** 
Intall all the dependences by executing :
>>> pip install -r requirements.txt

Test that your code works with :
>>> python main.py

Test that your code on a small dataset with :
>>> python eval.py
  --data_path data/my_cc_news.jsonl
  --pretrained_model google/gemma-3-270m
  --instruct_model google/gemma-3-270m-it
  --human_text_key text
  --machine_text_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt

Test the code on a directory of .jsonl dataset with :
>>> python eval_all.py 
    --datasets_dir data/datasets 
    --pretrained_model google/gemma-3-270m 
    --instruct_model google/gemma-3-270m-it 
    --results_dir output_gemini

Execute results.ipynb to see the results !!

**Note** : human_text_key is the associated field of the human text>. Same for machine_text_key.

## Info :

**metrics.py** : Contains the function to compute the perplexity and the cross perplexity 

**binoculars.py** : Contains the main class of the model use to evaluate the B_score of a dataset
    - compute_score : Compute the B-Score of a text with the binocular method

**results.ipynb** : Collect all the outputs, and plot the graphs for analysis. Also this file finds the Optimal threshold B.

**eval.py** : Evaluate the score of a specified dataset in the argument. It essentially take the arguments put in the terminal and use data_score.py to evaluate a dataset.

Code example : 
>>> python eval.py
  --data_path data/my_cc_news.jsonl
  --pretrained_model google/gemma-3-270m
  --instruct_model google/gemma-3-270m-it
  --human_text_key text
  --machine_text_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt

NB : my_cc_news.jsonl is a very hand crafted dataset to test

**data_score.py** : Extract the information of the dataset (jsonl file) and evaluate all the scores of the text and generated text

**eval_all.py** : Evaluate the score of all the datasets inside a directory and output the scores inside an output directory.

Code example : 
>>> python eval_all.py 
    --datasets_dir data/datasets 
    --pretrained_model google/gemma-3-270m 
    --instruct_model google/gemma-3-270m-it 
    --results_dir output_gemini


**binoculars_mistral.py** : Just a code associated to make it works with mistral

## Folders 

**data**: Constains the data
**data/code**: Used to generate the datasets
**img**: Results of the plots of results.ipynb 
**output_...**: Scores loaded into a csv files



