from binoculars import *
import torch
import sklearn
import json
from tqdm import tqdm

# Form of the Dataset : 

# keys : [ "text" , "generated_text" , others ... ]

def eval_dataset(data_path, observer, performer,h_key, m_key, max_token=512, batch_size=32, mode="low-fpr",max_size=100):
    
    # Make a dictionary out of the jsonl file
    with open(data_path, "r") as f:
        dataset = []
        i = 0
        for line in f:
            if i == max_size:
                break
            data = json.loads(line)
            #print("Keys in JSON:", list(data.keys()))
            dataset.append({
                "text": data[h_key] ,
                "generated_text": data[m_key]
            })
            i+=1
    
    # Create the Binoculars detector
    detector = Binoculars(observer, performer, max_token, mode)

    # Make a log dictionary with the text ( true or generated), score, true class, predicted class
    logs = []
    for i in tqdm(range(0, len(dataset), batch_size),desc="Processing batches",total=(len(dataset) + batch_size - 1) // batch_size):
        batch = dataset[i : i + batch_size]
        texts_human = [ (item["text"],0) for item in batch ]
        texts_ai = [ (item["generated_text"],1) for item in batch ]
        texts = texts_human + texts_ai

        # Log file
        for text, true_class in texts:
            score = detector.compute_score(text)
            predicted_class = 0 if detector.predict(text) == "Human" else 1
            logs.append({
                "text": text,
                "score": score,
                "true_class": true_class,
                "predicted_class": predicted_class
            })
    
    return logs
