from binoculars import *
import torch
import sklearn
import json
# Form of the Dataset : 

# keys : [ id, "text" , "generated_text" ]

def eval_dataset(data_path, observer, performer,h_key, m_key, max_token=512, batch_size=32, mode="low-fpr"):
    
    # Make a dictionary out of the jsonl file
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append({
                "id": data["id"],
                "text": data[h_key] ,
                "generated_text": data[m_key]
            })
    
    # Create the Binoculars detector
    detector = Binoculars(observer, performer, max_token, modeo)

    # Make a log dictionary with the text ( true or generated), score, true class, predicted class
    logs = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        texts_human = [ item["text"] for item in batch ]
        texts_ai = [ item["generated_text"] for item in batch ]

        # Log file
        for text, true_class in zip(texts_human, [0]*len(texts_human)):
            score = detector.compute_score(text)
            predicted_class = 0 if detector.predict(text) == "Human" else 1
            logs.append({
                "text": text,
                "score": score,
                "true_class": true_class,
                "predicted_class": predicted_class
            })

    return logs