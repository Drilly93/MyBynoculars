import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext

from metrics import perplexity, entropy

THRESHOLD = 1  # Example threshold value


class Binoculars:
    def __init__(self,
        observer= "google/gemma-3-270m",         #"tiiuae/falcon-7b", 
        performer = "google/gemma-3-270m-it",                  #"tiiuae/falcon-7b-instruct",
        max_token = 512,
        mode= "low-fpr"):

        self.max_token = max_token
        self.mode = mode
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer,
            device_map={"": self.device},
            load_in_8bit=True,
            torch_dtype="auto",
        )

        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer,
            device_map={"": self.device},
            load_in_8bit=True,
            torch_dtype="auto",
        )



        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.observer_model.eval()
        self.performer_model.eval()
                    
                    
    def tokenize(self, batch: list[str]):
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_token,
            return_token_type_ids=False
        ).to(self.device)

        return encodings # (input_ids, padding_mask) #type : transformers.BatchEncodin

    
    def compute_score(self, text):
   
        encodings = self.tokenize([text])
        # Optimize space VRAM with autocast
        use_amp = self.device.startswith("cuda")
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with torch.no_grad():
            with ctx:
                observer_logits = self.observer_model(**encodings).logits
                performer_logits = self.performer_model(**encodings).logits
        #print(observer_logits.shape, performer_logits.shape)

        # Binoculars score
        ppl = perplexity(encodings, performer_logits)  # array shape (B,) -> here B=1
        x_ppl = entropy(encodings, observer_logits, performer_logits,  self.tokenizer.pad_token_id)
        B_score = (ppl / x_ppl)
        return float(B_score[0])

    def predict(self, text):
        
        score = self.compute_score(text)
        return "Human" if score > 50 else "AI"