import os
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

# --- IMPORT CRITIQUE POUR MINISTRAL ---
# On tente d'importer la classe spécifique. 
# Si elle n'existe pas (mauvaise version de transformers), on le signale.
try:
    from transformers import Mistral3ForConditionalGeneration
except ImportError:
    Mistral3ForConditionalGeneration = None
    print("Installez Mistral3ForConditionalGeneration et Mettez à jour 'transformers'.")

from metrics import perplexity, entropy
from contextlib import nullcontext

THRESHOLD = 0.9

class Binoculars_Mistral:
    def __init__(self,
        observer="mistralai/Ministral-3-3B-Base-2512",
        performer="mistralai/Ministral-3-3B-Instruct-2512",
        max_token=512,
        mode="low-fpr"):

        self.max_token = max_token
        self.mode = mode
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Loading observer: {observer}")
        self.observer_model = self._load_specific_model(observer)
        
        print(f"Loading performer: {performer}")
        self.performer_model = self._load_specific_model(performer)

        self.observer_model.eval()
        self.performer_model.eval()


        self.tokenizer = AutoTokenizer.from_pretrained(observer, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

    def _load_specific_model(self, model_name):
        """Force l'utilisation de Mistral3ForConditionalGeneration"""
        if Mistral3ForConditionalGeneration is None:
            raise ImportError("Impossible de charger Ministral: la classe Mistral3ForConditionalGeneration est introuvable.")
            
        return Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": self.device},
            torch_dtype="auto"
        )

    def tokenize(self, batch: list[str]):
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_token,
            return_token_type_ids=False
        ).to(self.device)
        return encodings 

    def compute_score(self, text):
        encodings = self.tokenize([text])
        
        use_amp = self.device.startswith("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
        
        with ctx:
            observer_logits = self.observer_model(**encodings).logits
            performer_logits = self.performer_model(**encodings).logits

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(encodings, observer_logits, performer_logits, self.tokenizer.pad_token_id)
        B_score = (ppl / x_ppl)
        return float(B_score[0])

    def predict(self, text):
        score = self.compute_score(text)
        return "Human" if score > 50 else "AI"