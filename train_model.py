import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader



class SolonClassifier(nn.Module):
    def __init__(self, num_classes, model_name="OrdalieTech/Solon-embeddings-mini-beta-1.1"):
        super(SolonClassifier, self).__init__()
        self.solon = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for param in self.solon.parameters():
            param.requires_grad = False
        hidden_size = self.solon.config.hidden_size 
        self.classifier = nn.Linear(hidden_size, num_classes)

    def mean_pooling(self, model_output, attention_mask):
        """
        Effectue le mean pooling en ignorant les tokens de padding.
        """
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.solon(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        logits = self.classifier(pooled_output)
        return logits


def train_solon_classifier(model :SolonClassifier, dataloader, device, epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
def evaluate_solon_classifier(model: SolonClassifier, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def prepare_data(examples):
    """
    Applatit le batch et crée une liste d'entrées (texte, étiquette)
    où l'étiquette 0 est pour l'humain et 1 pour l'IA.
    """
    all_texts = []
    all_labels = []

    num_items_in_batch = len(examples["human_answers"])

    for i in range(num_items_in_batch):
        human_text = examples["human_answers"][i][0] if examples["human_answers"][i] else ""
        all_texts.append(human_text)
        all_labels.append(0)

        ai_text = examples["chatgpt_answers"][i][0] if examples["chatgpt_answers"][i] else ""
        all_texts.append(ai_text)
        all_labels.append(1)

    return {
        "text": all_texts,
        "label": all_labels
    }

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)



NUM_LABELS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.1 

model = SolonClassifier(num_classes=NUM_LABELS).to(device)
tokenizer = AutoTokenizer.from_pretrained("OrdalieTech/Solon-embeddings-mini-beta-1.1", trust_remote_code=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



print("Chargement du jeu de données...")
raw_datasets = load_dataset("xzyao/HC3-Evaluation")

print(f"Division du split 'train' en Entraînement et Validation ({100 * (1 - VALIDATION_FRACTION)}% / {100 * VALIDATION_FRACTION}%)")
split_datasets = raw_datasets["train"].train_test_split(
    test_size=VALIDATION_FRACTION, 
    seed=SPLIT_SEED
)

train_set = split_datasets["train"]
val_set = split_datasets["test"]


print("Préparation et tokenization des données...")

tokenized_train_set = train_set.map(
    prepare_data, 
    batched=True, 
    remove_columns=train_set.column_names
).map(tokenize_function, batched=True).rename_column("label", "labels")
tokenized_train_set = tokenized_train_set.remove_columns(["text"])

tokenized_val_set = val_set.map(
    prepare_data, 
    batched=True, 
    remove_columns=val_set.column_names
).map(tokenize_function, batched=True).rename_column("label", "labels")
tokenized_val_set = tokenized_val_set.remove_columns(["text"])


train_loader = DataLoader(
    tokenized_train_set, 
    batch_size=16, 
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    tokenized_val_set, 
    batch_size=16, 
    shuffle=False,
    collate_fn=data_collator
)

print(f"Taille du DataLoader d'entraînement: {len(train_loader)} lots")
print(f"Taille du DataLoader de validation: {len(val_loader)} lots")


print("\nDémarrage de l'entraînement...")
train_solon_classifier(model, train_loader, device, epochs=10)

print("\nDémarrage de l'évaluation...")
evaluate_solon_classifier(model, val_loader, device)


model_save_path = "solon_classifier.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nModèle sauvegardé dans {model_save_path}")