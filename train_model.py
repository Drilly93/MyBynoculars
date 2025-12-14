import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
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


NUM_LABELS = 2  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciation
model = SolonClassifier(num_classes=NUM_LABELS).to(device)
tokenizer = AutoTokenizer.from_pretrained("OrdalieTech/Solon-embeddings-mini-beta-1.1", trust_remote_code=True)

train_set = Dataset("xzyao/HC3-Evaluation", split="train", tokenizer=tokenizer)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
train_solon_classifier(model, train_loader, device, epochs=10)
val_set = Dataset("xzyao/HC3-Evaluation", split="validation", tokenizer=tokenizer)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
evaluate_solon_classifier(model, val_loader, device)

model_save_path = "solon_classifier.pth"
torch.save(model.state_dict(), model_save_path)
