# Model Architecture:
# [Input Text]
#      |
#      V
# [Tokenizer]
#   (input_ids, attention_mask)
#      |
#      V
# [Base Transformer (e.g., BERT)]
#      |
#      V
# [Additional Transformer Encoder Layer 1]
#      |
#      V
# [Additional Transformer Encoder Layer 2] (If num_additional_transformers > 1)
#      |
#      V
#    ... (More additional layers)
#      |
#      V
# [Additional Transformer Encoder Layer N] (Where N = num_additional_transformers)
#      |
#      V
# [Extract [CLS] Token Representation]
#      |
#      V
# [Dropout Layer]
#      |
#      V
# [Linear Layer (Classifier)]
#      |
#      V
# [Output Logits]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import random
import matplotlib.pyplot as plt
import transformers
from torch.nn import TransformerEncoderLayer
from load_data import load_combined_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


texts, labels = load_combined_data()



class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )
        return encoded_input['input_ids'].squeeze(0), encoded_input['attention_mask'].squeeze(0), torch.tensor(label, dtype=torch.long)

def collate_fn_transformer(batch):
    input_ids, attention_masks, labels = zip(*batch)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)

    return input_ids, attention_masks, labels



train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=SEED, stratify=labels
)


MODEL_NAME = 'bert-base-uncased'

train_transformer_dataset = TransformerDataset(train_texts, train_labels, MODEL_NAME)
val_transformer_dataset = TransformerDataset(val_texts, val_labels, MODEL_NAME)

BATCH_SIZE = 8 
train_transformer_loader = DataLoader(train_transformer_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_transformer)
val_transformer_loader = DataLoader(val_transformer_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_transformer)



class TransformerMegalodon(nn.Module):
    def __init__(self, model_name, num_additional_transformers, num_classes, dropout_rate):
        super().__init__()
        self.base_transformer = transformers.AutoModel.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        self.additional_transformers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=dropout_rate,
                activation=F.gelu,
                batch_first=True
            )
            for _ in range(num_additional_transformers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes 

    def forward(self, input_ids, attention_mask):
        outputs = self.base_transformer(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        padding_mask = (attention_mask == 0)

        current_sequence = last_hidden_state
        for transformer_layer in self.additional_transformers:
            current_sequence = transformer_layer(current_sequence, src_key_padding_mask=padding_mask)

        
        pooled_output = current_sequence[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output) 

        return logits



def train_transformer_model(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_true = []
    all_probs = [] 

    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask) 

        loss = criterion(logits, labels) 

        
        _, predicted_labels = torch.max(logits, dim=1)

        
        probs = F.softmax(logits, dim=1)

        all_preds.extend(predicted_labels.cpu().detach().numpy())
        all_true.extend(labels.cpu().detach().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_true, all_preds)
    
    precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)

    
    try:
         
        if len(set(all_true)) > 1:
             auc = roc_auc_score(all_true, all_probs, multi_class='ovr')
        else:
             auc = 0.0 
    except ValueError:
        
        
        auc = 0.0 

    return avg_loss, acc, precision, recall, f1, auc

def evaluate_transformer_model(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds_probs = []
    all_preds_labels = []
    all_true = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask) 

            loss = criterion(logits, labels)
            epoch_loss += loss.item()

            
            _, predicted_labels = torch.max(logits, dim=1)

            
            probs = F.softmax(logits, dim=1)

            all_preds_probs.extend(probs.cpu().numpy())
            all_preds_labels.extend(predicted_labels.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_true, all_preds_labels)
    
    precision = precision_score(all_true, all_preds_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true, all_preds_labels, average='weighted', zero_division=0)
    f1 = f1_score(all_true, all_preds_labels, average='weighted', zero_division=0)

    
    try:
         
        if len(set(all_true)) > 1:
             auc = roc_auc_score(all_true, all_preds_probs, multi_class='ovr')
        else:
             auc = 0.0 
    except ValueError:
         auc = 0.0 

    return avg_loss, acc, precision, recall, f1, auc, all_preds_probs, all_true



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 6 
DROPOUT_RATE = 0.3
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10 


num_additional_transformers = 1

model2 = TransformerMegalodon(
    MODEL_NAME, num_additional_transformers, NUM_CLASSES, DROPOUT_RATE
).to(device)

optimizer2 = AdamW(model2.parameters(), lr=LEARNING_RATE)

criterion2 = nn.CrossEntropyLoss()


train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_aucs, val_aucs = [], []

print(f"\n--- Training Approach 2: Transformer Megalodon ({MODEL_NAME}, Multi-class) ---")
best_val_loss = float('inf')
val_predictions_probs_for_auc_2 = None 
val_labels_for_auc_2 = None 

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_transformer_model(model2, train_transformer_loader, optimizer2, criterion2, device)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, current_val_preds_probs, current_val_labels = evaluate_transformer_model(model2, val_transformer_loader, criterion2, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_precisions.append(train_precision)
    val_precisions.append(val_precision)
    train_recalls.append(train_recall)
    val_recalls.append(val_recall)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        val_predictions_probs_for_auc_2 = current_val_preds_probs
        val_labels_for_auc_2 = current_val_labels
        

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC (OvR): {train_auc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC (OvR): {val_auc:.4f}")



epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(epochs_range, train_precisions, label='Train Precision (Weighted)')
plt.plot(epochs_range, val_precisions, label='Validation Precision (Weighted)')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title(f'Precision Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(epochs_range, train_recalls, label='Train Recall (Weighted)')
plt.plot(epochs_range, val_recalls, label='Validation Recall (Weighted)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title(f'Recall Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs_range, train_f1s, label='Train F1 Score (Weighted)')
plt.plot(epochs_range, val_f1s, label='Validation F1 Score (Weighted)')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title(f'F1 Score Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs_range, train_aucs, label='Train AUC (OvR)')
plt.plot(epochs_range, val_aucs, label='Validation AUC (OvR)')
plt.xlabel('Epoch')
plt.ylabel('AUC (OvR)')
plt.title(f'AUC (OvR) Curves (Approach 2: {MODEL_NAME}, Multi-class)')
plt.legend()

plt.tight_layout()
plt.show()





if val_predictions_probs_for_auc_2 is not None and val_labels_for_auc_2 is not None and len(set(val_labels_for_auc_2)) > 1:
    print("\nMulti-class ROC curve plotting is more involved (e.g., One-vs-Rest) and is skipped for simplicity.")
    print(f"Validation AUC (OvR) for the best epoch: {roc_auc_score(val_labels_for_auc_2, val_predictions_probs_for_auc_2, multi_class='ovr'):.4f}")
elif val_predictions_probs_for_auc_2 is not None and val_labels_for_auc_2 is not None:
     print("\nCannot calculate/plot multi-class ROC curve: only one class present in validation labels.")
else:
    print("\nNo validation predictions available for final multi-class AUC calculation.")