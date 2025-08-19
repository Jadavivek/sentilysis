# Gated Recurrent Convolutional Network (GRU + Attention + Parallel Convolutional Layers)
# [Input Text (Numericalized)]
#          |
#          V
# [Embedding Layer (+ Dropout)]
#          |
#          V
# [Bidirectional GRU Layer]
#          |
#    /-----+-----\
#    V           V
# [Attention Layer]  [Permute + Parallel Conv1d(+ReLU)+MaxPool1d (for k in kernel_sizes)]
#    |                 |
#    V                 V
# [Weighted Sum]     [Concatenate Pooled Features]
#    |                 |
#    \-----+-----/
#          |
#          V
# [Concatenate Features]
#          |
#          V
# [Dropout Layer]
#          |
#          V
# [Linear Layer (Classifier)]
#          |
#          V
# [Output Logits]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from collections import Counter
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import cycle

from load_data import load_combined_data


SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

texts, labels = load_combined_data()

if len(texts) != len(labels):
    raise ValueError("Texts and labels must have the same length.")
if not all(0 <= lbl <= 5 for lbl in labels):
    raise ValueError("Labels must be integers between 0 and 5.")


def build_vocab(texts, min_freq=1):
    token_counts = Counter()
    for text in texts:
        token_counts.update(text.split())
    vocab = {token: idx + 2 for idx, (token, count) in enumerate(token_counts.items()) if count >= min_freq}
    vocab["<pad>"] = 0  
    vocab["<unk>"] = 1  
    return vocab

vocabulary = build_vocab(texts)
vocab_size = len(vocabulary)

def numericalize(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in text.split()]


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [numericalize(text, vocab) for text in texts]
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts_list, labels_list = zip(*batch)
    texts_padded = pad_sequence(texts_list, batch_first=True, padding_value=vocabulary["<pad>"])
    labels = torch.stack(labels_list)
    return texts_padded, labels



train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=SEED,
    stratify=labels if len(set(labels)) > 1 else None
)

train_dataset = TextDataset(train_texts, train_labels, vocabulary)
val_dataset = TextDataset(val_texts, val_labels, vocabulary)

BATCH_SIZE = 8 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        energy = torch.tanh(self.attn(x))  
        weights = F.softmax(energy, dim=1)  
        return weights

class GatedRecurrentConvolutionalNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_filters, kernel_sizes, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocabulary["<pad>"])
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(2 * hidden_dim, num_filters, ks, padding='same') for ks in kernel_sizes
        ])
        self.attn_gru = Attention(2 * hidden_dim)

        fc_input_dim = (2 * hidden_dim) + (len(kernel_sizes) * num_filters)
        self.fc = nn.Linear(fc_input_dim, num_classes) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        
        embedded = self.dropout(self.embedding(text))  

        
        gru_out, _ = self.gru(embedded)  
        attn_weights_gru = self.attn_gru(gru_out)  
        attended_gru = torch.sum(gru_out * attn_weights_gru, dim=1)  

        
        permuted_gru_out = gru_out.permute(0, 2, 1) 
        conved = [F.relu(conv(permuted_gru_out)) for conv in self.convs]
        pooled = [F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in conved]
        cat_pooled_convs = torch.cat(pooled, dim=1)

        combined_features = torch.cat((attended_gru, cat_pooled_convs), dim=1)
        dropped_features = self.dropout(combined_features)
        output_logits = self.fc(dropped_features) 
        
        
        return output_logits



def train(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    epoch_loss = 0
    all_preds_indices = []
    all_true_labels = []
    all_pred_probs = [] 

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device) 

        optimizer.zero_grad()
        predictions_logits = model(texts) 

        loss = criterion(predictions_logits, labels) 

        
        predicted_probs = F.softmax(predictions_logits, dim=1) 
        predicted_indices = torch.argmax(predicted_probs, dim=1) 

        all_preds_indices.extend(predicted_indices.cpu().detach().numpy())
        all_true_labels.extend(labels.cpu().detach().numpy())
        all_pred_probs.extend(predicted_probs.cpu().detach().numpy())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_true_labels, all_preds_indices)
    
    precision = precision_score(all_true_labels, all_preds_indices, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_preds_indices, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_preds_indices, average='macro', zero_division=0)

    auc = 0.0
    if num_classes > 1 and len(set(all_true_labels)) > 1 : 
        try:
            
            
            
            all_pred_probs_np = np.array(all_pred_probs)
            auc = roc_auc_score(all_true_labels, all_pred_probs_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Train AUC calculation error: {e}")
            auc = 0.0 

    return avg_loss, acc, precision, recall, f1, auc

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    epoch_loss = 0
    all_pred_probs_eval = [] 
    all_preds_indices_eval = [] 
    all_true_labels_eval = []

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions_logits = model(texts) 

            loss = criterion(predictions_logits, labels)
            epoch_loss += loss.item()

            predicted_probs = F.softmax(predictions_logits, dim=1) 
            predicted_indices = torch.argmax(predicted_probs, dim=1) 

            all_pred_probs_eval.extend(predicted_probs.cpu().numpy())
            all_preds_indices_eval.extend(predicted_indices.cpu().numpy())
            all_true_labels_eval.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_true_labels_eval, all_preds_indices_eval)
    precision = precision_score(all_true_labels_eval, all_preds_indices_eval, average='macro', zero_division=0)
    recall = recall_score(all_true_labels_eval, all_preds_indices_eval, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels_eval, all_preds_indices_eval, average='macro', zero_division=0)
    
    auc = 0.0
    
    all_pred_probs_eval_np = np.array(all_pred_probs_eval)
    if num_classes > 1 and len(set(all_true_labels_eval)) > 1 and all_pred_probs_eval_np.shape[0] > 0:
        try:
            auc = roc_auc_score(all_true_labels_eval, all_pred_probs_eval_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Validation AUC calculation error: {e}")
            auc = 0.0
            
    return avg_loss, acc, precision, recall, f1, auc, all_pred_probs_eval_np, all_true_labels_eval



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 6 
DROPOUT = 0.5
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30 


embedding_dim = 50
hidden_dim = 64
num_filters = 32
kernel_sizes = [2, 3, 4]

model1 = GatedRecurrentConvolutionalNetwork(
    vocab_size, embedding_dim, hidden_dim, num_filters, kernel_sizes, NUM_CLASSES, DROPOUT
).to(device)

optimizer1 = AdamW(model1.parameters(), lr=LEARNING_RATE)
criterion1 = nn.CrossEntropyLoss() 


train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_aucs, val_aucs = [], []

print(f"--- Training Model for {NUM_CLASSES} classes ---")
best_val_loss = float('inf')
val_probs_for_roc = None 
val_labels_for_roc = None 

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train(model1, train_loader, optimizer1, criterion1, device, NUM_CLASSES)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, current_val_probs, current_val_labels = evaluate(model1, val_loader, criterion1, device, NUM_CLASSES)

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
        val_probs_for_roc = current_val_probs
        val_labels_for_roc = current_val_labels
        

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC (macro): {train_auc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC (macro): {val_auc:.4f}")


epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(18, 12)) 

plt.subplot(2, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(epochs_range, train_precisions, label='Train Precision (Macro)')
plt.plot(epochs_range, val_precisions, label='Validation Precision (Macro)')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision Curves')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(epochs_range, train_recalls, label='Train Recall (Macro)')
plt.plot(epochs_range, val_recalls, label='Validation Recall (Macro)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Curves')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs_range, train_f1s, label='Train F1 Score (Macro)')
plt.plot(epochs_range, val_f1s, label='Validation F1 Score (Macro)')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Curves')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs_range, train_aucs, label='Train AUC (Macro OVR)')
plt.plot(epochs_range, val_aucs, label='Validation AUC (Macro OVR)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC Curves')
plt.legend()

plt.tight_layout()
plt.show()


if val_probs_for_roc is not None and val_labels_for_roc is not None and len(set(val_labels_for_roc)) > 1:
    
    y_true_binarized = label_binarize(val_labels_for_roc, classes=list(range(NUM_CLASSES)))
    n_classes = y_true_binarized.shape[1]

    if n_classes <= 1: 
        print("ROC curve cannot be plotted: Not enough classes in validation labels for binarization.")
    else:
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            if len(np.unique(y_true_binarized[:, i])) > 1: 
                 fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], val_probs_for_roc[:, i])
                 roc_auc[i] = roc_auc_score(y_true_binarized[:, i], val_probs_for_roc[:, i])
            else:
                 fpr[i], tpr[i], roc_auc[i] = None, None, None 


        plt.figure(figsize=(8, 7))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            if fpr[i] is not None and tpr[i] is not None:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
            else:
                plt.plot([0,1],[0,0], linestyle='--', color=color, label=f'ROC curve of class {i} (not available)')


        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC (Best Validation Epoch, {n_classes} classes)')
        plt.legend(loc="lower right")
        plt.show()

elif val_probs_for_roc is not None and val_labels_for_roc is not None:
    print("ROC curve cannot be plotted: only one class present in validation labels or other data issue.")
else:
    print("No validation predictions available for ROC curve.")

print(f"Finished training. Best validation loss: {best_val_loss:.4f}")
if val_labels_for_roc is not None:
    print(f"Number of unique labels in best validation set for ROC: {len(set(val_labels_for_roc))}")
    if val_probs_for_roc is not None:
        print(f"Shape of probabilities for ROC: {val_probs_for_roc.shape}")

