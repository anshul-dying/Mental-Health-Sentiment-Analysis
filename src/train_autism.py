import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Use PyTorch's AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import os
import matplotlib.pyplot as plt
import nlpaug.augmenter.word as naw

# Convert Autism Attributes to Text
def autism_to_text(row):
    questions = {
        'A1': 'maintains eye contact',
        'A2': 'shows repetitive behavior',
        'A3': 'follows instructions',
        'A4': 'uses gestures appropriately',
        'A5': 'responds to social cues',
        'A6': 'engages in imaginative play',
        'A7': 'communicates verbally',
        'A8': 'shows sensory sensitivities',
        'A9': 'adapts to changes',
        'A10': 'interacts socially'
    }
    text = []
    for q, desc in questions.items():
        score = int(row[q])
        if score == 1:
            text.append(f"does not {desc}")
        else:
            text.append(f"{desc}")
    age = row.get('Age', 'unknown')
    sex = row.get('Sex', 'unknown').lower()
    text.append(f"{age}-year-old {sex}")
    return ', '.join(text)

# Augment Text Data
def augment_texts(texts, labels, n_aug=2):
    aug = naw.SynonymAug()
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(n_aug):
            augmented_texts.append(aug.augment(text)[0])
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Custom Dataset
class AutismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Autism Classifier Model
class AutismClassifier(nn.Module):
    def __init__(self):
        super(AutismClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load Autism Dataset
def load_autism_dataset(path='D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/data/Autism_Screening_Data_Combined.csv'):
    df = pd.read_csv(path)
    df['text'] = df.apply(autism_to_text, axis=1)
    texts = df['text'].tolist() 
    labels = df['Class'].map({'YES': 1, 'NO': 0}).tolist()
    # Augment data
    augmented_texts, augmented_labels = augment_texts(texts, labels, n_aug=2)
    return augmented_texts, augmented_labels

# Plot Metrics
def plot_metrics(metrics, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['train'], label='Train')
    plt.plot(metrics['epoch'], metrics['val'], label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Train Model
def train_model(model, train_loader, val_loader, device, epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []}
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, train_preds, train_probs, train_labels = 0, [], [], []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            sigmoid_probs = torch.sigmoid(logits).detach()  # Detach before converting
            train_probs.extend(sigmoid_probs.cpu().numpy())
            train_preds.extend([1 if p > 0.5 else 0 for p in sigmoid_probs.cpu().numpy()])
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_probs)

        # Validation
        model.eval()
        val_loss, val_preds, val_probs, val_labels = 0, [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.squeeze(), labels)
                val_loss += loss.item()
                sigmoid_probs = torch.sigmoid(logits).detach()  # Detach before converting
                val_probs.extend(sigmoid_probs.cpu().numpy())
                val_preds.extend([1 if p > 0.5 else 0 for p in sigmoid_probs.cpu().numpy()])
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        # Log metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_f1)
        metrics['train_auc'].append(train_auc)
        metrics['val_auc'].append(val_auc)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/autism_model', exist_ok=True)
            torch.save(model.state_dict(), 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/autism_model/best_model.pth')

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs', exist_ok=True)
    metrics_df.to_csv('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/autism_metrics.csv', index=False)

    # Plot metrics
    os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures', exist_ok=True)
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_loss'], 'val': metrics['val_loss']},
                 'Autism Model Loss', 'Loss', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/autism_loss.png')
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_acc'], 'val': metrics['val_acc']},
                 'Autism Model Accuracy', 'Accuracy', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/autism_accuracy.png')
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_f1'], 'val': metrics['val_f1']},
                 'Autism Model F1-Score', 'F1-Score', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/autism_f1.png')
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_auc'], 'val': metrics['val_auc']},
                 'Autism Model AUC', 'AUC', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/autism_auc.png')

    return metrics

# Main Execution
if __name__ == '__main__':
    # Initialize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutismClassifier()

    # Load and split data
    texts, labels = load_autism_dataset()
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = AutismDataset(train_texts, train_labels, tokenizer)
    val_dataset = AutismDataset(val_texts, val_labels, tokenizer)
    test_dataset = AutismDataset(test_texts, test_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Train
    metrics = train_model(model, train_loader, val_loader, device, epochs=5)

    # Evaluate on test set
    model.load_state_dict(torch.load('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/autism_model/best_model.pth'))
    model.eval()
    preds, probs, true_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            sigmoid_probs = torch.sigmoid(logits).detach()
            probs.extend(sigmoid_probs.cpu().numpy())
            preds.extend([1 if p > 0.5 else 0 for p in sigmoid_probs.cpu().numpy()])
            true_labels.extend(labels.cpu().numpy())
    
    test_auc = roc_auc_score(true_labels, probs)
    test_f1 = f1_score(true_labels, preds)
    test_acc = accuracy_score(true_labels, preds)
    print(f'Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}')

    # Save test metrics
    test_metrics = pd.DataFrame({'test_auc': [test_auc], 'test_f1': [test_f1], 'test_acc': [test_acc]})
    test_metrics.to_csv('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/autism_test_metrics.csv', index=False)    