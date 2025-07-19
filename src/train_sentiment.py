import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt

# Custom Dataset
class EmotionDataset(Dataset):
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Emotion Classifier Model
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Load Emotion Dataset
def load_emotion_dataset(path='D:/EDAI Project/data/train_ISEAR.parquet'):
    df = pd.read_parquet(path)
    # Map emotion names to integers
    emotion_map = {
        'sadness': 0,
        'joy': 1,
        'love': 2,
        'anger': 3,
        'fear': 4,
        'surprise': 5
    }
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(emotion_map)
        df = df.dropna(subset=['label'])  # Remove unmapped labels
    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

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
def train_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, train_preds, train_labels = 0, [], []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Log metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_f1)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/sentiment_model', exist_ok=True)
            torch.save(model.state_dict(), 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/sentiment_model/best_model.pth')

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs', exist_ok=True)
    metrics_df.to_csv('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/emotion_metrics.csv', index=False)

    # Plot metrics
    os.makedirs('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures', exist_ok=True)
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_loss'], 'val': metrics['val_loss']},
                 'Emotion Model Loss', 'Loss', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/emotion_loss.png')
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_acc'], 'val': metrics['val_acc']},
                 'Emotion Model Accuracy', 'Accuracy', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/emotion_accuracy.png')
    plot_metrics({'epoch': metrics['epoch'], 'train': metrics['train_f1'], 'val': metrics['val_f1']},
                 'Emotion Model F1-Score', 'F1-Score', 'D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/figures/emotion_f1.png')

    return metrics

# Main Execution
if __name__ == '__main__':
    # Initialize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Device Selection
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device} ({torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'})")

    model = EmotionClassifier(n_classes=6)

    # Load and split data
    texts, labels = load_emotion_dataset()
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Train
    metrics = train_model(model, train_loader, val_loader, device, epochs=3)

    # Evaluate on test set
    model.load_state_dict(torch.load('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/sentiment_model/best_model.pth'))
    model = model.to(device)
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    test_f1 = f1_score(true_labels, preds, average='weighted')
    test_acc = accuracy_score(true_labels, preds)
    print(f'Test F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}')

    # Save test metrics
    test_metrics = pd.DataFrame({'test_f1': [test_f1], 'test_acc': [test_acc]})
    test_metrics.to_csv('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/docs/emotion_test_metrics.csv', index=False)
