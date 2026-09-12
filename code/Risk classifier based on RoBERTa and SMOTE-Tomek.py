###Roberta+SMOTETemok
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.neighbors import NearestNeighbors
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import gc
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from math import ceil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random
import time
import os

print("1 模块导入成功")

# 设置全局随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

# ==================== 优化的模型定义 ====================

class Custom_Bert_Model(nn.Module):
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext'):
        super(Custom_Bert_Model, self).__init__()
        self.model_name = model_name
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True)

        # 冻结BERT模型的所有参数
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # 只解冻BERT模型的最后两层
        for param in self.bert_model.encoder.layer[-2:].parameters():  # 修改为解冻最后两层
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        final_hidden_state = outputs.hidden_states[-1]
        cls_hidden_state = final_hidden_state[:, 0, :]
        return final_hidden_state, cls_hidden_state


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # 增加隐藏层维度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(256)  # 添加LayerNorm

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


print("2 优化的模型定义成功")


# ==================== 优化的训练函数 ====================

def train_epoch(model, bert_model, loader, optimizer, scheduler, device, scaler, tokenizer, max_len=256):
    model.train()
    bert_model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc='Training', leave=False)

    for text, labels in progress:
        optimizer.zero_grad(set_to_none=True)
        labels = labels.to(device)

        # 预处理文本
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with autocast(enabled=torch.cuda.is_available()):
            _, text_hidden_embedding = bert_model(input_ids, attention_mask)
            outputs = model(text_hidden_embedding)
            loss = F.cross_entropy(outputs, labels.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / max(1, len(loader))


def evaluate(model, bert_model, loader, device, tokenizer, max_len=256, label_names=None):
    model.eval()
    bert_model.eval()
    preds, labels_all = [], []
    total_loss = 0.0
    progress = tqdm(loader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for text, labels in progress:
            labels = labels.to(device)
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            _, text_hidden_embedding = bert_model(input_ids, attention_mask)
            outputs = model(text_hidden_embedding)
            loss = F.cross_entropy(outputs, labels.long())

            total_loss += loss.item()
            y_pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            preds.extend(y_pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    # 综合评估指标
    macro_f1 = f1_score(labels_all, preds, average='macro')
    weighted_f1 = f1_score(labels_all, preds, average='weighted')
    acc = accuracy_score(labels_all, preds)
    balanced_acc = balanced_accuracy_score(labels_all, preds)
    avg_loss = total_loss / max(1, len(loader))

    print(f'Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Balanced Acc: {balanced_acc:.4f}')
    print(f'Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}')

    if label_names:
        print('\nClassification Report:')
        print(classification_report(labels_all, preds, target_names=label_names))

    return avg_loss, macro_f1, weighted_f1, acc, balanced_acc, labels_all, preds


print("3 优化的训练函数定义成功")

# ==================== 数据加载和预处理 ====================

# 读取Excel文件
excel_file = '长文本.xlsx'
xls = pd.ExcelFile(excel_file)

all_text_ls = []
all_label_ls = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df['text'] = df['text'].astype(str)
    text_ls = df['text'].tolist()
    label_ls = df['label'].tolist()

    for i in range(len(text_ls)):
        text_ls[i] = text_ls[i].strip("\n")

    all_text_ls += text_ls
    all_label_ls += label_ls

if len(all_label_ls) != len(all_text_ls):
    exit(0)

print("4 Excel文件读取成功")

# 数据集拆分
print("5 正在拆分数据集...")
train_val_text, test_text, train_val_labels, test_labels = train_test_split(
    all_text_ls, all_label_ls, test_size=0.2, random_state=42)

train_text, val_text, train_labels, val_labels = train_test_split(
    train_val_text, train_val_labels, test_size=0.25, random_state=42)

print(f"原始训练集大小: {len(train_text)}")
print(f"原始验证集大小: {len(val_text)}")
print(f"原始测试集大小: {len(test_text)}")

# SMOTE过采样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

print("正在加载 Sentence-BERT 模型...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)
sbert_model.eval()

def get_sbert_embeddings(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = sbert_model.encode(
                batch_texts,
                batch_size=batch_size,
                device=device,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.vstack(embeddings)

print("正在为训练集生成 SBERT 嵌入...")
X_train_sbert = get_sbert_embeddings(train_text)
y_train_labels = np.array(train_labels)

print("原始训练集数据分布：")
print(pd.Series(y_train_labels).value_counts())

print("正在进行 SMOTETomek 过采样...")
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled_sbert, y_train_resampled_labels = smote_tomek.fit_resample(
    X_train_sbert, y_train_labels
)

print("\n过采样后的训练集数据分布：")
print(pd.Series(y_train_resampled_labels).value_counts())

print("正在将过采样的向量映射回文本...")
nbrs = NearestNeighbors(n_neighbors=1).fit(X_train_sbert)
distances, indices = nbrs.kneighbors(X_train_resampled_sbert)
mapped_train_texts = [train_text[i] for i in indices.flatten()]

# 构建数据集
train_data_resampled = pd.DataFrame({'text': mapped_train_texts, 'label': y_train_resampled_labels})
val_data_original = pd.DataFrame({'text': val_text, 'label': val_labels})
test_data_original = pd.DataFrame({'text': test_text, 'label': test_labels})

train_data_resampled.to_csv('AP分级训练集3.0_SMOTE.csv', index=False, encoding='utf-8')
val_data_original.to_csv('AP分级验证集3.0.csv', index=False, encoding='utf-8')
test_data_original.to_csv('AP分级测试集3.0.csv', index=False, encoding='utf-8')

print("\n数据集已保存")
del X_train_sbert, X_train_resampled_sbert, sbert_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==================== 数据集定义 ====================

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, labels

print("6 混合采样完成")

# 创建数据加载器
train_dataset = CustomDataset(mapped_train_texts, y_train_resampled_labels)
valid_dataset = CustomDataset(val_text, val_labels)
test_dataset = CustomDataset(test_text, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("7 数据加载成功")

# ==================== 模型初始化和训练配置 ====================

# 超参数
model_name = 'hfl/chinese-roberta-wwm-ext'
input_size = 768
num_classes = 3
epochs = 30
batch_size = 16
lr = 2e-5
max_len = 256
warmup_ratio = 0.1
patience = 8  # 早停耐心值

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = Custom_Bert_Model(model_name).to(device)
model = SimpleNN(input_size, num_classes).to(device)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

# 优化器和学习率调度
optimizer = torch.optim.AdamW(
    [
        {'params': model.parameters(), 'lr': lr},
        {'params': bert_model.parameters(), 'lr': lr}
    ],
    weight_decay=0.01
)

num_training_steps = len(train_loader) * epochs
num_warmup_steps = int(num_training_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

scaler = GradScaler(enabled=torch.cuda.is_available())

print("8 模型初始化完成，开始训练...")

# ==================== 修复的训练循环 ====================

# 修改1：早停策略改为最小的Weighted-F1
best_val_loss = float('inf')
no_improve = 0
best_model_path = 'best_APEV-BERT2.0_optimized.pth'
best_bert_model_path = 'best_APEV-BERT2.0_bert_optimized.pth'

label_names = [str(i) for i in range(num_classes)]

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # 训练
    train_loss = train_epoch(model, bert_model, train_loader, optimizer, scheduler,
                             device, scaler, bert_tokenizer, max_len)
    print(f'Train Loss: {train_loss:.4f}')

    # 验证
    print('Validation:')
    val_loss, macro_f1, weighted_f1, acc, balanced_acc, _, _ = evaluate(
        model, bert_model, valid_loader, device, bert_tokenizer, max_len, label_names
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        torch.save(bert_model.state_dict(), best_bert_model_path)
        print(f'✅ Saved best model. val_loss: {val_loss:.4f}')
        no_improve = 0
    else:
        no_improve += 1
        print(f'No improvement for {no_improve}/{patience} epoch(s)')
        if no_improve >= patience:
            print(f'🚨 Early stopping triggered after {epoch + 1} epochs!')
            break

# 如果正常完成所有epochs
if no_improve < patience:
    print(f"✅ Training completed all {epochs} epochs without early stopping.")

print("9 模型训练完毕")

# ==================== 测试和评估 ====================

print("10 开始测试评估...")

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path, weights_only=True))
bert_model.load_state_dict(torch.load(best_bert_model_path, weights_only=True))

model.eval()
bert_model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for text, labels in test_loader:
        labels = labels.to(device)
        inputs = bert_tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        _, text_hidden_embedding = bert_model(input_ids, attention_mask)
        outputs = model(text_hidden_embedding)
        probs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 综合评估指标
accuracy = accuracy_score(all_labels, all_preds)
weighted_precision = precision_score(all_labels, all_preds, average='weighted')
weighted_recall = recall_score(all_labels, all_preds, average='weighted')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
macro_precision = precision_score(all_labels, all_preds, average='macro')
macro_recall = recall_score(all_labels, all_preds, average='macro')
macro_f1 = f1_score(all_labels, all_preds, average='macro')
class_precision = precision_score(all_labels, all_preds, average=None)
class_recall = recall_score(all_labels, all_preds, average=None)
class_f1 = f1_score(all_labels, all_preds, average=None)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)

print("--- 基础性能指标 ---")
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_acc:.4f}')
print(f'Macro F1: {macro_f1:.4f}')
print(f'Weighted F1: {weighted_f1:.4f}')

print("\n--- 详细分类报告 ---")
classification_rep = classification_report(all_labels, all_preds, digits=4)
print(classification_rep)

# 修改3：保存分类报告到文件
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("分类报告\n")
    f.write("========\n\n")
    f.write(classification_rep)
    f.write(f"\n准确率 (Accuracy): {accuracy:.4f}")
    f.write(f"\n加权F1 (Weighted F1): {weighted_f1:.4f}")
    f.write(f"\n宏平均F1 (Macro F1): {macro_f1:.4f}")
    f.write(f"\n平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")
    f.write(f"\n加权准确率 (weighted_precision): {weighted_precision:.4f}")
    f.write(f"\n加权召回率 (weighted_recall): {weighted_recall:.4f}")
    f.write(f"\n加权F1 (weighted_f1): {weighted_f1:.4f}")
    f.write(f"\n宏准确率 (macro_precision): {macro_precision:.4f}")
    f.write(f"\n宏召回率 (macro_recall): {macro_recall:.4f}")
    f.write(f"\n宏F1 (macro_f1): {macro_f1:.4f}")

print("分类报告已保存到 classification_report.txt")

# 可视化结果
conf_matrix = confusion_matrix(all_labels, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix (Counts)')

conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Confusion Matrix (Normalized)')

# 修改3：保存混淆矩阵图像
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 confusion_matrix.png")

# --- 多分类 ROC 曲线 (One-vs-Rest) ---
n_classes = 3
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])  # shape (n_samples, 3)

plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']

# 存储每个类别的 AUC
roc_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[f'Class {i}'] = roc_auc
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Class {i} vs Rest (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线基线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(True)

# 修改3：保存ROC曲线图像
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC曲线已保存到 roc_curve.png")

plt.tight_layout()
plt.show()

print("11 任务完成")
