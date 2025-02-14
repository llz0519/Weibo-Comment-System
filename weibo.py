import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
data_path = r'C:\Users\28903\Desktop\weibo_senti_100k.xlsx'
data = pd.read_excel(data_path)

# 查看数据
print(data.head())

# 分离特征和标签
X = data['review'].values
y = data['label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义数据集类
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('hfl-rbt3')
MAX_LENGTH = 128

# 创建数据集和数据加载器
train_dataset = WeiboDataset(X_train, y_train, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = WeiboDataset(X_test, y_test, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained('hfl-rbt3', num_labels=2)
model = model.to('cuda')  # 改成 'cpu' 如果没有使用 GPU

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


# 训练模型
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


# 评估
predictions, true_labels = evaluate_model(model, test_loader)

# 计算准确率和报告
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)