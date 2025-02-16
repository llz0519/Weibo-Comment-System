# 微博评论情感分析系统：基于BERT的文本分类
本项目实现了自动判断评论的情感方向是正面还是负面，调用了本地hfl-rbt3模型，利用了GPU加速处理
## 1.项目背景
情感分析是自然语言处理（NLP）中的一个重要任务，它的目标是通过分析文本，判断其中所表达的情感是积极、消极还是中立。在微博评论中，情感的判断有助于品牌营销、舆情监控等应用。
我们使用了BERT模型（Bidirectional Encoder Representations from Transformers）进行情感分类。BERT作为一种预训练的语言模型，具有出色的语境理解能力，能够为情感分析任务提供强大的支持。
## 2.项目概述
本项目使用了微博评论数据集，采用BERT模型对评论文本进行分类，预测评论的情感类别（积极/消极）。我们使用了Hugging Face的transformers库来加载BERT模型，借助PyTorch框架进行模型训练和评估。
## 3.数据预处理
首先，我们从本地读取了包含微博评论和情感标签的数据集。数据集格式为Excel文件，包含评论文本和标签（0表示消极，1表示积极）。我们将数据集分为训练集和测试集，80%的数据用于训练，20%的数据用于测试。
```python
data_path = r'C:\Users\28903\Desktop\weibo_senti_100k.xlsx'
data = pd.read_excel(data_path)
```
## 4. 定义数据集类
为了能将数据输入到BERT模型中，我们需要对文本进行分词处理，并将其转换为BERT所需的输入格式。我们使用了BertTokenizer进行分词和编码，将评论文本转化为input_ids、attention_mask等张量。

我们定义了一个WeiboDataset类来处理数据集：
```python
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
```
## 5. 加载BERT模型和训练设置
我们使用了Hugging Face的transformers库加载了hfl-rbt3（中文BERT模型）作为预训练模型。然后，我们设置了优化器AdamW，并将模型迁移到GPU上进行加速训练。
```python
model = AutoModelForSequenceClassification.from_pretrained('hfl-rbt3', num_labels=2)
model = model.to('cuda')  # 如果没有GPU，改成 'cpu'

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```
## 6. 训练模型
在训练过程中，我们通过批次加载数据，并将每个批次输入到BERT模型中进行训练。每个epoch训练结束后，我们输出当前的损失值，帮助我们评估训练进展。

```python
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
```
## 7. 模型评估
训练完成后，我们使用测试集评估模型的表现。我们计算了模型的准确率和分类报告（包括精确度、召回率和F1分数）。分类报告帮助我们更全面地了解模型在不同情感类别下的表现。
``` python
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
```
## 8. 结果与分析
```yaml
Accuracy: 0.85
Classification Report:
               precision    recall  f1-score   support

            0       0.96      0.99      0.98      11844
            1       0.99      0.96      0.98      12154

    accuracy                           0.98      8000
   macro avg       0.98      0.98      0.98      8000
weighted avg       0.98      0.98      0.98      8000
```
可以看到，模型在微博评论情感分类任务上达到了98%的准确率，表现良好。

## 9.模型简介
hfl-rbt3 是由哈工大（HIT）自然语言处理实验室（HFL）推出的一个中文BERT模型，属于 RoBERTa（A Robustly Optimized BERT Pretraining Approach）的变体。
RoBERTa 是在 BERT 基础上进行改进的模型，主要通过以下几个方面进行优化：

·移除了 BERT 中的 Next Sentence Prediction（NSP）任务；
·增加了训练的批量大小（batch size）和训练数据的量；
·对训练过程中的超参数进行了微调。


在今天的项目中，hfl-rbt3 被用作预训练模型，来对微博评论进行情感分类任务。这个模型在中文文本处理方面表现较好，适用于文本分类、情感分析、命名实体识别等任务。

此模型为本地模型，有需要的可以联系作者，另外不会实现GPU加速的也欢迎咨询。
本人目前在读本科，有任何问题欢迎探讨，VX：15735002648
