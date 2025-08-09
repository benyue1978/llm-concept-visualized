
# 🧠 理解大语言模型的原理与核心机制

本项目通过三个示例，逐步展示了大语言模型（如BERT、GPT等）在词义建模、上下文感知与生成预测方面的核心原理：

1. **Embedding（词向量）**：将词语映射到一个高维空间，保留语义结构（方向性语义）
2. **Self-Attention（自注意力）**：通过上下文动态分配注意力权重，理解多义词与句意
3. **Output Head（预测输出）**：对 `[MASK]` 等位置进行概率分布预测，模拟补全和回答问题

---

## 📦 环境准备

```bash
pip install -r requirements.txt
```

---

## ✅ 示例一：Embedding 表示语义

- **文件**：`word_embedding.py`  
- **模型**：word2vec-google-news-300  
- **功能**：演示词向量空间中的语义计算

```python
# 使用词向量做简单的代数计算
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)
```

🧠 *这是早期大模型中常用的词向量表示形式，展示词之间的“方向性语义”。*

---

## ✅ 示例二：Self-Attention 表示多义词上下文

- **文件**：`self_attention.py`  
- **模型**：bert-base-uncased  
- **功能**：展示 Transformer 中 self-attention 的可视化

两个句子：

- River 语境: `He sat on the bank and watched the river flow.`
- Finance 语境: `She walked to the bank to deposit some money.`

🎯 *我们展示了 “bank” 这个词如何根据上下文对“river”或“money”分配更多注意力，从而实现多义词 disambiguation。*

---

## ✅ 示例三：Output Head 模拟语言生成

- **文件**：`mask_prediction.py`  
- **模型**：bert-base-uncased  
- **功能**：对 `[MASK]` 位置生成概率最高的预测词

示例：

```python
# 输入句子
text = "The capital of France is [MASK]."

# 输出 top 5 候选词：
# 1. Paris
# 2. lyon
# 3. france
# ...
```

📌 *这展示了语言模型如何根据上下文推理并进行语言补全。*

---

## 🧪 小结

| 示例 | 展示核心 | 模型层次 | 技术关键词 |
|------|-----------|-----------|-------------|
| 词向量 | 语义结构 | 静态向量 | Word2Vec, Embedding |
| Attention | 多义词感知 | Transformer层 | Self-Attention, Heatmap |
| 输出预测 | Mask填充 | Output head | Masked LM, BERT Output |
