import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns

# 配置
model_name = 'bert-base-uncased'
layer_id = 11  # 可改为 -1 观察最后一层
target_word = 'bank'

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
model.eval()

def visualize_attention(sentence, title_suffix):
    # 编码输入
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 获取 attention
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # List[12] of shape (1, 12, seq_len, seq_len)

    # 平均指定层的所有头
    avg_attention = attentions[layer_id][0].mean(dim=0)

    # 过滤非特殊 token
    valid_indices = [i for i, t in enumerate(tokens) if t not in ['[CLS]', '[SEP]', '.']]
    filtered_tokens = [tokens[i] for i in valid_indices]
    filtered_attention = avg_attention[valid_indices][:, valid_indices]

    # # 可视化 attention heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(filtered_attention, xticklabels=filtered_tokens, yticklabels=filtered_tokens, cmap="viridis")
    # plt.title(f"Attention Heatmap (Layer {layer_id}, Avg Heads): {title_suffix}")
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()

    # 可视化特定词的注意力行
    if target_word in tokens:
        target_index = tokens.index(target_word)
        bank_row = avg_attention[target_index][valid_indices]

        plt.figure(figsize=(10, 2))
        sns.heatmap(bank_row.unsqueeze(0), xticklabels=filtered_tokens, cmap="viridis", cbar=True)
        plt.title(f"Attention from '{target_word}' to Others (Layer {layer_id}): {title_suffix}")
        plt.xticks(rotation=90)
        plt.yticks([])
        plt.tight_layout()
        plt.show()

# 🔁 输入两个句子
sentence1 = "He sat on the bank and watched the river flow."
sentence2 = "She walked to the bank to deposit some money."

# 🔍 分别可视化
visualize_attention(sentence1, "River Context")
visualize_attention(sentence2, "Finance Context")