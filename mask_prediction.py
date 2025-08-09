from transformers import BertTokenizer, BertForMaskedLM
import torch

# 载入模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# 输入句子
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors='pt')
mask_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

# 模型输出
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 获取MASK位置的预测结果
mask_logits = logits[0, mask_index.item()]
top_k = torch.topk(mask_logits, k=5)
top_tokens = [tokenizer.decode([idx]) for idx in top_k.indices]

print("Top-5 predictions for [MASK]:")
for i, word in enumerate(top_tokens):
    print(f"{i+1}. {word} (score={top_k.values[i].item():.4f})")