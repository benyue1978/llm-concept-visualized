import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def topk_next(model_name, text, topk=5):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name)
    mdl.eval()
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = mdl(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    p, idx = torch.topk(probs, topk)
    toks = tok.convert_ids_to_tokens(idx.tolist())

    # 格式化输出，去掉 'Ġ'
    results = []
    for t, prob in zip(toks, p):
        token_clean = t.replace("Ġ", "")  # 去掉前导空格符
        results.append((token_clean, round(float(prob), 3)))
    return results

def demo():
    base = "gpt2"  # 可改: gpt2-medium / gpt2-large / gpt2-xl

    # 1) 原始陈述句（弱）
    print("\n[Raw] The capital of France is")
    for tok, prob in topk_next(base, "The capital of France is"):
        print(f"  {tok:<10} | {prob:.3f}")

    # 2) 问答样式（更好）
    print("\n[QA] Q: What is the capital of France?\\nA:")
    for tok, prob in topk_next(base, "Q: What is the capital of France?\nA:"):
        print(f"  {tok:<10} | {prob:.3f}")

    # 3) Few-shot 表格模式（最稳）
    fs = (
        "Country: Germany\nCapital: Berlin\n"
        "Country: Japan\nCapital: Tokyo\n"
        "Country: France\nCapital:"
    )
    print("\n[Few-shot] Country→Capital")
    for tok, prob in topk_next(base, fs):
        print(f"  {tok:<10} | {prob:.3f}")

    # 4) 银行语境：更像动作意图
    print("\n[Bank] She walked into the bank to")
    for tok, prob in topk_next(base, "She walked into the bank to"):
        print(f"  {tok:<10} | {prob:.3f}")

if __name__ == "__main__":
    demo()