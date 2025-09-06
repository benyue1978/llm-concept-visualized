# gpt_generate_simple.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(
    model_name="gpt2",
    prompt="Once upon a time",
    steps=50,
    greedy=False,
    temperature=1.0,
    top_p=1.0,
):
    """
    两种解码模式：
    - greedy=True: 贪婪解码，每次取概率最高的 token，最稳定，但容易重复。
    - greedy=False: 采样解码，结合 temperature 和 top_p。
        * temperature < 1.0 → 更保守，减少随机性。
        * temperature > 1.0 → 更随意，更有创造性。
        * top_p (nucleus sampling) → 只在累计概率 <= top_p 的词里采样。
          常用范围 0.8~0.95，越小越保守，越大越多样。
    """

    tok = GPT2Tokenizer.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name)
    mdl.eval()

    input_ids = tok(prompt, return_tensors="pt").input_ids

    for _ in range(steps):
        with torch.no_grad():
            logits = mdl(input_ids).logits[0, -1]

        if greedy:
            # 贪婪：直接取概率最大值
            next_id = torch.argmax(logits).unsqueeze(0)
        else:
            # temperature 缩放
            scaled = logits / max(temperature, 1e-8)
            probs = torch.softmax(scaled, dim=-1)

            # nucleus sampling (top-p)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            if keep.sum() == 0:
                keep[0] = True
            truncated = sorted_probs * keep
            truncated /= truncated.sum()
            pick = torch.multinomial(truncated, 1)
            next_id = sorted_idx[pick]

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    text = tok.decode(input_ids[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    prompt = "Once upon a time"

    print("\n=== Greedy (贪婪解码) ===")
    print(generate_text("gpt2", prompt, steps=50, greedy=True))

    print("\n=== Sampling (temperature=0.9, top_p=0.95) ===")
    print(generate_text("gpt2", prompt, steps=50, greedy=False, temperature=0.9, top_p=0.95))