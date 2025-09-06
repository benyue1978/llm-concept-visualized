import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ==============================
# 配置
# ==============================
LAYER_ID = -1         # 可调：-12 ~ -1（负数代表从后往前数）
HEAD_MODE = 'mean'    # 'mean' | 'all'（'all' 返回 (num_heads, seq, seq) 便于进一步可视化）
TARGET = 'bank'

# 两个语境示例
SENT_RIVER = "He sat on the bank and watched the river flow"
SENT_FIN   = "She walked to the bank to deposit some money"

# ==============================
# 加载 BERT（输出注意力）
# ==============================
tok = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
bert.eval()


# ==============================
# 工具函数
# ==============================
def find_token_indices(tokens, target):
    """
    稳健定位目标词（考虑 WordPiece、重复出现）。
    这里对 'bank' 做精确匹配；如需合并子词，可在此扩展。
    """
    idxs = []
    for i, t in enumerate(tokens):
        if t == target:
            idxs.append(i)
    return idxs

def avg_attention(attentions, layer_id, head_mode='mean'):
    """
    attentions: List[num_layers] of (batch, num_heads, seq, seq)
    返回：'mean' -> (seq, seq)；否则 -> (num_heads, seq, seq)
    """
    att = attentions[layer_id][0]  # (num_heads, seq, seq)
    if head_mode == 'mean':
        return att.mean(0)         # (seq, seq)
    else:
        return att                  # (num_heads, seq, seq)

def visualize_target_attention(sentence, title, target=TARGET, layer_id=LAYER_ID, head_mode=HEAD_MODE):
    """
    可视化目标 token 对其他 token 的注意力行（均值头），或各头熵分布。
    """
    with torch.no_grad():
        enc = tok(sentence, return_tensors='pt')
        out = bert(**enc)

    tokens = tok.convert_ids_to_tokens(enc['input_ids'][0])
    att = avg_attention(out.attentions, layer_id, head_mode)

    # 选择要画的目标 token（如有多个，这里取第一个；也可遍历）
    idxs = find_token_indices(tokens, target)
    if not idxs:
        print(f"[WARN] target '{target}' not found in tokens:", tokens)
        return tokens, None
    t_idx = idxs[0]

    if head_mode == 'mean':
        row = att[t_idx]  # (seq,)
        # 过滤特殊符号仅用于显示（注意：注意力计算本身未改动）
        mask = [i for i, t in enumerate(tokens) if t not in ['[CLS]', '[SEP]']]
        show_tokens = [tokens[i] for i in mask]
        show_vals = row[mask].detach().cpu().numpy()

        plt.figure(figsize=(10, 2.2))
        plt.imshow(show_vals[np.newaxis, :], aspect='auto')
        plt.yticks([])
        plt.xticks(range(len(show_tokens)), show_tokens, rotation=80)
        plt.colorbar()
        plt.title(f"Attention from '{target}' (Layer {layer_id}) — {title}")
        plt.tight_layout()
        plt.show()
    else:
        # 展示每个头的注意力熵，定量对比头的“专注程度”
        probs = att[:, t_idx, :].detach().cpu().numpy()  # (heads, seq)
        probs = probs / (probs.sum(-1, keepdims=True) + 1e-12)
        ent = -(probs * (np.log(probs + 1e-12))).sum(-1)  # (heads,)

        plt.figure(figsize=(6, 3))
        plt.bar(range(len(ent)), ent)
        plt.title(f"Per-head attention entropy for '{target}' (Layer {layer_id}) — {title}")
        plt.xlabel("Head"); plt.ylabel("Entropy")
        plt.tight_layout(); plt.show()

    return tokens, out.last_hidden_state[0]  # (seq, hidden)

def get_hidden_for_token(sentence, token_text, which='first'):
    """
    取指定 token 的隐藏向量（最后一层）。which: 'first' | 'all'
    返回：向量 (hidden,) 或 list[(idx, vec)]
    """
    with torch.no_grad():
        enc = tok(sentence, return_tensors='pt')
        out = bert(**enc)

    tokens = tok.convert_ids_to_tokens(enc['input_ids'][0])
    idxs = find_token_indices(tokens, token_text)
    if not idxs:
        print(f"[WARN] '{token_text}' not found in tokens:", tokens)
        return None

    if which == 'first':
        return out.last_hidden_state[0, idxs[0]]  # (hidden,)
    else:
        return [(i, out.last_hidden_state[0, i]) for i in idxs]

def cos(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

def compare_semantics(sent1, sent2, target=TARGET):
    """
    同一目标词在不同语境的隐藏状态余弦相似度
    """
    h1 = get_hidden_for_token(sent1, target)
    h2 = get_hidden_for_token(sent2, target)
    if h1 is None or h2 is None:
        return None
    val = cos(h1, h2)
    print(f"Cosine similarity of '{target}' embeddings between contexts: {val:.3f}")
    return val


# ==============================
# 新增：三组对比并可视化
# ==============================
def trio_comparisons():
    """
    计算并展示：
      1) bank(river) vs river
      2) bank(finance) vs deposit
      3) bank(river) vs bank(finance)
    """
    # 1) bank(river)
    h_bank_river = get_hidden_for_token(SENT_RIVER, "bank")
    # 2) bank(finance)
    h_bank_fin   = get_hidden_for_token(SENT_FIN, "bank")
    # 3) river
    h_river      = get_hidden_for_token(SENT_RIVER, "river")
    # 4) deposit
    h_deposit    = get_hidden_for_token(SENT_FIN, "deposit")

    if any(v is None for v in [h_bank_river, h_bank_fin, h_river, h_deposit]):
        print("[WARN] Some tokens were not found; skip trio comparisons.")
        return

    c1 = cos(h_bank_river, h_river)   # bank(river) vs river
    c2 = cos(h_bank_fin,   h_deposit) # bank(fin)   vs deposit
    c3 = cos(h_bank_river, h_bank_fin)# bank(river) vs bank(fin)

    print(f"bank(river) vs river     : {c1:.3f}")
    print(f"bank(finance) vs deposit : {c2:.3f}")
    print(f"bank(river) vs bank(fin) : {c3:.3f}")

    # 条形图可视化
    labels = [
        "bank(river) vs river",
        "bank(fin) vs deposit",
        "bank(river) vs bank(fin)"
    ]
    vals = [c1, c2, c3]

    plt.figure(figsize=(7, 3.2))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=12, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Cosine similarity")
    plt.title("Contextualized Embedding Similarities (BERT)")
    plt.tight_layout()
    plt.show()


# ==============================
# 主流程演示
# ==============================
if __name__ == "__main__":
    # 1) 注意力可视化（均值头，目标词=bank）
    visualize_target_attention(SENT_RIVER, "River Context", target=TARGET, layer_id=LAYER_ID, head_mode='mean')
    visualize_target_attention(SENT_FIN,   "Finance Context", target=TARGET, layer_id=LAYER_ID, head_mode='mean')

    # 2) 同词跨语境的相似度（bank vs bank）
    compare_semantics(SENT_RIVER, SENT_FIN, target=TARGET)

    # 3) 三组对比（bank-river / bank-deposit / bank-bank）+ 可视化
    trio_comparisons()