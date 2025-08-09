from gensim.models import KeyedVectors

# 下载模型（首次运行时）
# 这个模型约为 1.5GB，会自动下载并缓存
from gensim.downloader import load
model = load("word2vec-google-news-300")  # Google News 300维词向量

# King - Man + Woman
try:
    result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print("king - man + woman ≈", result[0])
except KeyError as e:
    print(f"King-Man+Woman example failed: {e}")

# Paris - France + Germany
try:
    result = model.most_similar(positive=["Paris", "Germany"], negative=["France"], topn=1)
    print("Paris - France + Germany ≈", result[0])
except KeyError as e:
    print(f"Country-Capital example failed: {e}")
