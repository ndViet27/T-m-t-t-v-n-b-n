import pickle
import numpy as np
import networkx as nx
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import nltk
nltk.download('puntk_tab')

# Load dữ liệu
with open ('neg.pkl', 'rb') as fp:
    contents = pickle.load(fp)
with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())

# Pre-processing
contents_parsed = [content.lower().strip() for content in contents[:10]]

# Tách câu và từ
sentences = nltk.sent_tokenize(contents_parsed[3])  # Dùng pyvi để tách câu tiếng Việt
processed_sentences = []
for sentence in sentences:
    words = [word.replace("_", " ") for word in ViTokenizer.tokenize(sentence).split() if word not in stopwords]
    processed_sentences.append(" ".join(words))

# In danh sách câu đã tiền xử lý
print(contents_parsed[3])

# Tải mô hình word2vec
w2v = KeyedVectors.load_word2vec_format("vi.vec", binary=False)
vocab = w2v.key_to_index

# Hàm tính vector đại diện cho câu (bằng cách lấy trung bình các vector của các từ trong câu)
def sentence_to_vector(sentence, word_vectors):
    words = sentence.split()  # Tách từ trong câu
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    
    if len(vectors) == 0:
        return np.zeros_like(list(word_vectors.values())[0])
    
    return np.mean(vectors, axis=0)

# Tạo ma trận tương đồng
def build_similarity_matrix(sentences, word_vectors):
    sentence_vectors = [sentence_to_vector(sentence, word_vectors) for sentence in sentences]
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix

# Tạo ma trận tương đồng
similarity_matrix = build_similarity_matrix(processed_sentences, w2v)

# Xây dựng đồ thị từ ma trận tương đồng
graph = nx.from_numpy_array(similarity_matrix)

# Áp dụng thuật toán PageRank
scores = nx.pagerank(graph)

# Lấy các câu quan trọng nhất
num_sentences =  int(len(sentences)*0.5) # Chọn số câu tóm tắt bạn muốn
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])

# In kết quả tóm tắt
print("Tóm tắt văn bản:")
print(summary)


