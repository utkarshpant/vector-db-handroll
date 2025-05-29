from classes.vector_store import VectorStore
import numpy as np

vector_store = VectorStore()

sentences = ["Hi, my name is Utkarsh", "I am a software engineer",
             "I am interviewing with StackAI.", "Apples are my favorite fruit."]


# generate a vocabulary from our dataset
vocabulary = set()
for sentence in sentences:
    vocabulary.update(sentence.lower().strip('.,').split())

# for each word in the vocabulary, generate an index representing it in a vector
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}


def generate_vector(sentence):
    vector = np.zeros(len(word_to_idx), dtype=float)
    words = sentence.lower().strip('.,').split()
    for word in words:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1

    for i in range(len(words)):
        if words[i] not in word_to_idx:
            for vocab_word in word_to_idx:
                if words[i] in vocab_word or vocab_word in words[i]:
                    vector[word_to_idx[vocab_word]] += 0.5
    return vector


# Add vectors to the vector store
for sentence in sentences:
    vector = generate_vector(sentence)
    vector_store.add_vector(sentence, vector)

query = "My facourite fruit is the apple"

query_vector = generate_vector(query)
similar_vectors = vector_store.find_similar_vectors(query_vector, limit=2)
print("Query:", query, "\nQuery vector", query_vector, "\nSimilar Vectors:", similar_vectors)
