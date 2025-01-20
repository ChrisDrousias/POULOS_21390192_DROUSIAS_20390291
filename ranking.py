import json
import re
import unicodedata
from collections import defaultdict, Counter
import math

# Φόρτωση του ανεστραμμένου ευρετηρίου
with open('inverted_index.json', 'r', encoding='utf-8') as f:
    raw_inverted_index = json.load(f)

# Κανονικοποίηση κλειδιών ευρετηρίου και κατακερματισμός σε λέξεις
inverted_index = defaultdict(list)
word_to_docs = defaultdict(set)

for key, docs in raw_inverted_index.items():
    normalized_key = unicodedata.normalize('NFKD', key.strip().lower())
    inverted_index[normalized_key] = docs

    for word in normalized_key.split():
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict) and "match" in doc:
                    word_to_docs[word].add((doc["match"], doc.get("minute", "N/A")))
                elif not isinstance(doc, dict):
                    word_to_docs[word].add((doc, "N/A"))
        elif isinstance(docs, dict) and "match" in docs:
            word_to_docs[word].add((docs["match"], docs.get("minute", "N/A")))
        elif not isinstance(docs, dict):
            word_to_docs[word].add((docs, "N/A"))

# Υπολογισμός TF-IDF
def compute_tf_idf(query, inverted_index, word_to_docs):
    """Υπολογισμός TF-IDF για την κατάταξη αποτελεσμάτων."""
    total_docs = sum(len(docs) if isinstance(docs, list) else 1 for docs in inverted_index.values())
    normalized_query = unicodedata.normalize('NFKD', query.strip().lower())

    scores = defaultdict(float)
    for term in normalized_query.split():
        if term in word_to_docs:
            doc_list = word_to_docs[term]
            doc_frequency = len(doc_list)
            idf = math.log(total_docs / (1 + doc_frequency))

            for doc_id, minute in doc_list:
                scores[(doc_id, minute)] += idf

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results

# Vector Space Model (VSM)
def compute_vsm(query, inverted_index, word_to_docs):
    """Υπολογισμός VSM για την κατάταξη αποτελεσμάτων."""
    query_tokens = query.lower().strip().split()
    query_vector = Counter(query_tokens)

    doc_vectors = defaultdict(Counter)
    for term in query_tokens:
        if term in word_to_docs:
            for doc_id, minute in word_to_docs[term]:
                doc_vectors[(doc_id, minute)][term] += 1

    scores = {}
    for doc, vector in doc_vectors.items():
        dot_product = sum(query_vector[t] * vector[t] for t in query_tokens)
        query_norm = math.sqrt(sum(q ** 2 for q in query_vector.values()))
        doc_norm = math.sqrt(sum(v ** 2 for v in vector.values()))
        scores[doc] = dot_product / (query_norm * doc_norm) if query_norm and doc_norm else 0

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results

# BM25
def compute_bm25(query, inverted_index, word_to_docs, k1=1.5, b=0.75):
    """Υπολογισμός BM25 για την κατάταξη αποτελεσμάτων."""
    total_docs = sum(len(docs) if isinstance(docs, list) else 1 for docs in inverted_index.values())
    avg_doc_len = sum(len(docs) for docs in inverted_index.values()) / total_docs

    scores = defaultdict(float)
    query_tokens = query.lower().strip().split()

    for term in query_tokens:
        if term in word_to_docs:
            doc_list = word_to_docs[term]
            doc_frequency = len(doc_list)
            idf = math.log((total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)

            for doc_id, minute in doc_list:
                doc_len = len(doc_list)
                tf = 1
                scores[(doc_id, minute)] += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results

# Διεπαφή χρήστη για αναζήτηση
def query_interface():
    print("Μηχανή Αναζήτησης με Πολλαπλούς Αλγορίθμους Κατάταξης")
    print("Επιλέξτε Αλγόριθμο: 1) TF-IDF  2) VSM  3) BM25")
    while True:
        choice = input("Επιλογή: ")
        if choice not in {"1", "2", "3"}:
            print("Μη έγκυρη επιλογή. Προσπαθήστε ξανά.")
            continue

        query = input("Ερώτημα: ")
        if query.lower() == "exit":
            break

        if choice == "1":
            ranked_results = compute_tf_idf(query, inverted_index, word_to_docs)
        elif choice == "2":
            ranked_results = compute_vsm(query, inverted_index, word_to_docs)
        elif choice == "3":
            ranked_results = compute_bm25(query, inverted_index, word_to_docs)

        if ranked_results:
            print("Αποτελέσματα κατάταξης:")
            for rank, ((doc_id, minute), score) in enumerate(ranked_results, start=1):
                print(f"{rank}. {doc_id} (Minute: {minute}) (Score: {score:.4f})")
        else:
            print("Δεν βρέθηκαν αποτελέσματα για το ερώτημά σας.")

if __name__ == "__main__":
    query_interface()
