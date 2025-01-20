import json
from text_processing import preprocess_text, preprocess_dataset, save_json_inline
from inverted_index import inverted_index
from search_engine import load_inverted_index, search_in_index
from query_processing import process_query, process_boolean_query
from ranking import compute_tf_idf, compute_vsm, compute_bm25, word_to_docs
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Βήμα 1: Προεπεξεργασία dataset
print("Step 1: Preprocessing Dataset")
input_file = 'goalscorers.json'
output_file = 'goalscorers_processed.json'

data = preprocess_dataset(json.load(open(input_file, 'r', encoding='utf-8')))
save_json_inline(data, output_file)
print(f"Dataset processed and saved to {output_file}")

# Βήμα 2: Δημιουργία και αποθήκευση αντεστραμμένου ευρετηρίου
print("Step 2: Building Inverted Index")
index = inverted_index
for idx, record in enumerate(data):
    scorer = record["scorer"].lower()
    index[scorer].append(record)

with open("inverted_index.json", "w", encoding="utf-8") as outfile:
    json.dump(index, outfile, ensure_ascii=False, indent=4)
print("Inverted index built and saved to inverted_index.json")

# Βήμα 3: Επεξεργασία ερωτήματος και αναζήτηση
print("Step 3: Query Processing and Search")
inverted_index = load_inverted_index("inverted_index.json")
query = "angelos charisteas"

# Κανονικοποίηση ερωτήματος
normalized_query = query.lower().strip()
results = search_in_index(normalized_query, inverted_index)
print(f"Results for query '{query}': {results}")

# Βήμα 4: Αποτελέσματα κατάταξης
print("Step 4: Ranking Results")

ranked_results_tfidf = compute_tf_idf(normalized_query, inverted_index, word_to_docs)
print("TF-IDF Ranking Results:", ranked_results_tfidf)

ranked_results_vsm = compute_vsm(normalized_query, inverted_index, word_to_docs)
print("VSM Ranking Results:", ranked_results_vsm)

ranked_results_bm25 = compute_bm25(normalized_query, inverted_index, word_to_docs)
print("BM25 Ranking Results:", ranked_results_bm25)

# Βήμα 5: Αξιολόγηση συστήματος
print("Step 5: Evaluation")
ground_truth = [
    {
        "query": "angelos charisteas",
        "relevant_docs": ["Angelos Charisteas", "Angelos Charisteas"]
    }
]

evaluations = []
for item in ground_truth:
    query = item["query"]
    relevant_docs = set(item["relevant_docs"])

    results = search_in_index(query, inverted_index)
    retrieved_docs = [doc['scorer'] for doc in results]

    # Μετρικές
    y_true = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
    y_pred = [1] * len(retrieved_docs)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    evaluations.append({
        "query": query,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# Εκτύπωση και αποθήκευση αξιολογήσεων
for eval_result in evaluations:
    print(f"Evaluation for query '{eval_result['query']}': Precision={eval_result['precision']:.2f}, Recall={eval_result['recall']:.2f}, F1-Score={eval_result['f1']:.2f}")

with open("evaluation_results.json", "w", encoding="utf-8") as eval_file:
    json.dump(evaluations, eval_file, indent=4)
print("Evaluation results saved to evaluation_results.json")
