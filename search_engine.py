import json

def load_inverted_index(file_path):
    """Φόρτωση του ανεστραμμένου ευρετηρίου από αρχείο JSON με κανονικοποίηση."""
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_index = json.load(file)
    # Κανονικοποίηση όλων των κλειδιών του ευρετηρίου σε πεζά
    normalized_index = {key.lower(): value for key, value in raw_index.items()}
    return normalized_index

def search_in_index(query, inverted_index):
    """Αναζήτηση ερωτήματος στο ανεστραμμένο ευρετήριο."""
    query = query.lower().strip()  # Κανονικοποίηση του ερωτήματος
    results = inverted_index.get(query, [])  # Αναζήτηση στο ευρετήριο
    return results

def simple_search_engine():
    """Απλή διεπαφή χρήστη για αναζήτηση."""
    inverted_index = load_inverted_index('inverted_index.json')  # Φόρτωση ευρετηρίου

    print("Μηχανή Αναζήτησης")
    print("Αναζητήστε οτιδήποτε ή πληκτρολογήστε 'exit' για έξοδο.")
    
    while True:
        query = input("\nΕισάγετε το ερώτημά σας: ").strip()
        if query.lower() == "exit":
            print("Έξοδος από τη μηχανή αναζήτησης...")
            break

        results = search_in_index(query, inverted_index)
        
        if results:
            print(f"Βρέθηκαν {len(results)} αποτελέσματα για το '{query}':")
            for result in results:
                print(f"- {result}")
        else:
            print(f"Δε βρέθηκαν αποτελέσματα για το '{query}'.")

if __name__ == "__main__":
    simple_search_engine()
