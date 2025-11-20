import re
import numpy as np
import json
from search import search_query
from cacm_parser import parse_cacm_all


def parse_queries(path):
    queries = {}
    current_id = None
    text_buff = []

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('.I'):
                if current_id is not None:
                    queries[current_id] = " ".join(text_buff).strip()
                current_id = int(line.split()[1])
                text_buff = []
            elif line.startswith('.W'):
                continue
            else:
                text_buff.append(line)

        if current_id is not None:
            queries[current_id] = " ".join(text_buff).strip()
        
    return queries


def parse_query_results(path):
    query_res = {}
    with open (path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                quer_id = int(parts[0])
                doc_id = int(parts[1])
                query_res.setdefault(quer_id, set()).add(doc_id)
    return query_res


def average_precision(retreived, relevant):
    if not relevant: 
        return 0.0
    precisions = []
    retrieved_relevant_count = 0
    for i, doc_id in enumerate(retreived, start=1):
        if int(doc_id) in relevant:
            retrieved_relevant_count += 1
            precisions.append(retrieved_relevant_count / i)
    if not precisions:
        return 0.0
    return sum(precisions)/len(relevant)

def r_precision(retrieved, relevant):
    r = len(relevant)

    if r == 0: 
        return 0.0
    retrieved_at_r = retrieved[:r]
    count = 0
    for doc_id in retrieved_at_r:
        if int(doc_id) in relevant:
            count += 1
    return count/r
    

def evaluate_system(queries, qrels, doc_vectors, docs, idf, top_k=10, top_n=100):
    avgp_values = []
    rprec_values = []

    for qid, text in queries.items():
        relevant = qrels.get(qid, set())
        retrieved = [int(doc_id) for doc_id, _ in search_query(text, doc_vectors, docs, idf, top_k=top_k, top_n=top_n)]

        avgp = average_precision(retrieved, relevant)
        rprec = r_precision(retrieved, relevant)
        avgp_values.append(avgp)
        rprec_values.append(rprec)

        print(f"Query {qid}: AP = {avgp:.4f}, R-Precision = {rprec:.4f}") 
    
    print("\n***Overall Evaluation***")
    print(f"Mean Average Precision (MAP): {np.mean(avgp_values):.4f}")
    print(f"Average R-Precision: {np.mean(rprec_values):.4f}")
    print(f"Evaluated {len(queries)} queries.")


if __name__ == "__main__":

    docs = parse_cacm_all("cacm.all")

    with open("doc_vectors.json", encoding="utf-8") as f:
        doc_vectors = json.load(f)
    with open("idf.json", encoding="utf-8") as f:
        idf = json.load(f)

    queries = parse_queries("query.text")
    qrels = parse_query_results("qrels.text")

    print(f"Loaded {len(queries)} queries and {len(qrels)} relevance sets.")
    evaluate_system(queries, qrels, doc_vectors, docs, idf, top_k=10, top_n=100)