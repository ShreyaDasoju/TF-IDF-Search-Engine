import argparse
import json
import math
from nltk.stem import PorterStemmer
from cacm_parser import parse_cacm_all, load_stopwords, tokenize

#length normalization of both doc and query vectors
def normalize_vectors(vector):
    sum_of_squares = 0
    for weight in vector.values():
        sum_of_squares += weight **2
    norm = math.sqrt(sum_of_squares)

    # preventing division by zero
    if norm ==0: 
        return vector
    
    normalized_vector = {}
    for term, weight in vector.items():
        normalized_vector[term]= weight / norm

    return normalized_vector




#main component
#Index Elimination is the Top-K approach chosen and is implemented here. The idf-threshold is 0.1 by default.
#Subset of N=100 documents is now taken from the filtered documents, and then K=10 documents are returned from that subset, in ranked order
#by cosine similarity score
def search_query(query_text, doc_vectors, docs, idf, stopwords=set(), stemmer=None, idf_threshold=0.1, top_k=10, top_n=100):
    tokens = tokenize(query_text)
    processed_terms = []

    for term in tokens:
        t = term.lower()
        if stopwords and t in stopwords:
            continue
        if stemmer:
            t = stemmer.stem(t)

        if idf.get(t, 0) < idf_threshold:
            continue
        processed_terms.append(t)
    
    if not processed_terms:
        return []
    
    #Building the query vector
    query_tf = {}
    for t in processed_terms:
        query_tf[t]= query_tf.get(t,0)+1

    query_vector = {}
    for t, tf in query_tf.items():  
        idf_val = idf.get(t, 0)
        query_vector[t] = (1+math.log10(tf))*idf_val

    query_vector = normalize_vectors(query_vector)

    # Index elimination: only scoring the docs that contain query terms
    candidate_docs = set()
    for term in processed_terms:
        for doc_id, vec in doc_vectors.items():
            if term in vec:
                candidate_docs.add(doc_id)

    if len(candidate_docs) > top_n:
        pre_scores = {}
        for doc_id in candidate_docs:
            vec = doc_vectors[doc_id]
            sum_score = 0
            for t in processed_terms:
                sum_score += vec.get(t, 0)
            pre_scores[doc_id]= sum_score
        sorted_pre_scores = sorted(pre_scores.items(), key=lambda x: x[1], reverse=True)
        top_subset = []
        for i, (doc_id, sc) in enumerate(sorted_pre_scores):
            if i < top_n:
                top_subset.append(doc_id)
            else:
                break
        candidate_docs = set(top_subset)

    #compute cosine similarity for the remaining candidate documents
    scores = {}
    for doc_id in candidate_docs:
        vec = doc_vectors[doc_id]
        sum_score = 0
        for t in query_vector:
            sum_score += query_vector[t] * vec.get(t, 0)
        scores[doc_id]= sum_score

    # only returning the top k documents in ranked order
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]

#removed main program from here and moved to searchUI.py