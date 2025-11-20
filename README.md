This system supports index construction, vector-based document retrieval, and performance evaluation using IR metrics (MAP and R-Precision)

Programs are written in the following:
    - cacm_parser.py 
    - search.py
    - searchUI.py
    - eval.py
    - test.py 

----------------------------------------------------------------------------------
Instructions to run the programs:
----------------------------------------------------------------------------------
cacm_parser.py (building the index):
    - python cacm_parser.py --cacm cacm.all --stopwords stopwords.txt --no-stop --no-stem (no stopwords, no stemming)
    - python cacm_parser.py --cacm cacm.all --stopwords stopwords.txt (stopwords and stemming enabled)

    - this program generates dictionary.json, postings.json, idf.json, and doc_vectors.json

searchUI.py:
    - python searchUI.py
    - When prompted, enter your query and hit enter. Type ZZEND to end the program
    - Output is the top 10 results, with the rank, doc_id, similarity score, title and authors

eval.py:
    - python eval.py 
        - will return the AP and R-Precision values for each of the 65 queries, and at the end
        will return the overall evaluation metrics (MAP and Average R-Precision).


Implementation details:
    - Inverted Index Construction:
        - implemented in cacm_parser.py (is what 'invert' is according to assignment description)
        - each doucument's title and abstract are comvined and tokenized 
        - stopwrod removal and stemming flags can be set to enable or disable them
        - positings lists are ordered by document ID.
        
    -Weighting Scheme:
        - tf-idf weighting is calculated as so: weight = (tf)(idf) = (1 + log10(f))(log10(N/df))
        - This weighting is applied to both the documents and queries, and both are normalized when the cosine similarity is computed
    - Top-K Retrieval Method:
        - Method chosen: Index Elimination 
        - Low-idf query terms are removed before retrieval.
        - Only documents containing the remaining high-IDF query terms are considered (index elimination)
        - if the number of remaining documents > N=100, only the top 100 candidate documents are scored using cosine similarity
        - The top K=10 documents with the highest cosine similarity, are returned as the final results, and with the relevant information

Overall Metrics:
MAP: 0.0574
Average R-Precision: 0.0785

Note: 
Before running anything, run 'pip install nltk numpy'
