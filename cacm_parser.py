import argparse
import json
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer 

#accepts the path to the cacm.all file as input and returns a dictionary where the keys are the document IDs 
# and the values are dictionaries containing the title, abstract, authors, and the date of each document.
def parse_cacm_all(path):
    docs = {} #using dictionary structure to store
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    current_docId = None 
    current_field = None 
    buff = {"T": [], "W": [], "A": [], "B": [], "X":[]} 

    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('.I'): #.I indicates the start of a new document
            if current_docId is not None:
                docs[current_docId] = {
                    "title": " ".join(buff["T"]).strip(),
                    "abstract": " ".join(buff["W"]).strip(),
                    "authors": " ".join(buff["A"]).strip(),
                    "date": " ".join(buff["B"]).strip(),
                    "citations": buff["X"]
                }

                buff = {"T": [], "W": [], "A": [], "B": [], "X": []}

            current_docId = int(line[3:].strip()) 
            current_field = None
        elif line.startswith('.T'):
            current_field = "T"
        elif line.startswith('.W'):
            current_field = "W"
        elif line.startswith('.A'):
            current_field = "A"
        elif line.startswith('.B'):
            current_field = "B" 
        elif line.startswith('.X'):
            current_field = "X"
        elif line.startswith('.'):
            current_field = None
        else:
            if current_docId is not None and current_field is not None:
                if current_field != "X":
                    buff[current_field].append(line.strip())
                else:
                    parts = line.strip().split()
                    if len(parts) ==3:
                        _, mid, dest = parts
                        if mid == "5":
                            buff["X"].append(dest)
    
    if current_docId is not None:
        docs[current_docId] = {
            "title": " ".join(buff["T"]).strip(),
            "abstract": " ".join(buff["W"]).strip(),
            "authors": " ".join(buff["A"]).strip(),
            "date": " ".join(buff["B"]).strip(),
            "citations": buff["X"]
        }
    
    return docs


#Tokenization using regex to extract alphanumeric tokens
TOKEN_REGEX = re.compile(r"\b[0-9A-Za-z]+\b")
def tokenize(text):
    return TOKEN_REGEX.findall(text)


#loading stopwords from stopwords.txt --> renamed from common_words.txt
def load_stopwords(path):
    s = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word: s.add(word.lower())
    return s


#Building the inverted index
def index_builder(documents, stopwords=set(), stop=True, stem=False, stemmer=None):

    inverted = defaultdict(list)
    dictionary = {}

    for doc_id, cont in documents.items():
        #Combining the title and the abstract here
        text = (cont["title"] + " " + cont["abstract"]).strip()

        #Tokenization
        tokens = tokenize(text)

        #preprocessing the tokens
        #assignming a number (position) to each of the terms
        processed_text = []
        for i, token in enumerate(tokens, start=1):
            t = token.lower()
            if stop and t in stopwords:
                continue
            if stem and stemmer:
                t = stemmer.stem(t)
            processed_text.append((t, i))

        #grouping the positions of each term in the document
        term_pos = defaultdict(list)
        for t, pos in processed_text:
            term_pos[t].append(pos)

        #adding the postings for each term in the inverted index
        for term, positions in term_pos.items():
            inverted[term].append({"doc_id": doc_id, "term_freq": len(positions), "positions": positions})

    #building the dictionary (map term to document frequency)
    for term, postings in inverted.items():
        dictionary[term]=len(postings) # len(positings) is the document frequency of the term
        postings.sort(key=lambda x: x["doc_id"]) #lambda function to sort the postings by document ID

    return dictionary, inverted


# build the citation graph:
def build_citation_graph(docs):
    graph = {}
    doc_ids = set(docs.keys())

    for doc_id, data in docs.items():
        outlinks = []
        for c in data.get("citations", []):
            try:
                dest = int(c)
                if dest in doc_ids:
                    outlinks.append(dest)
            except:
                continue
        graph[doc_id] = outlinks
    return graph


# Computing PageRank
def compute_pagerank(graph, damping_factor = 0.85, max_iter=50, tol=1e-6):
    N = len(graph)
    pr = {node: 1.0/N for node in graph}
    outdeg = {node: len(graph[node]) for node in graph}

    for _ in range(max_iter):
        new_pr = {node: 0.0 for node in graph}

        for node in graph: 
            if outdeg[node] ==0:
                for other in graph: 
                    new_pr[other] += pr[node]/N
            else:
                share = pr[node]/outdeg[node]
                for dest in graph[node]:
                    new_pr[dest] +=share

        for node in new_pr:
            new_pr[node] = (1-damping_factor)*new_pr[node] + damping_factor/N
        
        diff = sum(abs(new_pr[n] - pr[n]) for n in graph)
        if diff < tol:
            break
        pr = new_pr
    
    return pr


# Main program
if __name__ == "__main__":
    #setting up the argument parser -- what command line arguments to accept
    #use --no-stop to disable stopword removal
    #use --no-stem to disable stemming
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacm", default="cacm.all", help = "Path to cacm.all file")
    parser.add_argument("--stopwords", default="stopwords.txt", help = "Path to stopwords.txt file")
    parser.add_argument("--no-stop", action="store_true", help = "Disable stopword removal")
    parser.add_argument("--no-stem", action="store_true", help = "Disable stemming option")
    parser.add_argument("--outdir", default=".", help = "Output directory")
    args = parser.parse_args()

    docs = parse_cacm_all(args.cacm)
    print("Parsed", len(docs), "documents")

    #Loading the stopwords (if enabled)
    stopwords = load_stopwords(args.stopwords) if not args.no_stop else set()
    print(f"Loaded {len(stopwords)} stopwords" if stopwords else "Stopword removal disabled")

    #Using the stemmer if enabled
    stemmer = PorterStemmer() if not args.no_stem else None
    if stemmer:
        print("Stemming enabled using Porter Stemmer")
    else:
        print("Stemming disabled")

    dictionary, postings = index_builder(
        docs, stopwords=stopwords, stop=not args.no_stop, stem=not args.no_stem, stemmer=stemmer)
    
    total_docs = len(docs)
    idf = {}
    for term, df in dictionary.items():
        idf[term] = math.log10(total_docs / df) 

    doc_vectors = {}
    for term, posting_list in postings.items():
        for posting in posting_list:
            doc_id = posting["doc_id"]
            tf = posting["term_freq"]
            weight = (1 + math.log10(tf)) * idf[term]
            doc_vectors.setdefault(doc_id, {})[term] = weight

    # --- Normalize document vectors ---
    def normalize_vector(vec):
        sum_of_squares = sum(weight ** 2 for weight in vec.values())
        norm = math.sqrt(sum_of_squares)
        if norm == 0:
            return vec
        return {term: weight / norm for term, weight in vec.items()}

    doc_vectors = {doc_id: normalize_vector(vec) for doc_id, vec in doc_vectors.items()}

    # --- Save supporting data files ---
    with open(f"{args.outdir}/idf.json", "w", encoding="utf-8") as f:
        json.dump(idf, f, indent=2, ensure_ascii=False)

    with open(f"{args.outdir}/doc_vectors.json", "w", encoding="utf-8") as f:
        json.dump(doc_vectors, f, indent=2, ensure_ascii=False)

    #Save the dictionary and postings to JSON files dictionary.json and postings.json
    with open(f"{args.outdir}/dictionary.json", "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)

    with open(f"{args.outdir}/postings.json", "w", encoding="utf-8") as f:
        json.dump(postings, f, indent=2, ensure_ascii=False)

    print("Index built and is saved to", args.outdir)

    # Build citation graph and then compute PageRank
    graph = build_citation_graph(docs)
    pagerank_scores = compute_pagerank(graph)

    # saving pagerank scores to a json file
    with open(f"{args.outdir}/pagerank.json", "w", encoding="utf-8") as f:
        json.dump(pagerank_scores, f, indent=2)
    print("PageRank computed and saved.")