import argparse
import json
from nltk.stem import PorterStemmer
from cacm_parser import parse_cacm_all, load_stopwords
from search import search_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive CACM Search Engine")
    parser.add_argument("--no-stop", action="store_true", help="stopword removal disabled for query")
    parser.add_argument("--no-stem", action="store_true", help="stemming disabled for query")
    parser.add_argument("--topK", type=int, default=10, help="Number of top results to display")
    parser.add_argument("--topN", type=int, default=100, help="Subset of documents for score calculation")

# flags added for pagerank integration:
    parser.add_argument("--w1", type=float, default=0.7, help="Weight for cosine similarity score (w1)")
    parser.add_argument("--w2", type=float, default=0.3, help="Weight for PageRank score (w2)")

    args = parser.parse_args()

    with open("idf.json", encoding="utf-8") as f:
        idf = json.load(f)
    with open("doc_vectors.json", encoding="utf-8") as f:
        doc_vectors = json.load(f)

    # Loading PageRank scores
    try:
        with open("pagerank.json", encoding="utf-8") as f:
            pagerank_scores = json.load(f)
        print("Loaded PageRank scores.")
    except:
        pagerank_scores = None
        print("pagerank.json not found. Running the search without PageRank integration")

    docs = parse_cacm_all("cacm.all")
    stopwords = load_stopwords("stopwords.txt") if not args.no_stop else set()
    stemmer = None if args.no_stem else PorterStemmer()

    print("***** Welcome to the CACM Search Engine *****")
    print("Type a query below, or type <<ZZEND>> to quit.\n")

    while True:
        query = input("Enter query: ").strip()
        if query == "ZZEND":
            print("Exited Program.")
            break
        if not query:
            print("Please enter a query: \n")
            continue

        #idf threshold is set at 0.1
        top_docs = search_query(
            query, doc_vectors, docs, idf,
            stopwords=stopwords, stemmer=stemmer,
            idf_threshold=0.1, top_k=args.topK, top_n=args.topN, pagerank=pagerank_scores, w1=args.w1, w2=args.w2
        )

        if not top_docs:
            print("No relevant documents were found for the query.\n")
            continue

        print(f"\nTop {args.topK} Results for the Query: \"{query}\"")
        print("-" * 50)
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            title = docs[int(doc_id)]["title"]
            authors = docs[int(doc_id)]["authors"]
            print(f"{rank}. Doc ID: {doc_id} | Score: {score:.4f}")
            print(f"   Title: {title}")
            print(f"   Authors: {authors}\n")
        print("-" * 50 + "\n")
