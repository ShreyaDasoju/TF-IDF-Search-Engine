import json
import argparse
from cacm_parser import parse_cacm_all
from math import fabs

# Building the graph
def build_citation_graph(docs):
    # undirected link graph
    graph = {doc_id: set() for doc_id in docs.keys()}

    for doc_id, data in docs.items():
        for (a,b) in data.get("X", []):
            graph[a].add(b)
            graph[b].add(a)

    return {doc: list(neighbors) for doc, neighbors in graph.items()}


# computing PageRank - using the power iteration method
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


# Main program:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PageRank for CACM citations")
    parser.add_argument("--cacm", default="cacm.all", help = "Path to cacm.all file")
    parser.add_argument("--out", default="pagerank.json", help="Output file for pagerank scores")
    args = parser.parse_args()

    docs = parse_cacm_all(args.cacm)
    print("Loaded", len(docs), "documents from", args.cacm)

    graph = build_citation_graph(docs)
    print("Graph contains", len(graph), "nodes")

    pr = compute_pagerank(graph)
    print("Computed PageRank scores.")

    max_pr = max(pr.values())
    pr = {doc: score/max_pr for doc, score in pr.items()}
    print("Normalized PageRank scores.")
    print("Saving PageRank scores to", args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pr, f, indent=2)

    print("Done.")