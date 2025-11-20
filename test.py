import argparse
import json
import re
import time
from nltk.stem import PorterStemmer
from cacm_parser import parse_cacm_all

#this class (test.py) is to handle the user queries to search for a term in the indexed documents 

TOKEN_REGEX = re.compile(r"\b[0-9A-Za-z]+\b")
def tokenize(text):
    return TOKEN_REGEX.findall(text)

def snippet_builder(text, position, text_len=10):
    words = text.split()
    index = position-1 #0-based index

    half = text_len // 2
    start = max(0, index - half)
    end = min(len(words), index + half)

    snippet = words[start:end]

    #handling edge cases if the term is tooo close to the start or end of the doc
    if (len(snippet)<text_len) and (end - text_len >=0):
        start = max(0, end-text_len)
        snippet = words[start:end]
    
    local_index = index - start
    if 0 <= local_index < len(snippet):
        snippet[local_index] = f"[{snippet[local_index]}]"
    
    return " ".join(snippet)
    
    

if __name__ == "__main__":

    #fixing bug for stemming: 
    #previously: was stemming the query by default so if stemming was turned off for cacm_parser, it was trying to match 'queri' to the stored [queries, query, Query] etc. in 
    # dictionary for example. This is wrong so now I set the flags option for this program too to disable and enable stemming here too to match the execution.  
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stem", action="store_true", help = "Disable stemming for the query")
    args = parser.parse_args()

    #loading the dictionary and postings JSON files
    with open("dictionary.json", encoding="utf-8") as f:
        dictionary = json.load(f)
    with open("postings.json", encoding="utf-8") as f:
        postings = json.load(f)

    docs = parse_cacm_all("cacm.all")
    stemmer = None if args.no_stem else PorterStemmer()

    print("Type a term to search, or type <<ZZEND>> to quit the program.\n")

    query_timings=[]

    while True:
        term = input("Enter term: ").strip()
        if term == "ZZEND":
            break
        word = term.lower()
        if stemmer:
            word= stemmer.stem(word)

        start = time.perf_counter() #start the timer for how long it takes to retrieve result

        if word in dictionary:

            df = dictionary[word]
            print(f"\n Term '{term}' (stemmed: '{word}') appears in {df} documents.\n")
            
            for posting in postings[word]:
                doc_id = posting["doc_id"]
                term_freq = posting["term_freq"]
                positions = posting["positions"]
                title = docs[doc_id]["title"]

                combined_text = (docs[doc_id]["title"] + " " + docs[doc_id]["abstract"])
                snippet = snippet_builder(combined_text, positions[0], text_len=10)

                print(f"Doc ID: {doc_id}")
                print(f"Title: {title}")
                print(f"Term Frequency: {term_freq}")
                print(f"Positions: {positions}")
                print(f"Summary: {snippet}\n")
        else:
            print(f"\nTerm '{term}' does not appear in any document.\n")

        time_taken = time.perf_counter() - start #only counts for one user-typed query
        query_timings.append(time_taken)
        print(f"Query Time : {time_taken:.6f} seconds\n")


    if query_timings:
        average_time = sum(query_timings)/len(query_timings)
        print(f"Average query time: {average_time:.6f} seconds for {len(query_timings)} queries.")
    print("Program ended.")


