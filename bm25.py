import math

avgdl = 41973
N = 17137

# idea taken from https://colab.research.google.com/github/pinecone-io/examples/blob/master/semantic_search_intro/bm25.ipynb#scrollTo=dmpRpbvaDXaM

def bm25(document_frequency, doc_length, log_tf, k=1.2, b=0.75):
    # term frequency...
    freq = math.pow(log_tf,10)  # or f(q,D) - freq of query in Doc
    tf = (freq * (k + 1)) / (freq + k * (1 - b + b * doc_length) / avgdl)
    # inverse document frequency...
    document_frequency  # number of docs that contain the word
    idf = math.log(((N - document_frequency + 0.5) / (document_frequency + 0.5)) + 1)
    return round(tf*idf, 4)