from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import chain
from nltk.stem import PorterStemmer
import pickle
import os
import math
import string

def tokenize_text(doc):
    """Separate text into word tokens

    Returns:
        word_tokens (list): a list of word tokens found in the doc
    """
    # Tokenize text into sentences
    sentences = sent_tokenize(doc)

    # Tokenize into words
    words = [word_tokenize(s) for s in sentences]

    # Flatten list
    word_tokens = list(chain.from_iterable(words))

    return word_tokens


def clean_tokens(word_tokens, remove_punc = False):
    """Perform case folding and stemming on tokens

    Returns:
        clean_tokens (list): a list of cleaned word tokens
    """
    if remove_punc:
        # Case Folding and remove punctuation
        case_folded = [t.lower() for t in word_tokens if t not in string.punctuation]
    else:
        case_folded = [t.lower() for t in word_tokens]

    # Stemming
    stemmer = PorterStemmer()
    clean_tokens = [stemmer.stem(t) for t in case_folded]

    return clean_tokens


def apply_log(termfreq_dic):
    """Applies log transformation on terms in the term frequency dictionary

    Returns:
        termfreq_dict (dict): dictionary with log values
    """
    for term in termfreq_dic:
        termfreq_dic[term] = 1 + math.log(termfreq_dic[term], 10)
    
    return termfreq_dic

def get_doc_length(tokens):
    """Calculate length of documents from tokens in that document after applying transformations 
        
    Returns: 
        dlength (float): Document length after applying log tf and cosine normalization
    """
    # Calculate term frequencies
    temp_dic = {}

    for token in tokens:
        if token not in temp_dic:
            temp_dic[token] = 1
        else:
            temp_dic[token] += 1

    # Calculate logtf
    temp_dic = apply_log(temp_dic)

    # Calculate doc length
    dlength = 0
    for term in temp_dic:
        dlength += temp_dic[term] ** 2

    return math.sqrt(dlength)
    
    

def invert(freq_dic, out_dic, out_postings):
    """Create dictionary and posting list from dictionary of terms, docIds and freq
    
    Return:
        Nothing
    """
    # Clear any old version of files
    if os.path.exists(out_dic):
        os.remove(out_dic)
    if os.path.exists(out_postings):
        os.remove(out_postings)

    out_d = {}

    for term in freq_dic:
        # create posting list
        posting = []
        for docId, tup in freq_dic[term].items():
            termFreq = tup[0]
            posList = tup[1]
            posting.append((docId, termFreq, posList))
        with open(out_postings, "ab") as op:
            pointer = op.tell()
            pickle.dump(posting, op)
        # add to dictionary
        out_d[term] = (len(posting), pointer)

    with open(out_dic, "ab") as od:
        pickle.dump(out_d, od)

    return

