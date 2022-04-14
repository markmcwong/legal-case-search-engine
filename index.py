#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import csv
from index_helper import * 

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_file, out_dict, out_postings, out_docLengths = "docLengthsFile.txt"):
    """
    build index from documents stored in the input file,
    then output the dictionary file and postings file
    """
    print('indexing...')
    # This is an empty method
    # Pls implement your code in below

    # Expand field size limit as some columns in csv have very large fields
    field_size_limit = sys.maxsize    
    while True:
        try:
            csv.field_size_limit(field_size_limit)
            break
        except OverflowError:
            field_size_limit = int(field_size_limit / 10)
        
    # Read contents from csv file
    doc_words = []
    with open(in_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names: {", ".join(row)}') ## document_id, title, content, date_posted, court
                line_count += 1
            else:
                print(row[0])
                doc_words.append((row[0], (row[1] + ' ' + row[2] + ' ' + row[4])[:200])) # Limit for testing to 200 characters
                
    print("End of file")

    # store token-docId pairs
    token_stream = []

    # store length of documents
    docLengths = []

    BREAKNUM = 1
    for doc in doc_words:
        # Do preprocessing on documents
        cleaned_tokens = clean_tokens(tokenize_text(doc[1]), remove_punc = True)
        # Save as term-docid-pos tuples
        new_tokens = [(term, doc[0], pos) for pos, term in enumerate(cleaned_tokens)]
        # Calculate document length
        docLengths.append((doc[0], get_doc_length(cleaned_tokens)))
        token_stream += new_tokens
        if BREAKNUM == 2:
            break # for testing purposes, check 2 doc   
        BREAKNUM += 1      

    # Save document lengths to file (to be used during searching)
    if os.path.exists(out_docLengths):
        os.remove(out_docLengths)
    ## <<<< currently doclengths not sorted, comes as list of tuples randomly arranged [(docid, length)...]
    with open(out_docLengths, "ab") as outdl_f:
        pickle.dump(docLengths, outdl_f)

    # Invert to dictionary
    # First, sort token_stream by term, docId, pos 
    token_stream.sort()

    # Create dictionary
    freq_dic = {}

    for token in token_stream:
        term = token[0]
        docId = token[1]
        pos = token[2]
        # store in frequency dictionary
        ## for new terms, record first docFreq and first termFreq and its position
        if term not in freq_dic:
            freq_dic[term] = {docId: (1, [pos])}
        else:
            ## For existing term, append docId and pos if from different document, add term frequency and pos otherwise
            if docId not in freq_dic[term]:
                freq_dic[term][docId] = (1, [pos])
            else:
                freq_dic[term][docId] = (freq_dic[term][docId][0] + 1, freq_dic[term][docId][1] + [pos])

    # Create dictionary and posting list
    invert(freq_dic, out_dict, out_postings)


input_file = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_file = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_file == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_file, output_file_dictionary, output_file_postings)
