#!/usr/bin/python3
import re
#import requests
import nltk
import sys
import getopt
import csv
import string
import time
from index_helper import *
from translator import britishize
from profiler import profile
from tqdm import tqdm

# Turn on/off compression methods
DELTA_COMPRESSION = True
VB_COMPRESSION = True


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

@profile
def build_index(in_file, out_dict, out_postings):
    """
    build index from documents stored in the input file,
    then output the dictionary file and postings file
    """
    global visited_id
    global N
    # Cache stemmed words to optimise stemming
    stemDict = {}
    # Creater stemmer
    stemmer = nltk.stem.PorterStemmer()

    print('indexing...')

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
    
    # Store visited docID to skip duplicate doc
    visited_id = {}
    
    with open(in_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        header = next(csv_reader) # get the first row (table header)
        #print(f'Column names: {", ".join(header)}') ## document_id, title, content, date_posted, court
        #print(header)
        
        for row in csv_reader:
            doc_id = row[0]
            if doc_id in visited_id:
                court_list = visited_id[doc_id]
                court_list.append(row[4])
                continue
                
            visited_id[doc_id] = [row[4]]
            #print(row[0]) ## print document ID
            doc_words.append((row[0], (row[1] + ' ' + row[2] + ' ' + row[4]))) #Combine title, content,court, Limit for testing to 200 characters
    
    print("End of file parsing")
    # testing for duplicate doc ID
    #print(visited_id["247336"])
    #print(visited_id["3926753"])
    
    
    # Get total number of documents
    N = len(doc_words)
    #print(N)
    
    # Initialize dictionary
    dictionary = {}
    dictionary['DOC_LENGTH'] = {}
    
    # debugging
    #doc_words = doc_words[3000:]
    
    BREAKNUM = 1
    for doc in tqdm(doc_words):
        if BREAKNUM == 1000:
            break # for testing purposes, check 2 doc
        BREAKNUM += 1
        """
        if BREAKNUM % 100 == 0:
            print("Current progress:", BREAKNUM)
            print("--- %s seconds ---" % (time.time() - start))
        """
        # Apply case folding, converting text to lower case
        text = doc[1].lower()
        
        # Remove non-latin characters
        regex = re.compile('[^\u0020-\u024F]')
        text = regex.sub('',text)
        
        # Standardize to british english
        text = britishize(text)
        
        # Remove single quotations
        text = text.replace("'","")
        
        # Remove all punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Initialize term frequency map to store term frequency and a list of positions
        freq_map = {}
        
        # Initialize term position
        pos = 0
        
        # Apply tokenization and stemming
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                # Skip term that are solely punctuations
                #if all(char in string.punctuation for char in word):
                #    continue
                    
                if word in stemDict:
                    stemmed_word = stemDict[word]
                else:
                    # Apply stemming
                    stemmed_word = stemmer.stem(word)
                    stemDict[word] = stemmed_word
                
                # Update term frequency map
                if stemmed_word not in freq_map:
                    freq_map[stemmed_word] = [1,[pos]]
                else:
                    freq_map[stemmed_word][0] += 1
                    freq_map[stemmed_word][1].append(pos)
                
                # Update position
                pos += 1
        
        # Initialise document length for this document
        doc_length = 0
        
        # Update document frequency and posting list
        for term, list in freq_map.items():
            tf = list[0]
            pos_list = list[1]
            
            # Apply log-frequency weighting scheme
            log_tf = 1 + math.log(tf,10)
            # Update document length
            doc_length += log_tf ** 2
            
            if DELTA_COMPRESSION:
                # Compress position list using delta representation
                pos_list = calculate_deltas(pos_list)
            
            # structure of each value in the dictionary:
            # (document_frequency, [(docID_1, log_tf_1, [position1,position2...]), (docID_2, log_tf_2, [position1,position2...]) ... (docID_n, log_tf_n, [position1,position2...])])
            
            if term not in dictionary:
                dictionary[term] = (1, [(int(doc[0]), log_tf, pos_list)])
            else:
                # Update current document frequency
                df = dictionary[term][0] + 1
                posting = dictionary[term][1] + [(int(doc[0]), log_tf, pos_list)]
                dictionary[term] = (df,posting)
        
        # Compute and store the document length for current document, to be used in normalization in searching
        dictionary['DOC_LENGTH'][int(doc[0])] = math.sqrt(doc_length)
    
    invert(dictionary,out_dict, out_postings)
    
def invert(dictionary,out_dict, out_postings):
    global visited_id
    
    print("Inverting...")
    
    f_dict = open(out_dict, "wb")
    f_post = open(out_postings, 'wb')
    # erase any existing file
    f_dict.truncate(0)
    f_post.truncate(0)
    
    # Write dictionaries and postings to dictionary.txt and posting.txt
    for term, value in tqdm(dictionary.items()):
        #print("term: ",term)
        #print("value:", value)
        
        if term == "DOC_LENGTH":
            pointer = f_post.tell()
            pickle.dump(value, f_post)
            dictionary[term] = pointer
            continue
        
        # Store the posting lists to disk
        df,posting = value
        pointer = f_post.tell()
        
        # Posting format: [(docID_1, log_tf_1, [position1,position2...]), (docID_2, log_tf_2, [position1,position2...]) ... (docID_n, log_tf_n, [position1,position2...])])
        if VB_COMPRESSION:
            # Apply variable byte encoding to posting
            encoded_posting = []
            for tuple in posting:
                docID = vbcode.VBEncodeNumber(int(tuple[0]))
                log_tf = vbcode.VBEncodeNumber(int(tuple[1]))
                pos_list = vbcode.VBEncode(tuple[2])
                encoded_posting.append((docID,log_tf,pos_list))
            
            pickle.dump(encoded_posting,f_post)
            
        else:
            pickle.dump(posting, f_post)
        
        # Convert df to idf
        global N
        idf = math.log((N / df),10)
        
        # Update dictionary with idf and pointer
        dictionary[term] = (idf,pointer)
    
    # Write the mapping between document id and a list of court names into posting.txt using the special key "DOC_COURT"
    pointer = f_post.tell()
    pickle.dump(visited_id, f_post)
    dictionary["DOC_COURT"] = pointer
    
    # Store the dictionary to disk
    pickle.dump(dictionary,f_dict)
    f_dict.close()
    f_post.close()
    print("Finished writing")
    
def calculate_deltas(numbers):
    if not numbers:
        return numbers

    deltas = [numbers[0]]
    for i, n in enumerate(numbers[1:]):
        deltas.append(n - numbers[i])
    return deltas

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

start = time.time()
build_index(input_file, output_file_dictionary, output_file_postings)
print("--- %s seconds ---" % (time.time() - start))
