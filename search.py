#!/usr/bin/python3
import math
import string
import nltk
import sys
import getopt
import pickle
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.wsd import lesk
import itertools
import sys

from regex import E
from bm25 import bm25
from model_request import request_for_sim_words
from translator import britishize
from vbcode import VBDecode
from setup import setup_dependencies

nltk.data.path.append("./nltk_data")

COURT_HIERARCHY = {'SG Court of Appeal': 2, 'SG Privy Council': 2, 'UK House of Lords': 2, 'UK Supreme Court': 2, 'High Court of Australia': 2, 'CA Supreme Court': 2, 'SG High Court': 1, 'Singapore International Commercial Court': 1, 'HK High Court': 1, 'HK Court of First Instance': 1, 'UK Crown Court': 1, 'UK Court of Appeal': 1, 'UK High Court': 1, 'Federal Court of Australia': 1, 'NSW Court of Appeal': 1, 'NSW Court of Criminal Appeal': 1, 'NSW Supreme Court': 1}

ps = PorterStemmer()
# global variable to hold the total number of documents indexed/searching through
num_of_docs = 17137
# dictionary terms and offset values(pointers) to be held in memory
dictionary = {}
# list of all doc_id's that have been searched for when a court was in a phrasal query
# these are stored so their scores are not accidentally increased twice
docs_with_court_queries_found = []
# initialise pointers to files
dictionary_file = postings_file = file_of_queries = output_file_of_results = posting_file = word2vec_model = doc_lengths_dict = None

def usage():
    print("usage: " +
          sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def default_dict_lambda():
    return [0, []]

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    Using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    @param dict_file [string]: name/path of the dictionary file provided by user
    @param postings_file [string]: name/path of of the postings file privided by user
    @param queries_file [string]: name/path of the queries file provided by user
    @param results_file [string]: name/path of the results file provided by user
    """
    print('running search on the queries...')
    global posting_file
    results_file = open(results_file, 'w')
    dict_file = open(dict_file, 'rb')
    queries_file = open(queries_file, 'r')
    posting_file = open(postings_file, 'rb')

    build_dictionary(dict_file)  # build global variable dictionary{}
    # process queries and post results
    process_query(queries_file, posting_file, results_file)


def text_preprocessing(query):
    """
    Process the text provided by tokenizing, stemming/lower casing, and standerising to British English
    @param file_content [string]: original query text
    @return [string]: a list of word processed tokens
    """
    content_in_tokens = nltk.word_tokenize(query)
    stemmed_lowered_tokens = [ps.stem(token.lower())
                              for token in content_in_tokens]
    stemmed_lowered_tokens_britishized = [britishize(token)
                              for token in stemmed_lowered_tokens]
    return stemmed_lowered_tokens_britishized

# process each query (each line) in queries_file and post the results to results_file
# one line query = one line results (max 10 docIDs)

def query_parser(line):
    """
    Create a query object that represent the queries requested by current line
    """
    # phrasal_words = re.findall(r'"(.+?)"', line)
    if '"' in line or 'AND' in line: # phrasal or boolean queries are present
        queries_generated = []
        tokens = line.split()
        is_searching_for_phrasal = False
        is_boolean_query_on = False
        temp_phrasal_words = ''
        for token in tokens:
            if token[0] == '"' and token[-1] == '"': # single term phrasal query
                if(is_boolean_query_on):
                    queries_generated[-1].update_second_query(PhrasalQuery(token[1:-1]))
                else:
                    queries_generated.append(PhrasalQuery(token[1:-1]))

            elif token[0] == '"': # multiple term phrasal query
                is_searching_for_phrasal = True
                temp_phrasal_words += token[1:] + ' '

            elif is_searching_for_phrasal: # multiple term phrasal query
                if token[-1] == '"': # if the last character of token is closing quotation mark
                    is_searching_for_phrasal = False # switch off phrasal search
                    temp_phrasal_words += token[:-1]

                    if(is_boolean_query_on):
                        queries_generated[-1].update_second_query(PhrasalQuery(temp_phrasal_words))
                    else:
                        queries_generated.append(PhrasalQuery(temp_phrasal_words))
                    temp_phrasal_words = ''

                else:
                    temp_phrasal_words += token + ' '

            elif not is_searching_for_phrasal and token[0] != '"' and token[-1] != '"' and token != 'AND': # must be a single free text query:
                print("is_boolean_query_on ", is_boolean_query_on)
                if(is_boolean_query_on):
                    queries_generated[-1].update_second_query(FreeTextQuery(token))
                else:
                    queries_generated.append(FreeTextQuery(token))
                temp_phrasal_words = ''

            if token == 'AND':
                is_boolean_query_on = True
                queries_generated[-1] = BooleanQuery(queries_generated[-1].query_string, queries_generated[-1])

        print("Queries objects Generated (non free-text): ", queries_generated)
        return queries_generated[-1]

    else:
        # must be a free text query
        query = FreeTextQuery(line)
        print('Free text query generated:', query)
        return query

def process_query(queries_file, posting_file, results_file):
    """
    process each query (each line) in queries_file and post the results to results_file
    one line query = one line results (max 10 docIDs)
    @param queries_file [string]: name/path of the queries file
    @param postings_file [string]: name/path of of the postings
    @param results_file [string]: name/path of the results file
    """
    lines = queries_file.readlines()

    for line in lines:  # for each query
        query = query_parser(line.strip())
        results = query.evaluate_query()
        print("Query string: ", query.query_string, '\n')

        if(results != None):
            # Apply bubble sort to move up docID from courts with higher priority
            results = list(map(lambda x: str(x[1]), results))
            #results = bubbleSort(results)
            results_file.write(' '.join(results) + "\n")
        else:
            results_file.write("\n")

def wordnet_expansion(sentence_in_tokens):
    """
    Given a list of tokens, expand each token using wordnet to generate synonyms.
    @param sentence_in_tokens [list]: list of tokens forming a query sentence
    """
    sentence_with_nltk_pos = [lesk(sentence_in_tokens, token) for token in sentence_in_tokens]
    synonyms = [set([str(lemma.name()) for lemma in word.lemmas() if '_' not in lemma.name()]) if word is not None else {} for word in sentence_with_nltk_pos ]
    # print(synonyms)
    return synonyms

def word2vec_expansion(sentence_in_tokens):
    """
    Given a list of tokens, expand each token using word2vec to generate synonyms.
    @param sentence_in_tokens [list]: list of tokens forming a query sentence
    """
    # global word2vec_model
    # if word2vec_model is None:
    #     from gensim.models import KeyedVectors
    #     word2vec_model = KeyedVectors.load("vectors.kv")

    # expanded_terms = [word2vec_model.most_similar(token)[:5] if token in word2vec_model else token for token in sentence_in_tokens]
    # print(sentence_in_tokens)
    # print(expanded_terms)

    res = request_for_sim_words(sentence_in_tokens)
    return res

class Query:
    def __init__(self, query_string, is_query_expanded = False):
        self.query_string = query_string
        self.result = []
        self.is_query_expanded = is_query_expanded

    def query_expansion(self, terms):
        # Expand terms in query using wordnet and word2vec
        wordnet_terms = wordnet_expansion(self.query_string.split())
        wordnet_terms = [[ps.stem(term) for term in terms] for terms in wordnet_terms]
        word2vec_terms = text_preprocessing(self.query_string)
        word2vec_terms = word2vec_expansion(word2vec_terms)
        word2vec_terms = [[term[0] for term in terms] for terms in word2vec_terms]

        full_terms_list = []
        for i in range(len(terms)):
            terms_to_return = []
            if(len(wordnet_terms[i]) <= 1):
                terms_to_return = list(wordnet_terms[i]) if len(wordnet_terms[i]) > 0 else list()
            else:
                terms_to_return = [list(wordnet_terms[i])[0]]

            if(type(word2vec_terms[i]) is str or len(word2vec_terms[i]) <= 1):
                if word2vec_terms[i] != []:
                    terms_to_return.append(word2vec_terms[i].translate(str.maketrans('', '', string.punctuation)))
                full_terms_list.append(terms_to_return)
            else:
                terms_to_return.append(word2vec_terms[i][0].translate(str.maketrans('', '', string.punctuation)))
                intersection = set(list(wordnet_terms[i])) & set(word2vec_terms[i])     
                if(len(intersection) > 0):
                    terms_to_return.extend(intersection)
                full_terms_list.append(terms_to_return)

            print(full_terms_list)

        return full_terms_list

    def evaluate_query(self):
        terms = text_preprocessing(self.query_string)
        # Create dictionary to store query log tf
        query_logtf_dic = {term: 1 + math.log(list(terms).count(term), 10) for term in terms}

        if(self.is_query_expanded):
            expanded_terms = list(itertools.chain.from_iterable(self.query_expansion(terms))) 
            expanded_terms = text_preprocessing(' '.join(expanded_terms))
            print("query expansion returned terms: ", expanded_terms)
            concatenated_terms = list(expanded_terms + terms)
            query_logtf_dic = {term: (1 + math.log(list(concatenated_terms).count(term), 10) 
                if term in terms else (1 + 0.2 * math.log(list(concatenated_terms).count(term), 10))) # 0.2 weight for expanded terms
                for term in (expanded_terms + terms)}
            # Create dictionary to store query log tf
            return type(self).generate_results(self, terms, expanded_terms, query_logtf_dic)

        if(type(self) == BooleanQuery):
            return type(self).generate_results(self) 
        else:
            # Phrasal query
            return type(self).generate_results(self, query_logtf_dic)

    def generate_results(self): # Parent method that should be overridden by child classes
        return []

    def __repr__(self):
        return '(' + self.__class__.__name__ + ': ' + self.query_string + ')'

class BooleanQuery(Query):
    def __init__(self, query_string, first_query):
        super().__init__(query_string)
        self.first_query = first_query

    def update_second_query(self, second_query):
        self.second_query = second_query
        self.query_string += ' AND ' + second_query.query_string

    def generate_results(self):
        first_results = self.first_query.evaluate_query()
        second_results = self.second_query.evaluate_query()
        # first_results_in_ids = list(map(lambda x: x[1], first_results))
        # second_results_in_ids = list(map(lambda x: x[1], second_results))

        result = {}
        for k, v in first_results + second_results:
            result[v] = (result.get(v, 0) + k)
        
        # overlapped = {k:v for k,v in result.items() if k in first_results_in_ids and k in second_results_in_ids}
        results_to_return = sorted(((val, did) for did, val in result.items()), reverse =
        True) # currently not a strict intersection
        return results_to_return


class FreeTextQuery(Query):
    def __init__(self, query_string):
        super().__init__(query_string, is_query_expanded = True)

    def generate_results(self, terms, expanded_terms, query_logtf_dic):
        # Initialize the dictionary for storing normalized vectors for each documents that contain a particular query term
        term_doc_dictionary = {}
        # Store unique docIDs that contain at least one of the term in the query
        candidates = set()
        
        # Initialize a dictionary to store the weighted vector for each term
        query_term_vector = {}
        
        for idx, term in enumerate(terms):
            term_doc_dictionary[term] = {}
            # if term not in dictionary, ignore that term
            if term in dictionary:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file) # load term postings
                term_posting_list = decompress_posting(term_posting_list) # Apply decompression
                ## Calculate query weight
                query_weight = query_logtf_dic[term] * dictionary[term][0] ##logtf * idf
                query_term_vector[term] = query_weight
                
                for doc in term_posting_list:
                    docID, log_tf = doc[0], doc[1]
                    doc_length = doc_lengths_dict[docID]
                    # Apply length normalization
                    term_doc_dictionary[term][docID] = log_tf / doc_length
                    candidates.add(docID)
                    
        # Initialize
        ranking = {}
    
        for docID in candidates:
            score = 0
            
            for idx, term in enumerate(terms):
                # vector score will be 0 if the document does not contain the term, so just skip
                if docID not in term_doc_dictionary[term]:
                    continue
                
                else:
                    # compute cosine similarity using dot product
                    term_query_score = query_term_vector[term]
                    term_doc_score = term_doc_dictionary[term][docID]
                    cos_similarity = term_query_score * term_doc_score
                    score += cos_similarity
            
            # Sort by decreasing order of the score and ascending order of docID for the same score
            #heapq.heappush(pq, (score, -1 * int(docID)))
            ranking[int(docID)] = score

        """
        # Add court score
        ptr = dictionary['DOC_COURT'] # pointer to another dictionary
        posting_file.seek(int(ptr))
        court_dic = pickle.load(posting_file) # dictionary containing docid -> [court...] info
        for did, val in ranking.items(): # repeat for each document
            courts = court_dic[str(did)]
            # extract greatest court value
            court_value = max([COURT_HIERARCHY[court] if court in COURT_HIERARCHY else 0 for court in courts]) / 2
            # modify score to include court value
            ranking[did] += court_value
        
        """
        
        # Sort and return docs in ranked order
        results_to_return = sorted(((val, did) for did, val in ranking.items()), reverse = True)
        #results_to_return = map(lambda x: str(-1 * x[1]), list(item for _, _, item in pq)
        print("number of docs returned: ", len(results_to_return))
        return results_to_return



class PhrasalQuery(Query):
    def __init__(self, query_string):
            super().__init__(query_string)

    def generate_results(self, query_logtf_dic):
        terms = text_preprocessing(self.query_string)
        previous_phrase_results = []
        for idx, term in enumerate(terms):
            if term not in dictionary:
                return []  # ignore term
            else:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file)  # load term postings
                term_posting_list = decompress_posting(term_posting_list) # apply decompression
                if (idx == 0):
                    previous_phrase_results = term_posting_list
                    continue
                else:
                    results_to_return = []
                    for item in previous_phrase_results:
                        for doc in term_posting_list:
                            # if the doc ID we are looking is greater than the item, skip the rest and move on
                            if(doc[0] > item[0]):
                                break

                            # (docID, log_idf, [pos])
                            if(item[0] == doc[0]):
                                # Create iterators for both lists to compare
                                last_round_iter = iter(item[2])
                                posting_iter = iter(doc[2])
                                # get the first item from both lists
                                last_round = next(last_round_iter, None)
                                posting = next(posting_iter, None)

                                while True:
                                    if(last_round + 1 == posting):
                                        if(len(results_to_return) == 0 or results_to_return[-1][0] != item[0]):
                                            results_to_return.append((doc[0], doc[1], [posting]))
                                        else:
                                            results_to_return[-1][2].append(last_round)
                                        last_round = next(last_round_iter, None)
                                        posting = next(posting_iter, None)
                                    elif(last_round < posting):
                                        last_round = next(last_round_iter, None)
                                    elif(last_round > posting):
                                        posting = next(posting_iter, None)

                                    if(posting is None or last_round is None):
                                        break
                    
                    # print("results_to_return", sum(map(lambda x: len(x[2]), results_to_return)))
                    if(results_to_return == []):
                        return []
                    else:
                        previous_phrase_results = results_to_return

                    # print(results_to_return)
                if previous_phrase_results == []:
                    return []

        # print("after", previous_phrase_results)
        #Score calculation based on relevant docs
        relevant_docs = list(map(lambda x: x[0], previous_phrase_results))
        # print('previous: ', relevant_docs)
        
        # Initialize the dictionary for storing normalized vectors for each documents that contain a particular query term
        term_doc_dictionary = {}
        # Store unique docIDs that contain at least one of the term in the query
        candidates = set()
        
        # Initialize a dictionary to store the weighted vector for each term
        query_term_vector = {}
        
        for idx, term in enumerate(terms):
            term_doc_dictionary[term] = {}
            # if term not in dictionary, ignore that term
            if term in dictionary:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file) # load term postings
                term_posting_list = decompress_posting(term_posting_list) # Apply decompression
                ## Calculate query weight
                query_weight = query_logtf_dic[term] * dictionary[term][0] ##logtf * idf
                query_term_vector[term] = query_weight
                
                for doc in term_posting_list:
                    if doc[0] not in relevant_docs:
                        continue
                    docID, log_tf = doc[0], doc[1]
                    doc_length = doc_lengths_dict[docID]
                    # Apply length normalization
                    term_doc_dictionary[term][docID] = log_tf / doc_length
                    candidates.add(docID)
                    
        # Initialize
        ranking = {}
    
        for docID in candidates:
            score = 0
            
            for idx, term in enumerate(terms):
                # vector score will be 0 if the document does not contain the term, so just skip
                if docID not in term_doc_dictionary[term]:
                    continue
                
                else:
                    # compute cosine similarity using dot product
                    term_query_score = query_term_vector[term]
                    term_doc_score = term_doc_dictionary[term][docID]
                    cos_similarity = term_query_score * term_doc_score
                    score += cos_similarity
            
            # Sort by decreasing order of the score and ascending order of docID for the same score
            #heapq.heappush(pq, (score, -1 * int(docID)))
            ranking[int(docID)] = score
        
        """
        # Add court score
        ptr = dictionary['DOC_COURT'] # pointer to another dictionary
        posting_file.seek(ptr)
        court_dic = pickle.load(posting_file) # dictionary containing docid -> [court...] info
        for did, val in ranking.items(): # repeat for each document
            courts = court_dic[str(did)]
            # extract greatest court value
            court_value = max([COURT_HIERARCHY[court] if court in COURT_HIERARCHY else 0 for court in courts]) / 2
            # modify score to include court value
            ranking[did] += court_value

            #if the phrase equals a court, add to the score of all docs from that court
            if self.query_string in courts:
                if did not in docs_with_court_queries_found:
                    docs_with_court_queries_found.append(did)
                    ranking[did] = ranking[did] * 1.3
        """
        # Sort and return docs in ranked order
        results_to_return = sorted(((val, did) for did, val in ranking.items()), reverse = True)
        #results_to_return = map(lambda x: str(-1 * x[1]), list(item for _, _, item in pq)
        print("number of docs returned: ", len(results_to_return))
        return results_to_return


def build_dictionary(dict_file):
    """
    create dictionary by reading in dict_file
    line by line, very small overhead for search function
    @param dict_file [string]: name/path of the dictionary file
    """
    global dictionary, doc_lengths_dict,court_dict
    dictionary = pickle.load(dict_file)
    posting_file.seek(int(dictionary['DOC_LENGTH']))
    doc_lengths_dict = pickle.load(posting_file)
    
    # Court dictionary to be used in court hierarchy evaluation
    posting_file.seek(dictionary['DOC_COURT'])
    court_dict = pickle.load(posting_file) # dictionary containing docid -> [court...] info


def idf_weight(doc_freq):
    """
    calculate idf-weight given document frequency
    @param doc_freq [int]: number of documents a term appears in
    """
    # num_of_docs is a global variable
    global num_of_docs
    return math.log(num_of_docs/doc_freq, 10)


def log_frequency_weight(query_tf):
    """
    calculate log-frequency-weight given the query's term frequency
    @param query_tf [int]: frequency of a given term
    """
    if(query_tf == 0):  # ignore taking log of 0
        return 0
    else:
        return 1 + math.log(query_tf, 10)

def decompress_posting(compressed_posting):
    """
    Decompress the posting list read from disk (which was compressed using variable byte 
    encoding, and delta compression)
    
    @param compressed_posting [list]: posting list in compressed form
    """
    decompressed_posting = []
    for tuple in compressed_posting:
        decompressed_tuple = decompress(tuple)
        decompressed_posting.append(decompressed_tuple)
    return decompressed_posting

def decompress(compressed_tuple):
    """
    Decompress the tuple inside posting list read from disk (which was compressed
    using variable byte encoding, and delta compression)
    
    @param compressed_tuple [tuple]: a tuple in posting list that is in compressed form
    """
    docID = VBDecode(compressed_tuple[0])[0]
    log_tf = VBDecode(compressed_tuple[1])[0]
    deltas = VBDecode(compressed_tuple[2])
    pos_lst = from_deltas(deltas)
    return (docID, log_tf, pos_lst)

def from_deltas(deltas):
    """
    Convert a list of delta numbers(difference between numbers) to the actual list of numbers
    @param deltas [list]: list of integers representing positional index differences
    """
    if not deltas:
        return deltas

    numbers = [deltas[0]]
    for i in deltas[1:]:
        numbers.append(i + numbers[-1])
    return numbers

def singleBubbleSort(results):
    #One round of bubble sort to increase ranking for documents with higher courts priority
    for x in range(len(results) - 1):
        docId1 = results[x]
        docId2 = results[x + 1]
        # If lower ranked doc id is from a court with higher priority
        if court_dict[docId2] > court_dict[docId1]:
            # swap the two doc id
            results[x] = docId2
            results[x + 1] = docId1
    return results

def bubbleSort(results):
    # Do 5 passes of bubbleSort on results
    for i in range(10):
        results = singleBubbleSort(results)
    return results
    
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

# setup_dependencies()
run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
