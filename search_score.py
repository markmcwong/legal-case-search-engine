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
from distutils.core import run_setup
from translator import britishize

nltk.data.path.append("./nltk_data")

ps = PorterStemmer()
# global variable to hold the total number of documents indexed/searching through
num_of_docs = 100
# dictionary terms and offset values(pointers) to be held in memory
dictionary = {}
# initialise pointers to files
dictionary_file = postings_file = file_of_queries = output_file_of_results = posting_file = word2vec_model = None

def usage():
    print("usage: " +
          sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


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
    run_setup('gensim-4.1.2/setup.py', script_args='install')

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
    if '"' in line or 'AND' in line:
        queries_generated = []
        tokens = line.split()
        is_searching_for_phrasal = False
        is_boolean_query_on = False
        temp_phrasal_words = ''
        for token in tokens:
            if token[0] == '"' and token[-1] == '"':
                if(is_boolean_query_on):
                    queries_generated[-1].update_second_query(PhrasalQuery(token[1:-1]))
                else:
                    queries_generated.append(PhrasalQuery(token[1:-1]))

            elif token[0] == '"':
                is_searching_for_phrasal = True
                temp_phrasal_words += token[1:] + ' '

            elif is_searching_for_phrasal:
                if token[-1] == '"': # if the last character of token is closing quotation mark
                    is_searching_for_phrasal = False # switch off phrasal search
                    temp_phrasal_words += token[:-1]

                    if(is_boolean_query_on):
                        queries_generated[-1].update_second_query(PhrasalQuery(temp_phrasal_words))
                    else:
                        queries_generated.append(PhrasalQuery(temp_phrasal_words))
                    temp_phrasal_words = ''

                else:
                    temp_phrasal_words += token[:-1]
            
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
        print("Query string: ", query.query_string, "\nAnd Results: ", results, '\n')

        if(results != None):
            results_file.write(' '.join(list(map(lambda x: x[1], results))) + "\n")
        else:
            results_file.write("\n")

def wordnet_expansion(sentence_in_tokens):
    sentence_with_nltk_pos = [lesk(sentence_in_tokens, token) for token in sentence_in_tokens]
    synonyms = [set([str(ps.stem(lemma.name())) for lemma in word.lemmas()]) for word in sentence_with_nltk_pos if word is not None]
    return synonyms

def word2vec_expansion(sentence_in_tokens):
    global word2vec_model
    if word2vec_model is None:
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load("vectors.kv")

    return [map(lambda x: ps.stem(x[0]), word2vec_model.most_similar(token)[:3]) if token in word2vec_model else token for token in sentence_in_tokens]

class Query:
    def __init__(self, query_string, is_query_expanded = False):
        self.query_string = query_string
        self.result = []
        self.is_query_expanded = is_query_expanded

    def query_expansion(self, terms):
        return wordnet_expansion(terms) + word2vec_expansion(terms)

    def evaluate_query(self):
        terms = text_preprocessing(self.query_string)
        # Create dictionary to store query log tf
        query_logtf_dic = {term: 1 + math.log(list(terms).count(term), 10) for term in terms}
         
        if(self.is_query_expanded):
            #terms = text_preprocessing(self.query_string)
            #terms = set(itertools.chain.from_iterable(self.query_expansion(terms))) # temporarily remove query expansion
            print("query expansion returned terms: ", terms)
            # Create dictionary to store query log tf
            return type(self).generate_results(self, query_logtf_dic)
        
        if(type(self) == BooleanQuery):
            return type(self).generate_results(self)
        else:
            ## Phrasal query
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
        return list(set(first_results) & set(second_results))
                   

COURT_HIERARCHY = {'SG Court of Appeal': 2, 'SG Privy Council': 2, 'UK House of Lords': 2, 'UK Supreme Court': 2, 'High Court of Australia': 2, 'CA Supreme Court': 2, 'SG High Court': 1, 'Singapore International Commercial Court': 1, 'HK High Court': 1, 'HK Court of First Instance': 1, 'UK Crown Court': 1, 'UK Court of Appeal': 1, 'UK High Court': 1, 'Federal Court of Australia': 1, 'NSW Court of Appeal': 1, 'NSW Court of Criminal Appeal': 1, 'NSW Supreme Court': 1}

class FreeTextQuery(Query):
    def __init__(self, query_string):
        super().__init__(query_string, is_query_expanded = True)

    def generate_results(self, query_logtf_dic):
        terms = text_preprocessing(self.query_string)
        scores = {}
        relevant_docs = []
        for idx, term in enumerate(terms):
            # if term not in dictionary, ignore that term
            if term in dictionary:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file) # load term postings
                term_posting_list = decompress_posting(term_posting_list) # Apply decompression
                ## Calculate query weight
                query_weight = query_logtf_dic[term] * math.log(num_of_docs/dictionary[term][0]) ##logtf * idf
                for doc in term_posting_list:
                    if doc[0] not in scores:
                        scores[doc[0]] = 0
                    scores[doc[0]] += query_weight * doc[1] ## scores[docid] += queryweight * docweight
                    relevant_docs.append(doc[0])
        # Normalize
        for i in relevant_docs:
            scores[i] = scores[i] / dictionary['DOC_LENGTH'][i]

        # Add court score
        ptr = dictionary['DOC_COURT'] # pointer to another dictionary
        posting_file.seek(ptr)
        court_dic = pickle.load(posting_file) # dictionary containing docid -> [court...] info
        for did, val in scores.items(): # repeat for each document
            courts = court_dic[did]
            # extract greatest court value 
            court_value = max([COURT_HIERARCHY[court] if court in COURT_HIERARCHY else 0 for court in courts])
            # modify score to include court value
            scores[did] += court_value
            

        # Sort and return docs in ranked order
        results_to_return = sorted(((val, did) for did, val in scores.items()), reverse = True)
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
                                            results_to_return.append(item)
                                        else:
                                            results_to_return[-1][1] += 1
                                            results_to_return[-1][2].append(last_round)
                                        last_round = next(last_round_iter, None)
                                        posting = next(posting_iter, None)
                                    elif(last_round < posting):
                                        last_round = next(last_round_iter, None)
                                    elif(last_round > posting):
                                        posting = next(posting_iter, None)

                                    if(next(posting_iter, None) is None or next(last_round, None) is None):
                                        break

                    # print(results_to_return)
                if previous_phrase_results == []:
                    return

        #Score calculation based on relevant docs
        relevant_docs = list(map(lambda x: x[0], previous_phrase_results))
        scores = {} 
        for idx, term in enumerate(terms):
            # if term not in dictionary, ignore that term
            if term in dictionary:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file) # load term postings
                term_posting_list = decompress_posting(term_posting_list)
                ## Calculate query weight
                query_weight = query_logtf_dic[term] * math.log(num_of_docs/dictionary[term][0]) ##logtf * idf
                for doc in term_posting_list:
                    if doc[0] not in scores:
                        scores[doc[0]] = 0
                    scores[doc[0]] += query_weight * doc[1] ## scores[docid] += queryweight * docweight
        # Normalize
        for i in relevant_docs:
            scores[i] = scores[i] / dictionary['DOC_LENGTH'][i]
        
        # Add court score
        ptr = dictionary['DOC_COURT'] # pointer to another dictionary
        posting_file.seek(ptr)
        court_dic = pickle.load(posting_file) # dictionary containing docid -> [court...] info
        for did, val in scores.items(): # repeat for each document
            courts = court_dic[did]
            # extract greatest court value 
            court_value = max([COURT_HIERARCHY[court] if court in COURT_HIERARCHY else 0 for court in courts])
            # modify score to include court value
            scores[did] += court_value

        # Sort and return docs in ranked order
        results_to_return = sorted(((val, did) for did, val in scores.items()), reverse = True)
        return results_to_return


def build_dictionary(dict_file):
    """
    create dictionary by reading in dict_file
    line by line, very small overhead for search function
    @param dict_file [string]: name/path of the dictionary file
    """
    global dictionary
    dictionary = pickle.load(dict_file)
    # print(dictionary)


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


def update_score(posting_list, query_tf, scores, weights, df):
    """
    Takes in a posting list and update the score for each document
    Posting List structure:
    {"df": int, "postings": [(docID, weighted_tf), (docID, weighted_tf), ...]}
    @param posting_list [dictionary]: all postings for a given term, {key = term, value = list of postings}
    @param query_tf [int]: frequency of a given term
    @param scores [dictionary]: scores for all relevant docIDs
    @param weights [dictionary]: weights of all relevant docIDs
    """
    # print(posting_list, '\n', df, query_tf)
    for doc in posting_list:
        # extract both values from tuple (docID, weighted_tf)
        doc_ID, doc_length, doc_pos = doc

        # Check if doc_ID is already in scores or weights, if not initalise to 0
        if doc_ID not in weights:
            weights[doc_ID] = 0
        if doc_ID not in scores:
            scores[doc_ID] = 0

        query_weight = log_frequency_weight(query_tf) * idf_weight(df)
        doc_weight = 1
        # doc_weight = doc_weighted_tf # no calculation needed as it is already weighted during indexing step
        scores[doc_ID] += (query_weight * doc_weight)

def decompress_posting(compressed_posting):
    """
    Decompress the posting list read from disk (which was compressed using variable byte encoding, and delta compression)
    """
    decompressed_posting = []
    for tuple in compressed_posting:
        decompressed_tuple = decompress(tuple)
        decompressed_posting.append(decompressed_tuple)
    return decompressed_posting

def decompress(compressed_tuple):
    """
    Decompress the tuple inside posting list read from disk (which was compressed using variable byte encoding, and delta compression)
    """
    docID = VBDecode(compressed_posting[0])[0]
    log_tf = VBDecode(compressed_posting[1])[0]
    deltas = VBDecode(compressed_posting[2])
    pos_lst = from_deltas(deltas)
    return (docID, log_tf, pos_lst)

def from_deltas(deltas):
    """
    Convert a list of delta numbers(difference between numbers) to the actual list of numbers
    """
    if not deltas:
        return deltas

    numbers = [deltas[0]]
    for i in deltas[1:]:
        numbers.append(i + numbers[-1])
    return numbers

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

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
