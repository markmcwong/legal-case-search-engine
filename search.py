#!/usr/bin/python3
import math
import string
import nltk
import sys
import getopt
import pickle
from collections import Counter
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()
# global variable to hold the total number of documents indexed/searching through
num_of_docs = 100
# dictionary terms and offset values(pointers) to be held in memory
dictionary = {}
# initialise pointers to files
dictionary_file = postings_file = file_of_queries = output_file_of_results = posting_file = None

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

    global posting_file
    results_file = open(results_file, 'w')
    dict_file = open(dict_file, 'rb')
    queries_file = open(queries_file, 'r')
    posting_file = open(postings_file, 'rb')

    build_dictionary(dict_file)  # build global variable dictionary{}
    # process queries and post results
    process_query(queries_file, posting_file, results_file)


def text_preprocessing(file_content):
    """
    Process the text provided by tokenizing, stemming/lower casing, and removing terms
    that are only punctuation
    @param file_content [string]: original text read from file
    @return [string]: a list of word processed tokens
    """
    content_in_tokens = nltk.word_tokenize(file_content)
    stemmed_lowered_tokens = [ps.stem(token.lower())
                              for token in content_in_tokens]
    stemmed_lowered_tokens_without_punc = [token for token in stemmed_lowered_tokens if not(
        all(char in string.punctuation for char in token))]
    return stemmed_lowered_tokens_without_punc

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
            results_file.write(' '.join(list(map(lambda x: x[0], results))) + "\n")
        else:
            results_file.write("\n")

class Query:
    def __init__(self, query_string, is_query_expanded = False):
        self.query_string = query_string
        self.result = []
        self.is_query_expanded = is_query_expanded

    def query_expansion(self, terms):
        # To-do
        return terms

    def evaluate_query(self):
        # if(self.is_query_expanded):
        #     terms = self.query_expansion(terms)
        result = type(self).generate_results(self)
        if result is None:
            return []
        else:
            if(type(self) == BooleanQuery): return type(self).generate_results(self)
            return set(map(lambda x: x[0], type(self).generate_results(self)))

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
        print(first_results, second_results, first_results & second_results)
        return first_results & second_results


class FreeTextQuery(Query):
    def __init__(self, query_string):
        super().__init__(query_string, is_query_expanded = True)    

    def generate_results(self):
        pass

class PhrasalQuery(Query):
    def __init__(self, query_string):
            super().__init__(query_string)

    def generate_results(self):
        terms = text_preprocessing(self.query_string)
        previous_phrase_results = []
        for idx, term in enumerate(terms):
            if term not in dictionary:
                return []  # ignore term
            else:
                posting_file.seek(int(dictionary[term][1]))
                term_posting_list = pickle.load(posting_file)  # load term postings
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
        
        return previous_phrase_results


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
