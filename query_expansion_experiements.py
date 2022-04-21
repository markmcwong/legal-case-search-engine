# import pandas as pd
# import numpy as np
# import pickle
# import scipy
# import nltk
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import string
# from tqdm.auto import tqdm
# import gensim
# from sklearn.metrics.pairwise import cosine_similarity
# import spacy
# import gensim.downloader as api
# from transformers import pipeline, AutoTokenizer
# from nltk.corpus import wordnet
# import itertools
# import nlpaug.augmenter.word as naw
# from nltk.wsd import lesk
# from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


# nlp = spacy.load("en_core_web_sm")
# word_vectors = api.load("glove-wiki-gigaword-50")
# unmasker = pipeline('fill-mask', model="nlpaueb/legal-bert-base-uncased", tokenizer="bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", use_fast=True)

# def NLPAug_experiment():
    # aug_wordnet = naw.SynonymAug(aug_src='wordnet',aug_max=2)
    # print('Original:', sentences[0])
    # print(aug_wordnet.augment(sentences[0], n=3))
    # print('================')
    # print('Original:', sentences[1])
    # print(aug_wordnet.augment(sentences[1], n=3))

    # TOPK=20 #default=100
    # # ACT = "substitute"
    # ACT = 'insert'

    # aug_bert = naw.ContextualWordEmbsAug(
    #     model_path='bert-base-uncased', 
    #     #device='cuda',
    #     action=ACT)

    # for ii in range(5):
    #     augmented_text = [aug_bert.augment(sent) for sent in sentences]
    #     print(augmented_text)

# def pretrained_bert_and_glove_experiment():
    # Directly copied Query Expander class from https://colab.research.google.com/github/fastforwardlabs/ff14_blog/blob/master/_notebooks/2020-07-22-Improving_the_Retriever_on_Natural_Questions.ipynb#scrollTo=NolyNCP2Fsxy
    # and removed elastic_search related code

    # class QueryExpander:
    #     '''
    #     Query expansion utility that augments ElasticSearch queries with optional techniques
    #     including Named Entity Recognition and Synonym Expansion
        
    #     Args:
    #         question_text
    #         entity_args (dict) - Ex. {'spacy_model': nlp}
    #         synonym_args (dict) - Ex. {'gensim_model': word_vectors, 'n_syns': 3} OR
    #                                   {'MLM': unmasker, 'tokenizer': base_tokenizer, 'n_syns': 3, 'threshold':0.3}
    #     '''
        
    #     def __init__(self, question_text, entity_args=None, synonym_args=None):
            
    #         self.question_text = question_text
    #         self.entity_args = entity_args
    #         self.synonym_args = synonym_args

    #         if self.synonym_args and not self.entity_args:
    #             raise Exception('Cannot do synonym expansion without NER! Expanding synonyms\
    #                             on named entities reduces recall.')

    #         if self.synonym_args or self.entity_args:
    #             self.nlp = self.entity_args['spacy_model']
    #             self.doc = self.nlp(self.question_text)
            
    #         self.build_query()
            
    #     def build_query(self):

    #         # build entity subquery
    #         if self.entity_args:
    #             self.extract_entities()
            
    #         # identify terms to expand
    #         if self.synonym_args:
    #             self.identify_terms_to_expand()
            
    #         # build question subquery
    #         self.construct_question_query()
            
    #         # combine subqueries
    #         sub_queries = []
    #         sub_queries.append(self.question_sub_query)
    #         if hasattr(self, 'entity_sub_queries'):
    #             sub_queries.extend(self.entity_sub_queries)

    #         print(sub_queries)
    #         self.query = sub_queries
        
    #     def extract_entities(self):
    #         '''
    #         Extracts named entities using spaCy and constructs phrase match subqueries
    #         for each entity. Saves both entities and subqueries as attributes.
            
    #         '''
            
    #         entity_list = [entity.text.lower() for entity in self.doc.ents]
    #         entity_sub_queries = []
            
    #         for ent in entity_list:
    #             eq = ent
                
    #             entity_sub_queries.append(eq)
            
    #         self.entities = entity_list
    #         self.entity_sub_queries = entity_sub_queries
            
            
    #     def identify_terms_to_expand(self):
    #         '''
    #         Identify terms in the question that are eligible for expansion
    #         per a set of defined rules
            
    #         '''
    #         if hasattr(self, 'entities'):
    #             # get unique list of entity tokens
    #             entity_terms = [ent.split(' ') for ent in self.entities]
    #             entity_terms = [ent for sublist in entity_terms for ent in sublist]
    #         else:
    #             entity_terms = []
        
    #         # terms to expand are not part of entity, a stopword, numeric, etc.
    #         entity_pos = ["NOUN","VERB","ADJ","ADV"]
    #         terms_to_expand = [idx_term for idx_term in enumerate(self.doc) if \
    #                            (idx_term[1].lower_ not in entity_terms) and (not idx_term[1].is_stop)\
    #                             and (not idx_term[1].is_digit) and (not idx_term[1].is_punct) and 
    #                             (not len(idx_term[1].lower_)==1 and idx_term[1].is_alpha) and
    #                             (idx_term[1].pos_ in entity_pos)]
            
    #         self.terms_to_expand = terms_to_expand

            
    #     def construct_question_query(self):
    #         '''
    #         Builds a multi-match query from the raw question text extended with synonyms 
    #         for any eligible terms

    #         '''
    #         if hasattr(self, 'terms_to_expand'):
                
    #             syns = []
    #             for i, term in self.terms_to_expand:

    #                 if 'gensim_model' in self.synonym_args.keys():
    #                     syns.extend(self.gather_synonyms_static(term))

    #                 elif 'MLM' in self.synonym_args.keys():
    #                     syns.extend(self.gather_synonyms_contextual(i, term))

    #             syns = list(set(syns))
    #             syns = [syn for syn in syns if (syn.isalpha() and self.nlp(syn)[0].pos_ != 'PROPN')]
                
    #             question = self.question_text + ' ' + ' '.join(syns)
    #             self.expanded_question = question
    #             self.all_syns = syns
            
    #         else:
    #             question = self.question_text
            
    #         qq = question
            
    #         self.question_sub_query = qq


    #     def gather_synonyms_contextual(self, token_index, token):
    #         '''
    #         Takes in a token, and returns specified number of synonyms as defined by
    #         predictions from a masked language model
            
    #         '''
            
    #         tokens = [token.text for token in self.doc]
    #         tokens[token_index] = self.synonym_args['tokenizer'].mask_token
            
    #         terms = self.predict_mask(text = ' '.join(tokens), 
    #                                     unmasker = self.synonym_args['MLM'],
    #                                     tokenizer = self.synonym_args['tokenizer'],
    #                                     threshold = self.synonym_args['threshold'],
    #                                     top_n = self.synonym_args['n_syns'])
            
    #         return terms


    #     @staticmethod
    #     def predict_mask(text, unmasker, tokenizer, threshold=0, top_n=2):
    #         '''
    #         Given a sentence with a [MASK] token in it, this function will return the most 
    #         contextually similar terms to fill in the [MASK]
            
    #         '''

    #         preds = unmasker(text)
    #         tokens = [tokenizer.convert_ids_to_tokens(pred['token']) for pred in preds if pred['score'] > threshold]
            
    #         return tokens[:top_n]
            

    #     def gather_synonyms_static(self, token):
    #         '''
    #         Takes in a token and returns a specified number of synonyms defined by
    #         cosine similarity of word vectors. Uses stemming to ensure none of the
    #         returned synonyms share the same stem (ex. photo and photos can't happen)
            
    #         '''
    #         try:
    #             syns = self.synonym_args['gensim_model'].similar_by_word(token.lower_)

    #             lemmas = []
    #             final_terms = []
    #             for item in syns:
    #                 term = item[0]
    #                 lemma = self.nlp(term)[0].lemma_

    #                 if lemma in lemmas:
    #                     continue
    #                 else:
    #                     lemmas.append(lemma)
    #                     final_terms.append(term)
    #                     if len(final_terms) == self.synonym_args['n_syns']:
    #                         break
    #         except:
    #             final_terms = []

    #         return final_terms

    #     def explain_expansion(self, entities=True):
    #         '''
    #         Print out an explanation for the query expansion methodology
            
    #         '''
            
    #         print('Question:', self.question_text, '\n')
            
    #         if entities:
    #             print('Found Entities:', self.entities, '\n')
            
    #         if hasattr(self, 'terms_to_expand'):
                
    #             print('Synonym Expansions:')
            
    #             for i, term in self.terms_to_expand:
                    
    #                 if 'gensim_model' in self.synonym_args.keys():
    #                     print(term, '-->', self.gather_synonyms_static(term))
                    
    #                 elif 'MLM' in self.synonym_args.keys():
    #                     print(term, '-->', self.gather_synonyms_contextual(i,term))
                
    #                 else:
    #                     print('Question text has no terms to expand.')
                        
    #             print()
    #             print('Expanded Question:', self.expanded_question, '\n')
            
    #         print('Query:\n', self.query)

    # sentences = ['quiet phone call', 'good grades trade scandals', "fertility treatment damages claim"]

    # BERT Model:

    # entity_args = {'spacy_model': nlp}
    # synonym_args = {
    #                 'MLM': unmasker, 
    #                 'tokenizer': tokenizer, 
    #                 'n_syns': 2,
    #                 # 'gensim_model': word_vectors,
    #                 'threshold': 0.05
    #                 }
                    
    # for question in sentences:
    #   qe_ner = QueryExpander(question, entity_args, synonym_args)
    #   qe_ner.explain_expansion(entities=True)

    # Glove Model: 

    # entity_args = {'spacy_model': nlp}
    # synonym_args = {
    #                 'n_syns': 2,
    #                 'gensim_model': word_vectors,
    #                 }
                    
    # for question in sentences:
    #   qe_ner = QueryExpander(question, entity_args, synonym_args)
    #   qe_ner.explain_expansion(entities=True)