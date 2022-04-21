# Generating the word2vec model to be used for query expansion
# Our full experiments are documented in https://colab.research.google.com/drive/10oYY8Ko4V4RYfER0v2R571x9Umr-9cXd#scrollTo=I_HrtggtOzG5

# Code are optimised for running on Colab:

# import gensim
# import pandas as pd
# import numpy as np
# import nltk
# import re
# from nltk.corpus import stopwords
# from gensim.models import KeyedVectors
# from google.colab import drive

# nltk.download('stopwords')

# Combined all preprocssing function into one function so we don't have to perform loops multiple times
# def text_preprocessing(sentence):
#     content_in_tokens = nltk.word_tokenize(sentence.replace('\n', ' '))
#     stemmed_lowered_tokens_without_punc = [stemmer.stem(regex.sub('', token.lower()).translate(str.maketrans('', '', string.punctuation)).replace("'","")) for token in content_in_tokens if token not in stop_words]
#     return stemmed_lowered_tokens_without_punc

# def generate_word2vec_model():
    # drive.mount('/gdrive')
    # data = pd.read_csv('/gdrive/MyDrive/Colab Notebooks/dataset.csv')

    # stop_words = set(stopwords.words("english"))
    # stemmer = PorterStemmer()

    # regex = re.compile('[^\u0020-\u024F]')

    # cleaned_data = data
    # cleaned_data['title'] = cleaned_data['title'].progress_apply(text_preprocessing)
    # cleaned_data['content'] = cleaned_data['content'].progress_apply(text_preprocessing)
    # combined_data = pd.concat([cleaned_data['title'], cleaned_data['content']]).reset_index(drop=True)

    # model = gensim.models.word2vec.Word2Vec(sentences=combined_data, vector_size=300, window=3, min_count=2)
    # model.wv.save_word2vec_format('/content/word2vec.bin')
