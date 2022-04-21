This is the README file for A0188307X-A0204208U-A0219963R-A0248149W's submission
Email(s): e0421414@u.nus.edu, e0323891@u.nus.edu, e055621@u.nus.edu, e0923880@u.nus.edu

== Python Version ==

We're using Python Version 3.10.3 for this assignment.

== General Notes about this assignment ==

Index.py:

Indexing is first done in-memory and then written into the disk at the end. The in-memory dictionary stores all the term as key and a tuple containing document frequency and a list of postings as the value to each term. Each posting is a list of document id, logarithmic term frequency, and a position list for positional indexing.

After writing to the disk, each term's value in the dictionary is now a tuple containing the inverse document frequency and a pointer which will be mapped to the location of the postings list information saved in postings.txt.

Besides terms, there are also two special keys “DOC_LENGTH" and “DOC_COURT” inside the final dictionary written to disk. They are mapped to the pointers to separate dictionaries saved in postings.txt which store the pre-computed document lengths and a list of court names for each document id. We find it more convenient and less prone errors to save everything inside the two mandatory file dictionary.txt and postings.txt, with the use of special keys.

The mapping between document ids to a list of courts is saved to optimise ranking in searching step according to the hierarchy of courts provided. A list of courts instead of just 1 court is saved because we observed that there exists entries that are mostly identical in all fields except for the court information. We kept a record of the document ids visited during our indexing step so as to avoid unnecessary processing on duplicate documents.

Structure of in-memory dictionary when terms are the key:
"term": (df, [(docID1, log_tf1, [positions]), (docID2, log_tf2,[positions]),(docID3, log_tf3,[positions]),...]

Structure of in-memory dictionary when "DOC_LENGTH" is the key:
"DOC_LENGTH" : { docID1 : doc_length1
                 ......
                }

Structure of in-memory dictionary when "DOC_COURT" is the key:
"DOC_COURT" : { docID1 : [court_name1, court_name2]
                docID2 : [court_name1]
                 ......
                }

Structure of saved dictionary when terms are the key:
"term": (idf, pointer to postings list in postings.txt)


== Actual indexing steps ==
1. Read the given csv file using Python csv modules.
2. Preprocess the text in each document.
a) Concatenate all the fields in a document into a single string except for the date. We ignored the dates because we felt that dates may not carry significant meaning in a legal setting as many unrelated cases can be written on the same date and it would not be very meaningful for lawyers to search for all the cases written on a specific date. Zones and fields besides the court are not used because we felt that the given documents come from many different sources and they lack a standard format for us to easily extract meaningful zones and fields.

However, we did consider court names for each document separately and stored them in a special dictionary inside postings.txt. Since a court hierarchy is provided, we may sort search results according to the importance of the court that a document is associated with during searching.

b) A series of text processing is done to standardise the text. Firstly, case folding was applied to convert all texts to lower case. Secondly, non-latin characters such as Chinese are removed. This was done because we observed that certain documents from courts in Hong Kong contains content that are written in Chinese. Since our query will be solely in English, we chose not to waste time in indexing them. Next, all punctuations are removed as we know that no punctuations will appear in queries. We also standardised the spelling of English words using an American to UK English translation table since the data set contains documents from countries and regions that have different English spelling systems. Finally, we tokenise the text into sentences and then words, and perform porter stemming to get the final token.

We chose not perform lemmatization as it would require part-of-speech tagging, which was not very ideal given that our queries are too short in length to generate meaningful context for the pos tagger. In addition, lemmatization would likely increase the indexing time significantly.

3.For each term, if the term is new, add the term to the in-memory dictionary.
Then add the occurring Document Id (if it has not already been added) and Term frequency
into the corresponding postings list in the postings and update the document frequency
in the dictionary.

4. For each document, use a dictionary called freq_map to keep track of the frequency of each term (tf) in the document. For each term, keep track of its position inside the document.

5. For each term in freq_map, convert tf to log_tf where log_tf = 1 + math.log(tf,10). Update the main dictionary with incremented document frequency and new list of [docID, log_tf, position list].

6. For each document, calculate the length of the document vector as sqrt(sum of (log_tf to the power of 2)). Store it in the main dictionary. The length is precomputed to be used for normalization during searching.
7. Convert document frequency to inverse document frequency for each term-document pair. Pickle the in-memory dictionary and write it to dictionary.txt and posting.txt.

Speed Optimisation
At first, our indexing process takes > 6hr to complete and it significantly delays our plan to test our indexing methods and search methods. To improve the time efficiency of indexing, we explored a few optimisation methods:

1. Cache for stemming
After profiling the execution of our index.py, we realised that calling the stemming methods from nltk takes up a significant portion of our execution time. The stemmer is called unnecessarily when the same word appeared many times so we decided to cache the mapping between original term and the stemmed word in a dictionary. This reduces our indexing time to around 6 hours to 3 hours.

2. Data Structure
Originally, we stored each posting in a tuple. However, given that tuple is an immutable object in Python, a new tuple is created to copy over the existing content whenever the same term is encountered at a different position of the document to update the existing posting. This significantly slows down our indexing process especially for long documents that incur a huge amount of updates in the posting and postings with great length. By changing immutable tuple to mutable list, we successfully reduced our indexing time from 3 hours to around 40 minutes.

== Index compression ==
In order to meet the size limit of 800MB, we used a few compression techniques in our indexing to optimise the size of files written.

1. Gap Encoding
In order to perform phrasal query, we stored the positional index of each term for a given document into a position list. However, the position index can become a very large number and takes up significant memory space, especially for long documents. We decided to compress our index using gap encoding which only stores the gaps between a list of numbers instead of the actual numbers. We can do this because we always iterate from the start of a document, and the positional index for a given term is always incremented. In other words, the position list is ordered so we can easily decode the list of gaps into the actual list of numbers.

2. Variable Byte Encoding
Despite gap encoding, we may still need to store some really large number if the gap between positions is large. In order to further reduce size of our postings.txt, we used variable byte encoding to convert numbers into byte streams of variable sizes that take minimum space for any given number. These byte streams can be easily converted back to numbers later.

====================================================================================================================================================================
Search.py:

- Our search function begins by opening up all necessary files, and building the dictionary in memory, since it should only use a small amount.
The dictionary will follow the same structure as that of dictionary.txt from indexing. The next step is processing the query

- When processing the query, our program will check for quotations and/or boolean operators (AND) to determine how to handle the given
query. If AND operators are present, the results from the different parts of the query will be intersected at the end. Quotations around
a phrase represents a phrasal search, while anything without quotations around it is a free-text search.

- Depending on the part of the query (free text, phrasal, or boolean), its results are generated using a corresponding class:
FreeTextQuery(Query), PhrasalQuery(Query), and BooleanQuery(Query)

  These three classes is where a majority of our document retrieval happens. In the FreeTextQuery class, the term(s) will be preprocessed
  the same way as we did for indexing, except for when we know that the query will not be of a certain format (eg. no need to preprocess removal of punctuations since the queries are punctuation free). For free text queries, all relevant documents (to at least one term) in the query are returned initially.
  Depending on the number of query terms received, this will affect the score later. This is done using a generate_results method, and the
  postings are gathered by loading the specific spot in the pickled postings file (using the offset stored in the dictionary).

  In the PhrasalQuery class, the term(s) are also preprocessed in the same way, and the postings are accessed using the same
  loading of pickled content as in FreeTextQuery. PhrasalQuery's generate_results method, however, also utilizes positional
  indexing to make sure that the elements in the document appear in the correct order before returning a docID.

  The BooleanQuery class evaluates the LHS and RHS of the "AND" operator separately, then combines the results together. For example, the query may be quiet and "phone call". In this case, the LHS is evaluated as a free text query and the RHS as a phrasal query. We chose to implement a non-strict boolean after some experimentation as we found that returning only documents in the intersection is too small, resulting in a low number of actual relevant documents. This gave us a higher score as well, which matches the intuition that in practice the person searching using boolean may not have a strict requirement of having two parts both.

  It should be noted that all three of these classes inherit from the parent class Query, which handles query expansion and query evaluation
  through corresponding methods. The query expansion utilizies word2vec and wordnet, which will be discussed in greater detail in the BONUS.docx

== Different Approaches to Search ==

- It should be noted that a number of different approaches were tested during the lifetime of this project's development for query searching.
  They can be seen through the different files included, and they all have their own strength's and weaknesses.

  freetext.py: We wanted to see how the performance would be affected if we make the requirements less strict for phrasal searches, and instead
  decided to treat them like free text queries of a longer length, which would allow for more advanced searches using the boolean operator as well.
  What we found was that the performance of the searching was relatively the same in most cases, and the search operation would still pick up on
  phrases even if we did not take positional indexing into account. However, there were times where some false positives would be returned.
  A particular downside of this approach was that some words, such as "running", have many different meanings, and the phrasal queries are much better
  at establishing context for the word, while treating it as freetext did not take the different uses of "running" into account, and counted all as relevant.

  search.py & search_test.py: Search_test.py does not use the previously established court hierarchy system of increasing document scores based on court
  importance. So a document from the Supreme court will appear closer to the top of search.py's results, while this is not the case for search_test.py.
  We also tried the difference of having a strict vs. non-strict boolean operator during the project lifecycle in these files. In other words, while we
  originally thought of the AND operator as only returning overlapping documents, we wanted to see how the search engine would be affected by making it
  more accepting. The reasons for this seem relevant: the user may not know exactly what they are looking for when using the boolean operator, and if their
  search may end up removing documents that would actually be highly relevant. For example, "fertility treatment" AND damages may, with a strict operator,
  cause the results to not include highly relevant documents about fertility treatment. With a non-strict operator, the intersection would end up being
  larger; this had the benefit of providing more leeway to the user for their results, and it would still increase the scores of documents that were
  more relevant to terms on both sides of the operator; therefore, the idea was that documents with intersection would score highly regardless.

  We also tried not using any previously established court hierarchy system to increase document scores based on court importance. So a document from
  the Supreme court will have its score increased solely based on the fact that it is from such a high court, and will likely rank higher than a similar
  document from somewhere such as the HK High Court.

  We also tried the difference of having a strict vs. non-strict boolean operator during the project lifecycle. In other words, while we
  originally thought of the AND operator as only returning overlapping documents, we wanted to see how the search engine would be affected by making it
  more accepting. The reasons for this seem relevant: the user may not know exactly what they are looking for when using the boolean operator, and if their
  search may end up removing documents that would actually be highly relevant. For example, "fertility treatment" AND damages may, with a strict operator,
  cause the results to not include highly relevant documents about fertility treatment. With a non-strict operator, the intersection would end up being
  larger; this had the benefit of providing more leeway to the user for their results, and it would still increase the scores of documents that were
  more relevant to terms on both sides of the operator; therefore, the idea was that documents with intersection would score highly regardless.

 == Our Final Decisions on Search Methods ==

  In the end, we chose our particular search methods based on their results on the dataset.
  We chose to keep our experimentations with non-strict boolean operators, as we believe they increase leniency while still keeping the most relevant results at
  the top of the document rankings.
  We no longer use court hierarchy in our search rankings, due to it giving worse results.
  Phrasal search functions more strictly, and is not treated merely as free text.

  The query refinement methods used can be seen in the Bonus.docx



== Sorting according to Court Hierarchy ==
At first, we tried to assign an arbitrary additional score for courts of different categories to rank important courts higher. We tried to assign 2 for most important courts, 1 for important courts and 0 for others. However, the performance (evaluated using 3 sample queries) for this implementation was not ideal:
Average AF2: 0.1410081933
Average MAP: 0.0550766675

It is possibly because the arbitrary score assigned was too large and hence affected the ranking from Vector Space Model significantly. To solve that, we explored the use of 5 rounds of bubble sort to swap ranking between adjacent document ids according to the relative hierarchy of their courts. As every pass of bubble sort can only affect ranking of adjacent documents which we assume to be mostly equivalent in terms of relevance, this approach will fine tune ranking of documents minimally without distorting results from Vector Space Model. The performance (evaluated using 3 sample queries) for this implementation is:

Average AF2: 0.1186321336
Average MAP: 0.2665584308

While the average MAP increases, the average AF2 is still below the baseline. One possible reason is that the assumption that documents from more important courts are of higher relevance does not always hold. It is sometimes possible that the lawyer is researching on cases from local courts and hence finds documents from less important courts more relevant.

== Evaluation of Search Performance ==
Our experimentation and results are discussed in Bonus.docx

== Experimentation with Evaluation and Ranking Document Similarity:

  - BM25
  We attempted to use BM25 score as the metric rather than TF-IDF, however the average MAP and MAF2 was lower in general across all three given examples
  as BM25 scores factor in the document length and the average length of all documents. It could possibly due to the fact that the document id provided
  by relevance feedback file has a longer document length in general and hence the scoring and rank is lower compared to TF-IDF.
  Hence we decided not to replace TF-iDF with BM25 with our final implementation.

  - Doc2Vec and Word2Vec calculating Similarity
  We have also attempted to use Doc2Vec and Word2Vec to calculate the similarity between the query and the documents in CSV.
  For Word2Vec, we create the vectors that represents each document by averaging the sum of word vectors that represents each word of the documents,
  and use cosine similarity to find the closest vectors for the query.

  However due to the size limitation constraints and the large size of the generated model, while we would need to somehow determine a hard threshold to cut off the "not so irrelevant documents",
  which is especially difficult with regards to boolean queries. We have not decided to further experiment with the Doc2Vec / Word2Vec calculating document similarity method.
  Details in https://colab.research.google.com/drive/10oYY8Ko4V4RYfER0v2R571x9Umr-9cXd#scrollTo=wJaXk54lr3d8

********************************

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- index.py - used for creating indexes for the files in the specified directory
- search.py - used for performing search on the given query input file and store the results as a file
- dictionary.txt - used for storing the dictionary terms and their corresponding postings offset in the posting file
- postings.txt - used for storing the document frequency and the list of every doc ID and its weighted term frequency that contains the term
- Bonus.docx - A word document explaining our different approaches to query refinement
- freetext.py - A version of the search engine treating everything as free text
- index_helper.py -
- bm25.py - used for experiments with BM25 scoring method replacing TF-IDF
- model_request.py - used for sending request to the heroku server for getting similar words in query expansions
- query_expansion_experiments.py (commented out code) - used for experimenting with different query expansion techniques such as using Pretrained BERT / Glove model and NLPAug
- translator.py - list of words used to translate American English to British English
- vbcode.py - encoding and decoding methods for variable byte encoding
- word2vec_model.py (commented out code) - used for creating the word2vec model trained on the legal corpus using gensim



== Work allocation ==

While most of the assignment we discussed and implemented together, a general work breakdown is as follow:

A0188307X: Indexing process, positional indexing, court hierarchy, score calculation in search
A0204208U: Indexing process, compression techniques
A0219963R: Search process, query expansion techniques
A0248149W: Search process, court score


== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] I/We, A0219963R-A0204208U-A0188307X-A0248149W, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

== References ==
Variable byte encoding:
http://nlp.stanford.edu/IR-book/html/htmledition/variable-byte-codes-1.html
https://github.com/utahta/pyvbcode/blob/master/vbcode.py

Word2Vec/Doc2Vec for document similarity:
https://www.kaggle.com/code/namansood/document-ranking-ir-system-word2vec-embeddings/notebook
https://www.analyticsvidhya.com/blog/2020/08/information-retrieval-using-word2vec-based-vector-space-model/

Query Expansion:
BERT and Glove model:
https://colab.research.google.com/github/fastforwardlabs/ff14_blog/blob/master/_notebooks/2020-07-22-Improving_the_Retriever_on_Natural_Questions.ipynb#scrollTo=NolyNCP2Fsxy
https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32
https://stackoverflow.com/questions/59865719/how-to-find-the-closest-word-to-a-vector-using-bert
https://huggingface.co/nlpaueb/legal-bert-base-uncased?text=quiet+%5BMASK%5D+call+from+the+defendant

WAF2 evaluation:

bm25:
https://colab.research.google.com/github/pinecone-io/examples/blob/master/semantic_search_intro/bm25.ipynb
