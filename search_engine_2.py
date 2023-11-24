import heapq  # Importing heapq module for heap operations
import pandas as pd
import nltk
#from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity

def search_engine_2(df, terms, lst_stopwords, query, K, tfidf, tfidf_data, main_sparse):
    # Initializing a Porter stemmer from nltk for word stemming
    stemmer = nltk.PorterStemmer()
    # Tokenizing and stemming the input query
    cleaned_query = [stemmer.stem(word) for word in nltk.word_tokenize(query) if not word in lst_stopwords and word.isalnum()]
    # Filtering the terms in the query that are present in the TF-IDF data columns
    cleaned_query_in_vocab = [term for term in cleaned_query if term in tfidf_data.columns]
    # Finding document indexes that contain the terms in the cleaned query
    indexes = set()
    for term in cleaned_query_in_vocab:
        term_matches = terms.loc[terms['term'] == term, 'reverse'].tolist()
        if term_matches:
            term_indexes = set(term_matches[0])
            if not indexes:
                indexes = term_indexes
            else:
                indexes = indexes.intersection(term_indexes)

    # Create the TfidfVectorizer only if there are terms in cleaned_query_in_vocab
    if cleaned_query_in_vocab:
        # Computing TF-IDF for the input query
        query_saprse = tfidf.transform([cleaned_query_in_vocab])
        cossim_data = cosine_similarity(main_sparse[list(indexes)], query_saprse)

        # Finding the top K similar documents using a max heap
        max_heap = []
        for idx, sim in zip(indexes, cossim_data):
            heapq.heappush(max_heap, (sim[0], idx))
            if len(max_heap) > K:
                heapq.heappop(max_heap)

        # Retrieving the top K documents and their information
        top_k_documents = heapq.nlargest(K, max_heap)
        final_output = []
        for sim, idx in top_k_documents:
            doc_info = df.iloc[idx]
            final_output.append({
                "courseName": doc_info["courseName"],
                "universityName": doc_info["universityName"],
                "description": doc_info["description"],
                "url": doc_info['url'],
                "similarity": sim
            })

        # Creating a DataFrame from the final output and returning it
        final_output = pd.DataFrame(final_output)
        return final_output
    else:
        # If cleaned_query_in_vocab is empty, return an empty DataFrame
        return pd.DataFrame()
