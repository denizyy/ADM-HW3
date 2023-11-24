import heapq  # Importing heapq module for heap operations
import pandas as pd  
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from nltk.corpus import stopwords


def search_engine_3(df, terms, termsF, lst_stopwords, query, K):
    # Reading the precomputed TF-IDF data from a CSV file
    tfidf_faculty = pd.read_csv('/content/tfidf_faculty')

    # Initializing a Porter stemmer from nltk for word stemming
    stemmer = nltk.PorterStemmer()

    # Tokenizing and stemming the input query
    cleaned_query = [stemmer.stem(word) for word in nltk.word_tokenize(query) if not word in lst_stopwords and word.isalnum()]

    # Filtering the terms in the query that are present in the TF-IDF data columns
    cleaned_query_in_faculty_vocab = [term for term in cleaned_query if term in tfidf_faculty.columns]

    # Finding document indexes that contain the terms in the cleaned query
    indexes = set()
    for term in cleaned_query_in_faculty_vocab:
        term_matches = termsF.loc[termsF['term'] == term, 'reverse'].tolist()
        if term_matches:
            term_indexes = set(term_matches[0])
            if not indexes:
                indexes = term_indexes
            else:
                indexes = indexes.intersection(term_indexes)

    # Create the TfidfVectorizer only if there are terms in cleaned_query_in_faculty_vocab
    if cleaned_query_in_faculty_vocab:
        query_tfidf_data = tfidf_faculty.loc[df.index.isin(indexes), cleaned_query_in_faculty_vocab]

        # Computing TF-IDF for the input query
        query_tfidf_vectorizer = TfidfVectorizer(input='content', lowercase=False)
        query_results = query_tfidf_vectorizer.fit_transform([' '.join(cleaned_query)])
        query_result_dense = query_results.toarray()[0]

        # Calculate Euclidean distances between query TF-IDF and documents' TF-IDF
        euclidean_data = []
        for index, row in query_tfidf_data.iterrows():
            euclidean_dist = euclidean(row.values, query_result_dense)
            euclidean_data.append(euclidean_dist)


        # Finding the top K similar documents using a min heap
        min_heap = []

        for idx, score in zip(indexes, euclidean_data):
            heapq.heappush(min_heap, (-score, idx))  # Using negative score for min heap
            if len(min_heap) > K:
                heapq.heappop(min_heap)
        

        # Retrieving the top K documents and their information
        top_k_documents = heapq.nlargest(K, min_heap)
        final_output = []
        for sim, idx in top_k_documents:
            doc_info = df.iloc[idx]
            final_output.append({
                "facultyName": doc_info["facultyName"],
                "universityName": doc_info["universityName"],
                "description": doc_info["description"],
                "url": doc_info['url'],
                "similarity": -sim #Using -sim because we used negative scores for the min heap
            })

        # Creating a DataFrame from the final output and returning it
        final_output = pd.DataFrame(final_output)
        return final_output
    else:
        # If cleaned_query_in_vocab is empty, return an empty DataFrame
        return pd.DataFrame()
