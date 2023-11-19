import heapq
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_engine_2(df, terms, lst_stopwords, query, K):
    tfidf_data = pd.read_csv('/home/theballer/Desktop/Sapienza Courses/ADM/ADM-HW3/tfidf_data.csv')
    
    stemmer = nltk.PorterStemmer()

    cleaned_query = [stemmer.stem(word) for word in nltk.word_tokenize(query) if not word in lst_stopwords and word.isalnum()]

    indexes = set()
    cleaned_query_in_vocab = [term for term in cleaned_query if term in tfidf_data.columns]

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
        query_tfidf = TfidfVectorizer(input='content', lowercase=False, tokenizer=lambda text: text, vocabulary=cleaned_query_in_vocab)
        query_df = tfidf_data.loc[df.loc[df.index.isin(indexes)].index][cleaned_query_in_vocab]
        query_results = query_tfidf.fit_transform([cleaned_query_in_vocab])
        query_result_dense = query_results.todense()
        query_tfidf_data = pd.DataFrame(query_result_dense.tolist(), columns=[cleaned_query_in_vocab])
        cossim_data = cosine_similarity(query_df, query_tfidf_data)

        max_heap = []
        for idx, sim in zip(indexes, cossim_data):
            heapq.heappush(max_heap, (sim[0], idx))
            if len(max_heap) > K:
                heapq.heappop(max_heap)

        top_k_documents = heapq.nlargest(K, max_heap)

        final_output = []
        for sim, idx in top_k_documents:
            doc_info = df.iloc[idx]
            final_output.append({
                "courseName": doc_info["courseName"],
                "universityName": doc_info["universityName"],
                "description": doc_info["description"],
                "url": doc_info['url'],
                "Similarity": sim
            })

        final_output = pd.DataFrame(final_output)
        return final_output
    else:
        # If cleaned_query_in_vocab is empty, return an empty DataFrame
        return pd.DataFrame()
