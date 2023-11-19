import nltk
import pandas as pd

def search_engine_1(df, terms, query, stemmer, lst_stopwords):
    """
    Find documents that contain all the words from the query.
    
    :param df: DataFrame containing the documents.
    :param terms: DataFrame with term information and their occurrences.
    :param query: The query string.
    :param stemmer: Stemmer object (e.g., from nltk).
    :param lst_stopwords: List of stopwords.
    :return: DataFrame containing the relevant documents.
    """
    # Clean and stem the query
    cleaned_query = [stemmer.stem(word) for word in nltk.word_tokenize(query) if not word in lst_stopwords and word.isalnum()]

    # Find indices of documents containing all query terms
    indexes = set()
    for term in cleaned_query:
        term_matches = terms.loc[terms['term'] == term, 'reverse'].tolist()

        if term_matches:
            term_indexes = set(term_matches[0])

            if not indexes:
                indexes = term_indexes
            else:
                indexes = indexes.intersection(term_indexes)

    # Return the DataFrame filtered by the indices
    return df.loc[df.index.isin(indexes)] if indexes else pd.DataFrame()
