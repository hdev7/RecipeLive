# Miscellaneous scripts for modeling notebook
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Given a dataframe containing elements in the NMF-transformed matrix, return back
# another dataframe with each of the element converted into percentages
def getPercentages(df):
    '''Return a dataframe back, with elements of transformed dtm matrix (df) converted into percentages,
    so it would be easier to inspect intuitively
    Input:
        df - is df containing transformed dtm matrix
    '''
    n_rows = df.shape[0]
    dfout = df.copy()
    for i in range(n_rows):
        sumrow = np.sum(df.iloc[[i],:], axis=1)
        dfout.iloc[[i],:] = df.iloc[[i],:].apply(lambda x: 100*x/(sumrow))
    return dfout


# Create a function to quickly check topic words as a function of n_topics
def getNmfTopics(df, n_topics, n_words=10):
    ''' Function returns the breakdown of topics and their related words
    Input:
        df       : a dataframe
        n_topics : number of topics used in nmf
        n_words  : output n words describing topics
    '''

    # Preprocess
    sentences_tokens = df['summary'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    doc_word = vectorizer.fit_transform(sentences_tokens)

    # Build an NMF
    nmf = NMF(n_topics)
    doc_topic = nmf.fit_transform(doc_word)

    return display_topics(nmf, vectorizer.get_feature_names(), no_top_words=n_words)

# Create a function to collect top words in each topic
def collect_topics(model, feature_names, no_top_words, topic_names=None):
    '''Function returns the top keywords of each topic from the NMF model
    Input:
        feature_names = from model, usually vectorizer.get_feature_names()
        no_top_words  = the number of words to collect from each topic
    '''
    collection= dict()
    for ix, topic in enumerate(model.components_):
        collection['Topic_'+str(ix+1)] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return collection

# Create a function to display words in topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''Function returns the bag of words from model (eg., NMF)
    Input:
        feature_names = from model, usually vectorizer.get_feature_names()
        no_top_words  = the number of words to be displayed from each topic
    '''
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix+1)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# A function that outputs another dataframe with a 'similarity' column added
def similarEntries(Hp, Qp,n=3):
    '''function returns another dataframe that looks like Hp, with 'similarity' column added
    based on cosine-similarity with Qp
    Input:
        HP - a dataframe containing observations
        Qp - the transformed query, after processing
    '''
    print(Hp.shape)
    print(Qp.shape)

    nrows = Hp.shape[0]
    if Hp.columns[-1] =='title':
        Hp = Hp.iloc[:,:-1] ## <- without title

    Hp = np.asarray(Hp)
    Qp = np.asarray(Qp)

    print(Hp.shape)
    print(Qp.shape)

    out = []
    for j in range(nrows):
        out.append(cosine_similarity(Qp.reshape(1,-1), Hp[j,:].reshape(1,-1)))

    cos_sim = out

    return [each[0][0] for each in cos_sim]

# A function to return a dataframe containing cosine similarity with the query
def Recommender(Hp, query, name, vectorizer,model, top_n=10):
    '''Function returns another dataframe that contains 'similarity' column, based on
    cosine similarity of the transformed query and every entry in the input dataframe, Hp
    Input:
        Hp         - a DataFrame containing observations
        query      - a string of query
        name       - the name of scientist being queried, used as index in output Qp
        top_n      - the number of similar documents to inspect
        vectorizer - is the vectorizer object
        model      - is model object, e.g., nmf
    '''
    cols = ['academia', 'comic', 'indian', 'fictional', 'european', 'russian','compsci','TV',
            'physicist', 'title']  # < without title

    # Transform the query using the same vectorizer as above
    doc_q = vectorizer.transform([query])
    # Use nmf model from above to transform the vectorized query
    doc_topic_q = model.transform(doc_q)

    # Create a dataframe for the query
    Hp = Hp[cols]
    Qp = getPercentages(pd.DataFrame(doc_topic_q.round(3),
                    index= [name],
                    columns = cols[:-1]))  # <- Use all columns, but 'title'

    # cosine similarities
    cos_sim = similarEntries(Hp, Qp)

    # Add a new column
    Hp['similarity']= cos_sim

    return Hp.sort_values(by='similarity', ascending=False).head(top_n), Qp