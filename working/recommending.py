import numpy as np
import pandas as pd 
import logging
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from collections import defaultdict
from parsing import ingredient_parser
import unidecode
import ast

data = pd.read_csv('data/Parsed_allRecipe.csv')
data1 = pd.read_csv('data/allRecipe.csv')
data['parsed_new'] = data1.Ingredients.apply(ingredient_parser)

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed_new.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

corpus = get_and_sort_corpus(data)
#print(f"Length of corpus: {len(corpus)}")

# calculate average length of each document 
lengths = [len(doc) for doc in corpus]
avg_len = float(sum(lengths)) / len(lengths)

sg = 0 # CBOW: build a language model that correctly predicts the center word given the context words in which the center word appears
workers = 8 # number of CPUs
window = 9 # window size: average length of each document 
min_count = 1 # unique ingredients are important to decide recipes 

model_cbow = Word2Vec(corpus, sg=sg, workers=workers, window=window, min_count=min_count)

#Summarize the loaded model
#print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.index_to_key)
words.sort()
# print(words)

#Acess vector for one word
# print(model_cbow.wv['chicken stock'])

#model_cbow.save('data/model_cbow.bin')

class MeanEmbeddingVectorizer(object):

	def __init__(self, word_model):
		self.word_model = word_model
		self.vector_size = word_model.wv.vector_size

	def fit(self):  # comply with scikit-learn transformer requirement
		return self

	def transform(self, docs):  # comply with scikit-learn transformer requirement
		doc_word_vector = self.word_average_list(docs)
		return doc_word_vector

	def word_average(self, sent):
		"""
		Compute average word vector for a single doc/sentence.
		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""
		mean = []
		for word in sent:
			if word in self.word_model.wv.index_to_key:
				mean.append(self.word_model.wv.get_vector(word))

		if not mean:  # empty words
			# If a text is empty, return a vector of zeros.
			logging.warning("cannot compute average owing to no vector for {}".format(sent))
			return np.zeros(self.vector_size)
		else:
			mean = np.array(mean).mean(axis=0)
			return mean

	def word_average_list(self, docs):
		"""
		Compute average word vector for multiple docs, where docs had been tokenized.
		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
		return np.vstack([self.word_average(sent) for sent in docs])
	
loaded_model = Word2Vec.load('data/model_cbow.bin')
if loaded_model:
    print("Successfully loaded model")

mean_vec_tr = MeanEmbeddingVectorizer(loaded_model)
doc_vec = mean_vec_tr.transform(corpus)

def get_recommendations(N, scores):
    # load in recipe dataset
    df_recipes = pd.read_csv("data/Parsed_allRecipe.csv")
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(
        columns=['recipe', 'ingredients', 'score', 'url'])
    count = 0
    for i in top:
        recommendation.at[count, 'recipe'] = title_parser(
            df_recipes['Name'][i])
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(df_recipes['Parsed_Ingredients'][i])
        recommendation.at[count, 'url'] = df_recipes['URL'][i]
        recommendation.at[count, 'directions'] = df_recipes['Steps'][i]
        recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i]))
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title


# neaten the ingredients being outputted
def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)

    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):

        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):  # comply with scikit-learn transformer requirement
        """
		Fit in a list of docs, which had been preprocessed and tokenized,
		such as word bi-grammed, stop-words removed, lemmatized, part of speech filtered.
		Then build up a tfidf model to compute each word's idf as its weight.
		Noted that tf weight is already involved when constructing average word vectors, and thus omitted.
		:param
			pre_processed_docs: list of docs, which are tokenized
		:return:
			self
		"""

        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)  # must be list of text string

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)  # used as default value for defaultdict
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):
        """
		Compute average word vector for a single doc/sentence.
		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )  # idf weighted

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            # logging.warning(
            #     "cannot compute average owing to no vector for {}".format(sent)
            # )
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):
        """
		Compute average word vector for multiple docs, where docs had been tokenized.
		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
        return np.vstack([self.word_average(sent) for sent in docs])

def RecSys(ingredients, N=5,mean=False):
    """
    The reccomendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of of cosine similarity. 
    :param ingredients: a list of ingredients
    :param N: the number of reccomendations returned 
    :return: top 5 reccomendations for cooking recipes
    """

    model = Word2Vec.load("data/model_cbow.bin")
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")

    data = pd.read_csv('data/Parsed_allRecipe.csv')
    data1 = pd.read_csv('data/allRecipe.csv')
    data['parsed_new'] = data1.Ingredients.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)
    if mean:
        # get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        # use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    # load in tdidf model and encodings
    # with open("data/encoding.pkl", 'rb') as f:
    #     tfidf_encodings = pickle.load(f)
    # with open("data/model.pkl", "rb") as f:
    #     tfidf = pickle.load(f)

    # parse the ingredients using my ingredient_parser
    # try:
    #     ingredients_parsed = ingredient_parser(ingredients)
    # except:
    #     ingredients_parsed = ingredient_parser([ingredients])

    # # use our pretrained tfidf model to encode our input ingredients
    # ingredients_tfidf = tfidf.transform([ingredients_parsed])
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    # parse ingredient list
    input = ingredient_parser(input)
    # get embeddings for ingredient doc
    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)

    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations


if __name__ == "__main__":
    # test ingredients
    test_ingredients = "chicken,onion,garlic"
    recs = RecSys(test_ingredients)
    print(recs.score)