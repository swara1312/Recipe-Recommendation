import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# load in parsed recipe dataset
df_recipes = pd.read_csv("data/Parsed_allRecipe.csv")
# Tfidf needs unicode or string types
df_recipes['Parsed_Ingredients'] = df_recipes.Parsed_Ingredients.values.astype('U')

# TF-IDF feature extractor
# tfidf = TfidfVectorizer(max_features=500, lowercase=True,
#                         analyzer='word', ngram_range=(1, 5))
tfidf = TfidfVectorizer()
tfidf.fit(df_recipes['Parsed_Ingredients'])
tfidf_recipe = tfidf.transform(df_recipes['Parsed_Ingredients'])

# save the tfidf model and encodings
with open("data/model.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("data/encoding.pkl", "wb") as f:
    pickle.dump(tfidf_recipe, f)
