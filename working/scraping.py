import pandas as pd
from bs4 import BeautifulSoup 
import requests
import time
import csv

attribute = ["Number", "URL", "Name","Ingredients",  "Steps"]
with open('data/allRecipe.csv', 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)  # creating a csv writer object
    writer.writerow(attribute)

main_url = "https://www.jamieoliver.com/recipes/category/course/mains/"
recipeNo = 0

# Fetching html from the website
page = requests.get(main_url)
# BeautifulSoup enables to find the elements/tags in a webpage 
soup = BeautifulSoup(page.text, "html.parser")
                     
links = []
for link in soup.find_all('a'):
    links.append(link.get('href'))

recipe_urls = pd.Series([a.get("href") for a in soup.find_all("a")])

recipe_urls = recipe_urls[(recipe_urls.str.count("-")>0) & 
                          (recipe_urls.str.contains('/recipes/')==True) &
                          (recipe_urls.str.contains('-recipes/')==True) &
                          (recipe_urls.str.contains('course')==False) &
                          (recipe_urls.str.contains('books')==False) &
                          (recipe_urls.str.endswith('recipes/')==False)].unique()

# df = pd.DataFrame()

# df["recipe_urls"] = "https://www.jamieoliver.com" + df["recipe_urls"].astype("str")

# print(recipe_urls)

urls = []
for url in recipe_urls:
    recipe_link = "https://www.jamieoliver.com" + url
    urls.append(recipe_link)


for url in urls:
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    title = soup.find("h1").text.strip()

    ingredients = []
    for li in soup.select('.ingred-list li'):
        ingred = ' '.join(li.text.split())
        ingredients.append(ingred)

    directions = []
    allSteps = soup.find_all(attrs={"class": "recipeSteps"})
    for step in allSteps:
        directions.append(step.getText())
    
    recipeNo = recipeNo + 1

    attributes = [recipeNo, url, title,ingredients, directions]

    with open('data/allRecipe.csv', 'a+', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)  # creating a csv writer object
            writer.writerow(attributes)

print("ALL DONE YAY")
   