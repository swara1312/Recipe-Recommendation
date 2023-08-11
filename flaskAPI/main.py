from flask import Flask, render_template, jsonify, request
import json
import requests
import pickle
import pandas as pd
import logging
import sys
sys.path.append("working")
import recommending


app = Flask(__name__,template_folder='template')


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")


@app.route('/recipe', methods=["GET", "POST"])
def recipe():
    ingredients = request.args.get('ingredients')
    recipe = recommending.RecSys(ingredients)
    # logging.warning(ingredients)

    # We need to turn output into JSON.
    recipes = {}
    for index, row in recipe.iterrows():
        # logging.warning(row)
        recipes[index] = {
            'recipe': str(row['recipe']),
            'score': float(row['score'])*100,
            'ingredients': str(row['ingredients']),
            'url': str(row['url']),
            'directions': row['directions']
        }
    return render_template("result.html", result=recipes)


if __name__ == "__main__":
    app.run(debug=True)