# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:14:14 2019

@author: hemanth
"""
import os
import json
import requests
from bs4 import BeautifulSoup
from recipe_scrapers import scrape_me

def get_recipe(url):
    try:
        scrap = scrape_me(url)
    except:
        print('Could not scrape URL {}'.format(url))
        return {}

    try:
        title = scrap.title()
    except AttributeError:
        title = None

    try:
        ingredients = scrap.ingredients()
    except AttributeError:
        ingredients = None

    try:
        instructions = scrap.instructions()
    except AttributeError:
        instructions = None

    try:
        picture_link = scrap.image()
    except AttributeError:
        picture_link = None
        
    try:
        soup = BeautifulSoup(requests.get(url).text,'html.parser')
        tags = []
        for i in soup.findAll('dt',{'itemprop': 'recipeCategory'}):
            tags.append(i.text)
    except AttributeError:
        tags = None

    return {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions,
        'picture_link': picture_link,
        'tags':tags
    }


def get_all_recipes(page_num):
    base_url = 'http://www.epicurious.com'
    search_url_str = 'search/?content=recipe&page'
    url = '{}/{}={}'.format(base_url, search_url_str, page_num)
    
    try:
        page = requests.get(url).text
        soup = BeautifulSoup(page,'html.parser')
        recipe_link_items = soup.select('div.results-group article.recipe-content-card a.view-complete-item')
        recipe_links = [r['href'] for r in recipe_link_items]
        return {base_url + r: get_recipe(base_url + r) for r in recipe_links}
    except Exception as e:
        print(e)
        
def save_recipes(recipes):
    p = os.path.join(os.getcwd(),'recipes_epicurious.json')
    
    with open(p,'a') as f:
        f.write(json.dumps(recipes))
        f.write(',') #take care of this. replace },{ in your json file with ,     for a smooth json.load()
        f.close()
    
    

if __name__ == '__main__':
    for i in range(1,2000,1):
        save_recipes(get_all_recipes(i))
    
