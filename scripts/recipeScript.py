import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm, tnrange, tqdm_notebook
tqdm.pandas()

from itertools import combinations
import matplotlib.pyplot as plt

from keras.preprocessing import image

import nltk
from nltk.corpus import stopwords

# nltk libraries
nltk.data.path.append(r'C:\Users\heman\Documents\DS\nltk_data')
nltk.download('stopwords', download_dir=r'C:\Users\heman\Documents\DS\nltk_data')

# FUnction to print the number of cuisines
def printNum(df1,list_cuisines, dish_type='cuisine'):
    '''function prints out number of recipes with titles contained in list_cuisines
    input:
        dish_type    : 'cuisine' or 'type'
        list_cuisines : e.g., ['Japanese','Italian', 'Chinese']
    '''
    temp = pd.DataFrame([], columns=[dish_type,'number'])
    for each in sorted(list_cuisines):
        num = df1[df1['title'].str.contains(each)].shape[0]
        temp = temp.append(pd.DataFrame([[each,num]], columns=[dish_type,'number']))

    return temp

# Function to parse text
def preprocessText(list_of_words):
    '''Steps to process string 
    Input:
        list_of_words - 
    '''
    from textblob import TextBlob
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk import wordpunct_tokenize, word_tokenize
    
    nltk.data.path.append(r'C:\Users\heman\Documents\DS\nltk_data')
    stop_words = stopwords.words('english')
    
    #doc = [w for w in doc.split()]                 # tokenizing
    #doc = [w for w in word_tokenize(doc) if w.lower() in words ] # Tokenize, lower, punctuation, 
    
    doc = [w.lower() for w in list_of_words]       # Lower the text.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if len(w)>=2]            # Remove single-letter words
    doc = [w for w in doc if w.isalnum()]  # punctuation removed
    #doc = [TextBlob(w).correct() for w in doc]     # Correct mis-spelling
    return doc


# Add more "ingredient"-related common words to the stopwords
stopwords_ingred = stopwords.words('english')
ingred_words = ['cup','cups','c','lb','lbs','pound','pounds',
                'gram','grams','g','cc','ccs','ml','mls',
                'ounce','oz','ounces',
                'teaspoon','teaspoons','tsp','tsps',
                'tablespoon','tablespoons','tbsp','tbsps',
                'box','boxes',
                'small','large','medium','half','whole',
                'splash','dash','sprigs','pinch',
                'can','cans','slice','slices',
                'pkg','package','packages','jar','jars',
                'chopped','sliced','grated','finely chopped','fresh',
                'freshly','freshly packed','firmly','firmly packed',
                'packed','halves','minced','chopped','peeled',
                'thinly','sliced','room temperature', 'ground',
                'cut','inch','shredded','grass','fed'
               ]
stopwords_ingred.extend(ingred_words)


# Create a function that returns a dataframe where the colName is
def convertLowerTags(df_in, list_colNames =['tag_ingredient','tag_special','tag_tech', 'tag_tag']):
    '''Function returns df, where words in list_colName are converted into lowercase;
    Empty list [] or 0 zeros are labeled [none]
    Input:
        df_in     -  dataframe in
        colName   -  e.g., 'tag_tag'
    '''
    for each in list_colNames:
        temp = df_in.loc[:,[each]]
        print('working on ',each)
        for i in range(len(temp)):

            if temp.loc[i,each] ==[]:
                temp.at[i,each] = ['none']
            if temp.loc[i,each] ==0:
                temp.at[i,each] = ['none']
            else:
                li = temp.loc[i, each]
                temp.at[i, each] = [str(w).lower() if li != 0 else ('none') for w in li]
        df_in[each] = temp
    return df_in

# Create a function to strip secondary components in the ingredients
# e.g., salt, pepper, italian seasoning, garlic powder, etc.
def removeSecondary(list_ingred):
    '''Function returns a list of ingredients with secondary components removed
    E.g., salt, pepper, seasonings, etc.
    Input:
        list_Ingred - list of ingredients
    '''
    list_secondary = ['salt','pepper','powder',
                      'oil','butter','margarine','mustard','cayenne',
                      'water ','ketchup','sugar','baking soda','cinnamon',
                      'dried basil','dried parsley','ground cumin','cumin',
                      'italian seasoning','paprika', 'mayonnaise', 'garlic',
                      'onion','shoyu','olive oil','vinegar','soy sauce'
                      ]

    list_out =[]
    for sent in list_ingred:
        if any(word.lower() in sent for word in list_secondary):
            pass
        else:
            list_out.append(sent)
    return list_out

# Pre-processing list of sentences
def preprocessText(list_sentences, stopwords_ingred=stopwords_ingred):
    '''Process list of sentences
    Input:
        list_sentences  : list of sentences
        stopwords_ingred: the stopwords
    '''
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk import wordpunct_tokenize, word_tokenize
    nltk.data.path.append(r'C:\Users\heman\Documents\DS\nltk_data')

    sent = ', '.join(list_sentences)
    doc = [w.lower() for w in wordpunct_tokenize(sent)]
    doc = [w for w in doc if w.isalpha()]  # punctuation removed
    doc = [w for w in doc if not w in stopwords_ingred]  # Remove stopwords.
    doc = [w for w in doc if len(w)>=2]     # Remove words with only a single letter

    return doc


# Function to filter a dataframe based on a list of ingredients
# to check if there's an ingredient in a recipe of df_in
def filterbyIng(df_in, list_ing):
    '''function returns another dataframe that includes recipes with, at least, one component from list_ing
    the output dataframe contains at least one ingredient
    Input:
        df_in    : dataframe to examine
        list_ing : list of ingredients,e.g., ['cheese','lettuce','tomato']
    '''
    from itertools import combinations

    df_in['ingred_string'] = df_in['ing'].apply(lambda x: ' '.join(x))

    df_out = pd.DataFrame([],columns= df_in.columns)
    for each in list_ing:
        df_containing = df_in[df_in['ingred_string'].str.contains(each)]
        df_out = df_out.append(df_containing)
        df_out = df_out.drop_duplicates(subset='id')

    return df_out

# A function to find recipes with intersections of ingredients  (in df_in)
def intersect(df_in, list_ingred=['bok choy','garlic', 'shoyu']):
    '''Function to return a dataframe containing recipes, ordered by the measure of intersections. e.g.,
    If 3 ingredients, then it respectively outputs recipes with 3-way intersection, 2-way, no intersection
    Input:
        df_in       : initial dataframe
        list_ingred : list of 3 ingredients to include, ['bok choy','garlic','shoyu']
    '''
    from functools import reduce
    from itertools import combinations

    # Import list
    df_in['ingred_list'] = df_in['ingredients'].progress_apply(preprocessText)

    # Join the list into a full sentence (string)
    df_in['ingred_string'] = df_in['ingred_list'].apply(lambda x: ' '.join(x))

    intersect_ids = []
    # Make a dictionary containing dataframes,w/ key:val => index: id's (in dataframe)
    dict_dfs = dict()
    for ind, ingred in enumerate(list_ingred):
        # Build dataframe
        df_out = pd.DataFrame([],columns= df_in.columns)
        df_containing = df_in[df_in['ingred_string'].str.contains(ingred)]
        df_out = df_out.append(df_containing)
        # append ids of dataframe to the dictionary
        dict_dfs[ind] = df_out.id.values.tolist()
    print(f'1. Recipes with {list_ingred[0]}: {len(dict_dfs[0])}, {list_ingred[1]}: {len(dict_dfs[1])}, {list_ingred[2]}: {len(dict_dfs[2])}')
    # First find id's with 3-way intersections
    ids_3way = list(set.intersection(*map(set, [dict_dfs[0],dict_dfs[1],dict_dfs[2]])))
    print('2. Recipes using all 3 ingredients: ', len(ids_3way))
    df_dummy = pd.DataFrame([], columns=df_in.columns)
    df_3way = df_dummy.append(df_in[df_in.id.isin(ids_3way)])
    df_3way['num_ingred'] = df_3way['ingred_string'].apply(lambda sent: len(sent.split(' ')))
    try:
        df_3way = df_3way.sort_values(by=['num_ingred'], ascending=True).reset_index(drop=True)
    except KeyError:
        print('flag: cannot sort, because 3-way intersection df is empty')

    # Then find id's with 2-way intersections
    ids_2way = []
    for x,y in combinations([dict_dfs[0],dict_dfs[1],dict_dfs[2]], 2):
        ids_2way += list(set.intersection(*map(set, [x,y])))
        ids_2way = list(set(ids_2way))
    print('3. Recipes using 2 of 3 ingredients: ', len(ids_2way))
    df_2way = df_dummy.append(df_in[df_in.id.isin(ids_2way)])
    df_2way['num_ingred'] = df_2way['ingred_string'].apply(lambda sent: len(sent.split(' ')))
    try:
        df_2way= df_2way.sort_values(by=['num_ingred'], ascending=True)
    except KeyError:
        print('flag: cannot sort, because 2-way intersection df is empty')

    # Finally, add individual
    for ingred in list_ingred:
        df_each = df_dummy.append(df_in[df_in['ingred_string'].str.contains(ingred)])
    df_each = df_each.drop_duplicates(subset='id')
    print('4. Recipes using 1 of 3 ingredients: ',len(df_each))
    df_each['num_ingred'] = df_each['ingred_string'].apply(lambda sent: len(sent.split(' ')))
    try:
        df_each = df_each.sort_values(by=['num_ingred'], ascending=True)
    except KeyError:
        print('flag: cannot sort, because this indiv-df is empty')

    # concat all df's and
    df_intersect = df_3way.append(df_2way).append(df_each)
    df_intersect = df_intersect.drop_duplicates(subset='id')
    print('5. Total recipe collection: ', len(df_intersect))

    return df_3way.reset_index(drop=True), df_2way.reset_index(drop=True), df_each.reset_index(drop=True), df_intersect.reset_index(drop=True)


# Function to show pictures in directories (Path)
def displayPhoto(path):
    img_width, img_height= 224,224
    img = image.load_img(path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.                    # Normalize to [0,1] for plt.imshow application
    plt.imshow(img_tensor)
    plt.show()

def showcasePhotos(df_in):
    '''Function returns photos (w/ Title) and ingredients
    It will output the same number of figures as the rows in df_in
    Input:
        df_in  - dataframes
    '''
    total_row = df_in.shape[0]
    df_in = df_in.copy().reset_index(drop=True)

    for i in range(total_row):

        try:
            print(f'{i+1}. Picture-{i+1}')
            temp = df_in.loc[[i],'picture_link'].apply(lambda link: link.split('/')[-1])
            imgid = temp.values[0]
            path_file = f'images/epicurious_images/{imgid}'

            plt.title(df_in.loc[[i],'title'].values[0])
            displayPhoto(path_file)

            listIng = df_in.loc[[i],'ingredients'].values[0]
            listIngMain = removeSecondary(df_in.loc[[i],'ingredients'].values[0])
            print('All  Ingredients:', *listIng,  '\n',sep='\n')
            print('Main Ingredients:', *listIngMain,  '\n',sep='\n')

        except AttributeError:
            print( "   Unfortunately, there's no image for the `%s` recipe\n" %(df_in.loc[[i],'title'].values[0]) )
            listIng = df_in.loc[[i],'ingredients'].values[0]
            listIngMain = removeSecondary(df_in.loc[[i],'ingredients'].values[0])
            print('All  Ingredients:', *listIng,  '\n',sep='\n')
            print('Main Ingredients:', *listIngMain,  '\n',sep='\n')


# Function to convert string format of nutritional content '4 g(20%)' into floats
# i.e., 4.0
def convertGrams(df_subset):
    '''Function returns a df, with '_content' column converted into float(mg)
    '''

    list_col = [each for each in df_subset.columns.tolist() if 'content' in each]

    for colname in list_col:
        if colname != 'sodium_content':
            print('converting',colname)
            for i in tqdm_notebook(range(df_subset.shape[0]), desc='loop'):
                if df_subset.loc[i, colname] != 0:
                    gramHold = df_subset.loc[i, colname].split('g')[0]
                    df_subset.at[i,colname] = float(gramHold)
                else:
                    df_subset.at[i,colname] = float(0)
        else:
            print('converting',colname)
            for i in tqdm_notebook(range(df_subset.shape[0]), desc='loop'):
                if df_subset.loc[i, colname] != 0:
                    gramHold = df_subset.loc[i, colname].split('mg')[0]
                    df_subset.at[i,colname] = float(gramHold)
                else:
                    df_subset.at[i,colname] = float(0)
    return df_subset



# Create a Keto-category: 'yes' or 'no', based on fat_calories making up >70% of total_calories
def ketoPercent(df_subset, threshold=0.70):
    '''Function to
    Input:
        df_subset   - is the dataframe containing 'fat_cal' and 'total' columns
        threshold   - cutoff for %-fat_cal over total_cal considered keto (e.g., 70%)
    '''
    for ind in range(df_subset.shape[0]):
        if df_subset.loc[ind,'fat_cal'] == 0:
            pass
        else:
            if df_subset.loc[ind,'total'] ==0:
                pass
            else:
                if (df_subset.loc[ind,'fat_cal'] / df_subset.loc[ind,'total'])>= threshold:
                    df_subset.at[ind,'ketolikely'] = 'yes'
                else:
                    pass
    return df_subset

# Function to return a list of unique 'tags_'
def countUniqueTag(df_subset, colName='tag_tech' ):
    '''Function to count total unique keywords in colName
    Input:
        df_subset   - input dataframe containing 'tag_' columns
        colName     - column name with 'tag_', e.g., 'tag_tech'
    '''
    collect_tech=[]
    for row in range(df_subset.shape[0]):
        for list_tech in df_subset.loc[[row],colName]:
            for tech in list_tech:
                if tech not in collect_tech:
                    collect_tech.append(tech)
                else:
                    pass
    return collect_tech, len(collect_tech)

# Function to re-prioritize the order of rows appearing in the dataframe,
# based on the 'tags_' selected
# create a function to prioritize

def orderRecipeBy(df_in, dict_tags = {'tag_cuisine':['japanese','chinese']}):
    ''' Function returns df, filtered by tags in dict_tags.
    The function temporarily removes observations without the tags and appended them to the end
    Input:
        df_in     -  dataframe containing 'tag_'-columns
        dict_tags -  dictionary of tags, e.g., {'tag_cuisine':['japanese','chinese']}
    '''
    df_prio = pd.DataFrame([], columns=df_in.columns)
    df_deprio = pd.DataFrame([], columns=df_in.columns)

    # For each observation (row in df), check if tags (in dict_tags) are found
    # If so, add the observation to the temporary folder
    for irow in range(df_in.shape[0]):

        for key,tagList in dict_tags.items():
            for eachTag in tagList:

                ith_observation = df_in.loc[[irow], key]
                if eachTag in ith_observation.values[0]:
                    df_prio = pd.concat([df_prio, df_in.loc[[irow],:]])
                else:
                    df_deprio = pd.concat([df_deprio, df_in.loc[[irow],:]])
    print('Number of observations w/ given tag: ', df_prio.shape[0])
    df_out = df_prio.append(df_deprio)
    return df_out.drop_duplicates(subset='id')


# Change strangeeWord in the tagList into correctWord
# Create a function that returns a dataframe where the colName is
def convertStdLetters(df_in, colName ='tag_source', strangeWord='bon appétit', replaceWith='bon appetit'):
    '''Function returns df, where words in colName are converted into std letters;
    Input:
        df_in       -  dataframe in
        colName     -  'tag_source' column
        strangeWord -  word to replace e.g., 'bon appétit'
        replaceWith -  replacement word, e.g., 'bon appetit'
    '''
    print('checking ',colName)
    df_out = df_in.copy()

    for i in range(df_in.shape[0]):
        li = df_out.loc[i, colName]
        if li == 0:
            lout = ['none']
        else:
            lout = [str(w).replace(strangeWord, replaceWith) for w in li]

        df_out.at[i,colName] = lout

    return pd.DataFrame(df_out)

# The final function
def outputRecipes(df_in):
    '''Function returns another dataframe of only relevant data
    Input:
        df_in   - dataframe of interest
    Output:
        df_out  - containing 'title','ingredient','instructions'
        links   - list of url's for each recipe
        images  - list of image_paths
    '''
    df_out = df_in[['title','ingredients','instructions']]
    links  = [str(url) for url in df_in.loc[:,'url']]
    #images = [str(each).split('/')[-1] for each in df_in.loc[:,'picture_link']]
    #image_paths = [f'/images/epicurious_images/{u}' for u in images]

    return df_out, links#, image_paths

# Function to create a label of recipes, based on modeled Topics
def labelTopic(df_in):
    '''Function returns a dataframe with labels based on Topics 
    Input:
        df_in  - dataframe to input
    '''
    df_in_ = df_in.copy()
    df_in_['label'] = ''
    for irow in tnrange(df_in_.shape[0]):
        label_index = np.argmax(np.asarray(df_in_.iloc[irow,:].tolist()))
        label_name = str(df_in_.columns[label_index])
        df_in_.iloc[irow,-1] = label_name
    return df_in_

# Adding a 5-category-columns 'mainCat' in the original df_all
def subcatToMaincat(df_in):
    '''Function returns a dataframe with labels  
    Input:
        df_in    -  dataframe
    '''
    df_in['mainCat'] = ''
    for irow in tnrange(df_in.shape[0]):
        if df_in.at[irow,'subCat'] in ['3','9','15']:
            df_in.at[irow,'mainCat'] = 'maindish'
            
        elif df_in.at[irow,'subCat'] in ['6','7','11','12','13','14','16','17','18','19','20']:
            df_in.at[irow,'mainCat'] = 'sidedish'

        elif df_in.at[irow,'subCat'] in ['4','5','8','10']:
            df_in.at[irow,'mainCat'] = 'dessert'

        elif  df_in.at[irow,'subCat'] in ['1']:
            df_in.at[irow,'mainCat'] = 'condiments'

        elif  df_in.at[irow,'subCat'] in ['2']:
            df_in.at[irow,'mainCat'] = 'salad'

    return df_in