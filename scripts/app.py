# Dependencies
from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from kerasUtil import *
from recipeScript import *
from miscScript import *
import os
import subprocess
import keras.backend.tensorflow_backend as tb


# Your API definition
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    #running the windows_script for loading the images into test dir
    #os.system('python windows_script.py')
    subprocess.call("python windows_script.py",shell=True)
    # Specify directory of test data (images)
    image_path_dir = 'data/test/'

    # Loop over each file
    result=[]
    for pic_file in os.listdir(image_path_dir):    
        image = load_img(image_path_dir+pic_file, target_size=(224,224))
        image = img_to_array(image) / 255
        image = np.expand_dims(image, axis=0)
        
        # extractor is called above
        tb._SYMBOLIC_SCOPE.value = True #some error that was fixed, caused by keras
        features = extractor.predict(image)
        # predictor
        class_predicted = predictor.predict_classes(features)
        probabilities = predictor.predict_proba(features)
        
        inID = class_predicted[0]
        label = inv_map[inID]
        
        result.append(label)
    
    predictions = 'The images are: ' + ', '.join(result)
    print(predictions)
    text = "<strong> {} </strong>".format(predictions)

    # Find recipes with intersections
    d3, d2, d1, d = intersect(df, result)
    # Output recipes
    out3, link3 =outputRecipes(d3)
    out2, link2 =outputRecipes(d2)
    out1, link1 =outputRecipes(d1)

    text = text + "<p><strong> Here are the top recipes for all 3 ingredients:<strong></p>"

    print('Here are the top recipes for all 3 ingredients: ','\n')
    for h,l in zip(out3.title.head(3),link3):
        z = "<p><a href={}>{}</a></p>".format(l,h)
        text = text + z
        print(h, end=" ")
        print(l,'\n')
        
    print('Here are the top recipes for 2 of 3 ingredients: ','\n','\n')
    text = text + "<strong>" + "Here are the top recipes for 2 of 3 ingredients:"+"</strong>"
    for h,l in zip(out2.title.head(3),link2):
        z = "<p><a href={}>{}</a></p>".format(l,h)
        text = text + z
        print(h, end=" ")
        print(l,'\n')

    print('Here are the top recipes for (at least) 1 of 3 ingredients: ','\n','\n')
    text = text + "<strong>" + "Here are the top recipes for (at least) 1 of 3 ingredients:"+"</strong>"
    for h,l in zip(out1.title.head(3),link1):
        z = "<p><a href={}>{}</a></p>".format(l,h)
        text = text + z
        print(h, end=" ")
        print(l,'\n')

    return text



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    
    with open('df_epi_ingred.pkl','rb') as fin:
        df = pickle.load(fin)
    extractor = callModel('monet') #kerasUtil.py
    predictor = models.load_model('models/trained_model.hdf5')  #load_model is from keras
    class_dictionary = np.load('models/train_class_indices.npy').item()
    inv_map = {v:k for k,v in class_dictionary.items()}

    app.run(port=port, debug=True)
