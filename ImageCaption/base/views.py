from django.shortcuts import render
from django.http import HttpResponse
from keras.models import load_model
import os
import numpy as np 
import pickle

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical, plot_model

model = load_model('model/best_model11111.h5')
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    model = load_model('model/best_model11111.h5')
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

def home(request):
    return render(request, 'index.html')

vgg_model = VGG16() 
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
def captionbackend(request):
    print("Caption backend called")
    
    if 'image' not in request.FILES:
        error_message = "Please upload an image to generate a caption."
        return render(request, 'index.html', {'error_message': error_message})
    
    file = request.FILES['image']
    filename = file.name
    file_path = os.path.join('static/user_uploaded', filename)
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    caption = predict_caption(model, feature, tokenizer, max_length=35)
    print("Generated Caption:", caption)
    
    # Pass the caption and image path to the template
    return render(request, 'index.html', {'caption': caption, 'image_url': file_path})
