#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 07:20:27 2023

@author: kleveri2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:35:53 2023

@author: kleveri2
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import fasttext
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import layers
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
print(stops)


ds = tfds.load("huggingface:civil_comments", split = 'train')

count = 0
file1 = open("test.txt","w")



x = []
y = []
xwords = []
ywords = []
wrds = []
count = 0
obscene = []
clear = []
clrcnt = 0
obscnt = 0

wordSents = {}

obsceneSentence = []

obswrds = []
clrwrds = []
model = fasttext.load_model("bigram.bin")

#  This goes through the input file and adds each sentence to either obscene  or clear words 
for j in ds:
    count = count + 1
    print(count)
    txt = str(tf.keras.backend.get_value(j['text']).decode("utf-8"))
    txt2 = ""
    # Im not gappt I had to do this
    for letter in txt:
        if letter not in "`~1!2@3#4$5%6^7&8*9(0)-_=+qQwWeErRtTyYuUiIoOpP[{]}\|aAsSdDfFgGhHjJkKlL;:'zZxXcCvVbBnNmM,<.>/? ":
            continue
        txt2 = txt2 + letter
        
    
    wrds.append(txt2)
    
    for i in stops:
        i2 = " " + i + " "
        txt2 = txt2.replace(i2, " ")
        
        i3 = " " + i.capitalize() + " "
        txt2 = txt2.replace(i2, " ")
    
    vec = model.get_sentence_vector(txt2)
    sentencearr = np.array(vec)

    isalnum = 0
    hasexclam = 0
    hasat = 0
    hashash = 0
    hasmoney = 0
    haspercent = 0
    hascarat = 0
    hasand = 0
    hasstar = 0


    for i in txt2.split(" "):
        if i.isalnum():
            isalmum = 1
        if '!' in i:
            hasexclam = 1
        if '@' in i:
            hasat = 1
        if '#' in i:
            hashash = 1
        if '$' in i:
            hasmoney = 1
        if '%' in i:
            haspercent = 1
        if '^' in i:
            hascarat = 1
        if '&' in i:
            hasand = 1
        if '*' in i:
            hasstar = 1
    sentencearr = np.append(sentencearr,isalmum)
    sentencearr = np.append(sentencearr,hasexclam)
    sentencearr = np.append(sentencearr,hasat)
    sentencearr = np.append(sentencearr,hashash)
    sentencearr = np.append(sentencearr,hasmoney)
    sentencearr = np.append(sentencearr,haspercent)
    sentencearr = np.append(sentencearr,hascarat)
    sentencearr = np.append(sentencearr,hasand)
    sentencearr = np.append(sentencearr,hasstar)
        
    #print(txt2)
    #If it desesrves an obscene label
    if tf.keras.backend.get_value(j['obscene']) >= .5  or tf.keras.backend.get_value(j)["insult"] >= .5 or tf.keras.backend.get_value(j['sexual_explicit']) >= .5 or tf.keras.backend.get_value(j['severe_toxicity']) >= .5:
        
        obsceneSentence.append(txt2)
        
        #Add the sentence to the obscene list
        obscene.append(sentencearr)
        
        #Add each word as an obscene word
        for w in txt2.split(" "):
            if w in stops:
                continue
            if w == " ":
                continue
            
            if w not in wordSents.keys():
                wordSents[w] = [0,1]
            else:
                wordSents[w][1] = wordSents[w][1] + 1
                
                
            xwords.append(model.get_sentence_vector(w))
            ywords.append("obscene")
            obswrds.append(w)
                    
        obscnt = obscnt + 1
    #If its clear
    else:
        
        #Add each clear sentence to the clear list
        clear.append(sentencearr)
        for w in txt2.split(" "):
            if w in stops:
                continue
            if w == " ":
                continue
            
            if w not in wordSents.keys():
                wordSents[w] = [1,0]
            else:
                wordSents[w][0] = wordSents[w][0] + 1
            
            #add each word as a clear word
            xwords.append(model.get_sentence_vector(w))
            ywords.append('clear')
            #clrwrds.append((w))

        clrcnt = clrcnt + 1
        
    #stop at 10000 words
    if count == 100000:
        break
print(wordSents)

print("HERE:::: ", len(obscene))


#Add an equal amount of clear and obscene sentences to x and y to avoid oversampling
for i in range(6250):
    x.append(obscene[i])
    y.append("obscene")
    x.append(clear[i])
    y.append("clear")
    print(obscene[i])

print("HERE:::: ", len(obscene))
    

xwords.clear()
ywords.clear()


### SENTENCE TRAINING

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.fit_transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

#split into test and train
xtrain = x[0:10000]
xtest = x[10000:12500]
ytrain = dummy_y[0:10000]
ytest = dummy_y[10000:12500]


#uncomment for randomforest, comment neural network
'''
clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(np.array(xtrain),np.array(ytrain))
print(xtest)
ypred = clf.predict(np.array(xtest))

print("accuracy: ", sklearn.metrics.accuracy_score(np.array(ytest),ypred))

'''

print(len(ytrain), len(ytest))


#Use the neural network. Comment this for randomforest
model2 = Sequential()


#model2.add(layers.Embedding(input_dim = 8000, output_dim=32))
#model2.add(layers.LSTM(64,input_shape=(100,2), activation='relu'))
#model2.add(Dense(512, activation = "relu"))

model2.add(Dense(128, activation = "relu"))
model2.add(Dense(64, activation = "relu"))
#model2.add(layers.Dropout(.2))
#model2.add(layers.Flatten())

model2.add(Dense(32, activation = "relu"))
#model2.add(Conv2D(32, (1,1), activation="relu", padding = 'same'))
model2.add(Dense(16, activation = "relu"))

model2.add(layers.Dropout(.2))
model2.add(Dense(2, activation = 'softmax'))

model2.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = .0003), metrics=['accuracy'])
#model2.fit(np.array(xtrain),np.array(ytrain),epochs=200, batch_size = 90)

#model2.save("sentenceModel")

print(len(ytrain), len(ytest))

#_, accuracy = model2.evaluate(np.array(xtest), np.array(ytest))
accuracy = 0
print(accuracy * 100)




## WORDS TRAINING

tempclrwrds = []
tempobswrds = []
#Add an equal amount of words to xwords and ywords to avoid oversampling
for i in obswrds:
    
    vec = model.get_sentence_vector(i)
    wordarr = np.array(vec)

    isalnum = 0
    hasexclam = 0
    hasat = 0
    hashash = 0
    hasmoney = 0
    haspercent = 0
    hascarat = 0
    hasand = 0
    hasstar = 0


    if i.isalnum():
        isalmum = 1
    if '!' in i:
        hasexclam = 1
    if '@' in i:
        hasat = 1
    if '#' in i:
        hashash = 1
    if '$' in i:
        hasmoney = 1
    if '%' in i:
        haspercent = 1
    if '^' in i:
        hascarat = 1
    if '&' in i:
        hasand = 1
    if '*' in i:
        hasstar = 1
    wordarr = np.append(wordarr,isalmum)
    wordarr = np.append(wordarr,hasexclam)
    wordarr = np.append(wordarr,hasat)
    wordarr = np.append(wordarr,hashash)
    wordarr = np.append(wordarr,hasmoney)
    wordarr = np.append(wordarr,haspercent)
    wordarr = np.append(wordarr,hascarat)
    wordarr = np.append(wordarr,hasand)
    wordarr = np.append(wordarr,hasstar)
    
    
    sent = wordSents[i]
    if (wordSents[i][0]) >= (wordSents[i][1]):
        #xwords.append(model.get_sentence_vector(w))
        tempclrwrds.append(wordarr)
        
        #ywords.append("clear")
        print("clear: ", i)
    else:
        #xwords.append(model.get_sentence_vector(w))
        tempobswrds.append(wordarr)
        #ywords.append("obscene")
        print("obscene: ", i)
        
print(len(tempclrwrds), len(tempobswrds))


#Add an equal amount of clear and obscene sentences to x and y to avoid oversampling
for i in range(15988):
    xwords.append(tempobswrds[i])
    ywords.append("obscene")
    xwords.append(tempclrwrds[i])
    ywords.append("clear")
print(len(ywords))
        
'''
    xwords.append(obswrds[i])
    ywords.append("obscene")
    xwords.append(clrwrds[i])
    ywords.append("clear")
    '''
print(len(ywords))
encoderwords = LabelEncoder()
encoderwords.fit(ywords)
encoded_ywords = encoderwords.fit_transform(ywords)
dummy_ywords = np_utils.to_categorical(encoded_ywords)

#split into test and train
xwordstrain = xwords[0:12790]
xwordstest = xwords[12790:31976]
ywordstrain = dummy_ywords[0:12790]
ywordstest = dummy_ywords[12790:31976]


model3 = Sequential()

#model3.add(layers.Embedding(input_dim = 100000, input_shape = (100,1), output_dim=64))
model3.add(Dense(128, activation = "relu"))
#model3.add(Dense(128, activation = "relu"))

#model3.add(layers.LSTM(64, activation='relu'))
model3.add(Dense(64, activation = "relu"))
model3.add(Dense(32, activation = "relu"))
model3.add(Dense(16, activation = "relu"))

model3.add(Dense(2, activation = 'sigmoid'))
model3.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(.0003), metrics=['accuracy'])
model3.fit(np.array(xwordstrain),np.array(ywordstrain),epochs=200, batch_size = 50)

#model3.save("wordModel")
_, accuracy2 = model3.evaluate(np.array(xwordstest), np.array(ywordstest))

#print(model3.predict(np.array(model.get_sentence_vector("dog")).reshape(1,-1)))
#print(model3.predict(np.array(model.get_sentence_vector("F**K")).reshape(1,-1)))

'''

clf2 = RandomForestClassifier(n_estimators=500, random_state=0)
clf2.fit(np.array(xwordstrain),np.array(ywordstrain))
#print(xwordstest)
ywordspred = clf.predict(np.array(xwordstest))

print("accuracy: ", sklearn.metrics.accuracy_score(np.array(ytest),ypred))
'''

''' IGNORE THIS RIGHT NOW
#model3.summary()

#a = np.array(model.get_sentence_vector("F**K"))
#a =a.reshape(1,-1)
#print(a)
#result = model3.predict(a)
#print(result)

#print(len(obscene))

maxi = 0
maxistr = ""
for badword in obscene:
    for word in badword.split():
        print(word)
        #print(model.get_word_vector(word))
        #print(np.array(model.get_sentence_vector(word)))
        #print(np.array(model.get_sentence_vector(word)).shape)
        a = np.array(model.get_sentence_vector(word))
        a =a.reshape(1,-1)
        #print(a)
        result = model3.predict(a)
        print(result)
        if result[0][0] < result[0][1]:
            print(word, "obscene")
        if result[0][1] > maxi:
            maxi = result[0][1]
            maxistr = word
            print(word)
print(maxistr)
'''
for sentence in obsceneSentence:
    
    for i in sentence.split(" "):
        vec = model.get_sentence_vector(i)
        wordarr = np.array(vec)
    
        isalnum = 0
        hasexclam = 0
        hasat = 0
        hashash = 0
        hasmoney = 0
        haspercent = 0
        hascarat = 0
        hasand = 0
        hasstar = 0
    
    
        if i.isalnum():
            isalmum = 1
        if '!' in i:
            hasexclam = 1
        if '@' in i:
            hasat = 1
        if '#' in i:
            hashash = 1
        if '$' in i:
            hasmoney = 1
        if '%' in i:
            haspercent = 1
        if '^' in i:
            hascarat = 1
        if '&' in i:
            hasand = 1
        if '*' in i:
            hasstar = 1
        wordarr = np.append(wordarr,isalmum)
        wordarr = np.append(wordarr,hasexclam)
        wordarr = np.append(wordarr,hasat)
        wordarr = np.append(wordarr,hashash)
        wordarr = np.append(wordarr,hasmoney)
        wordarr = np.append(wordarr,haspercent)
        wordarr = np.append(wordarr,hascarat)
        wordarr = np.append(wordarr,hasand)
        wordarr = np.append(wordarr,hasstar)
        
        
        result = model3.predict(wordarr.reshape(1,-1), verbose = 0)
        if result[0][1] > result[0][0]:
            sentence = sentence.replace(i, "****")
    print(sentence)
'''
for i in obswrds:
    vec = model.get_sentence_vector(i)
    wordarr = np.array(vec)

    isalnum = 0
    hasexclam = 0
    hasat = 0
    hashash = 0
    hasmoney = 0
    haspercent = 0
    hascarat = 0
    hasand = 0
    hasstar = 0


    if i.isalnum():
        isalmum = 1
    if '!' in i:
        hasexclam = 1
    if '@' in i:
        hasat = 1
    if '#' in i:
        hashash = 1
    if '$' in i:
        hasmoney = 1
    if '%' in i:
        haspercent = 1
    if '^' in i:
        hascarat = 1
    if '&' in i:
        hasand = 1
    if '*' in i:
        hasstar = 1
    wordarr = np.append(wordarr,isalmum)
    wordarr = np.append(wordarr,hasexclam)
    wordarr = np.append(wordarr,hasat)
    wordarr = np.append(wordarr,hashash)
    wordarr = np.append(wordarr,hasmoney)
    wordarr = np.append(wordarr,haspercent)
    wordarr = np.append(wordarr,hascarat)
    wordarr = np.append(wordarr,hasand)
    wordarr = np.append(wordarr,hasstar)
    
    
    result = model3.predict(wordarr.reshape(1,-1), verbose = 0)
    if result[0][1] > result[0][0]:
        print("OBSCENE: ", i)

    #break
    '''
print(accuracy * 100)
print(accuracy2 * 100)
print("END")