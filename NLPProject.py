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
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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


obswrds = []
clrwrds = []
model = fasttext.load_model("unigram.bin")

#  This goes through the input file and adds each sentence to either obscene  or clear words 
for j in ds:
    count = count + 1
    
    txt = str(tf.keras.backend.get_value(j['text']).decode("utf-8"))
    txt2 = ""
    # Im not gappt I had to do this
    for letter in txt:
        if letter not in "`~1!2@3#4$5%6^7&8*9(0)-_=+qQwWeErRtTyYuUiIoOpP[{]}\|aAsSdDfFgGhHjJkKlL;:'zZxXcCvVbBnNmM,<.>/? ":
            continue
        txt2 = txt2 + letter
    wrds.append(txt2)
    
    #If it desesrves an obscene label
    if tf.keras.backend.get_value(j['obscene']) >= .5  or tf.keras.backend.get_value(j)["insult"] >= .5 or tf.keras.backend.get_value(j['sexual_explicit']) >= .5 or tf.keras.backend.get_value(j['severe_toxicity']) >= .5:
        #Add the sentence to the obscene list
        obscene.append(model.get_sentence_vector(txt2))
        
        #Add each word as an obscene word
        for w in txt2.split(" "):
            xwords.append(model.get_sentence_vector(w))
            ywords.append("obscene")
            obswrds.append(model.get_sentence_vector(w))
                    
        obscnt = obscnt + 1
    #If its clear
    else:
        
        #Add each clear sentence to the clear list
        clear.append(model.get_sentence_vector(txt2))
        for w in txt2.split(" "):
            
            #add each word as a clear word
            xwords.append(model.get_sentence_vector(w))
            ywords.append('clear')
            clrwrds.append(model.get_sentence_vector(w))

        clrcnt = clrcnt + 1
        
    #stop at 10000 words
    if count == 10005:
        break


#Add an equal amount of clear and obscene sentences to x and y to avoid oversampling
for i in range(600):
    x.append(obscene[i])
    y.append("obscene")
    x.append(clear[i])
    y.append("clear")
    

xwords.clear()
ywords.clear()

#Add an equal amount of words to xwords and ywords to avoid oversampling
for i in range(20000):
    xwords.append(obswrds[i])
    ywords.append("obscene")
    xwords.append(clrwrds[i])
    ywords.append("clear")
        
### SENTENCE TRAINING

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.fit_transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

#split into test and train
xtrain = x[0:1100]
xtest = x[1100:1200]
ytrain = dummy_y[0:1100]
ytest = dummy_y[1100:1200]


#uncomment for randomforest, comment neural network
'''
clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(np.array(xtrain),np.array(ytrain))
print(xtest)
ypred = clf.predict(np.array(xtest))

print("accuracy: ", sklearn.metrics.accuracy_score(np.array(ytest),ypred))

'''



#Use the neural network. Comment this for randomforest
model2 = Sequential()


#model2.add(layers.Embedding(input_dim = 8000, output_dim=32))
#model2.add(layers.LSTM(64,input_shape=(100,2), activation='relu'))
model2.add(Dense(64, activation = "relu"))
model2.add(Dense(32, activation = "relu"))
model2.add(Dense(16, activation = "relu"))

model2.add(layers.Dropout(.2))
model2.add(Dense(2, activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = .0003), metrics=['accuracy'])
model2.fit(np.array(xtrain),np.array(ytrain),epochs=200, batch_size = 30)

model2.save("sentenceModel")

_, accuracy = model2.evaluate(np.array(xtest), np.array(ytest))
print(accuracy * 100)




## WORDS TRAINING
encoderwords = LabelEncoder()
encoderwords.fit(ywords)
encoded_ywords = encoderwords.fit_transform(ywords)
dummy_ywords = np_utils.to_categorical(encoded_ywords)

#split into test and train
xwordstrain = xwords[0:19000]
xwordstest = xwords[19000:20000]
ywordstrain = dummy_ywords[0:19000]
ywordstest = dummy_ywords[19000:20000]


model3 = Sequential()

#model3.add(layers.Embedding(input_dim = 100000, input_shape = (100,1), output_dim=64))
#model3.add(layers.LSTM(64, activation='relu'))
model3.add(Dense(64, activation = "relu"))
model3.add(Dense(32, activation = "relu"))
model3.add(Dense(16, activation = "relu"))

model3.add(Dense(2, activation = 'sigmoid'))
model3.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(.0003), metrics=['accuracy'])
model3.fit(np.array(xwordstrain),np.array(ywordstrain),epochs=300, batch_size = 50)

#model3.save("wordModel")
_, accuracy2 = model3.evaluate(np.array(xwordstest), np.array(ywordstest))

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
    #break
print(accuracy * 100)
print(accuracy2 * 100)
print("END")