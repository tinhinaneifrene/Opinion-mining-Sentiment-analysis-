
import numpy as np #pour les calcules  d'algèbre linéaire
import pandas as pd #Pour la lecture du fichier Sentiment.csv de données

from keras.preprocessing.text import Tokenizer #le Tokenizer permet de transformer le texte en une séquence d'entiers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re #Pour les regex
from keras.utils.vis_utils import plot_model




#Récupération des données
#lecture du Airline_tweets.csv contenant le texte labellisé 
data = pd.read_csv('./data/Airline_Tweets.csv')
data = data[data.airline_sentiment != "neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) #enlever les caractères spéciaux (#,@...)
print(data[ data['airline_sentiment'] == 'Positive'].size)
print(data[ data['airline_sentiment'] == 'Negative'].size)

    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

# Creation du model
model = Sequential()
model.add(Embedding(max_fatures, embed_dim ,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax')) #softmax est la bonne méthode pour un réseau qui utilise une entropie croisée catégorielle

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
Y = pd.get_dummies(data['airline_sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
plot_model(model, to_file='modelUtiliser.png')

#Validation 
validation_size = 3000 # taille de phrase à tester 

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print('\033[32m' + "test score: %.2f" % (score))
print("test accuracy: %.2f" % (acc) + '\033[0m') 

# Test

TextToTest = ['she is the best ']
print("Phrase à tester ==> " + str(TextToTest))

#vectorization et padding du texte 
TextToTest= tokenizer.texts_to_sequences(TextToTest)
TextToTest = pad_sequences(TextToTest, maxlen=32, dtype='int32', value=0)

#le Vecteur de la phrase
print('\033[32m'+"Le vecteur de la phrase qui va servir d'entrée pour le model" + '\x1b[0m')
print(TextToTest)

sentiment = model.predict(TextToTest,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("Sentiment ==> negatif")
elif (np.argmax(sentiment) == 1):
    print("Sentiment ==> positif")

