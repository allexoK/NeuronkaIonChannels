import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.layers as L
from tensorflow.keras import Model
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

#Nacitani dat
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv", dtype=dict(time=str))

#Priprava dat
n_classes = train.open_channels.unique().shape[0]
seq_len = 1000
X = train.signal.values.reshape(-1, seq_len, 1)
y = train.open_channels.values.reshape(-1, seq_len, 1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
X_test = test.signal.values.reshape(-1, seq_len, 1)

#Popis vnitrni struktury neuronky
inputs = L.Input(shape=(seq_len, 1))
x = L.Dense(32, activation='linear')(inputs)
#Klasifikatory
x = L.Bidirectional(L.GRU(32, return_sequences=True))(x)
x = L.Bidirectional(L.GRU(32, return_sequences=True))(x)
x = L.Dense(n_classes, activation='softmax')(x)

#Vytvatreni modelu
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer = optimizers.Adam(lr=0.01), loss='sparse_categorical_crossentropy')

#Trening modelu
model.fit(
    X_train, y_train, 
    batch_size=64,
    epochs=5,
    callbacks=[
        callbacks.ReduceLROnPlateau(),
        callbacks.ModelCheckpoint('model.h5')
    ],
    validation_data=(X_valid, y_valid)
)

#Uloz nacviceny model
model.load_weights('model.h5')

#Udelej predicku testovacich dat
valid_pred = model.predict(X_valid, batch_size=64).argmax(axis=-1)

#Kolik dat spravne predikovano
print(f1_score(y_valid.reshape(-1), valid_pred.reshape(-1), average='macro'))

#Predikce submission souboru
test_pred = model.predict(X_test, batch_size=64).argmax(axis=-1)
sub.open_channels = test_pred.reshape(-1)
sub.to_csv('submission.csv', index=False)