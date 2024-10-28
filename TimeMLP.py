import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Globale Variablen
X = None
Y = None
x_train = None
x_test = None
y_train = None
y_test = None
model = None
scaler = None

def MLPDatenvorbereitung():
    global X, Y, x_train, x_test, y_train, y_test, scaler  # Globalen Zugriff deklarieren

    pfadData = '../Testdaten/final_df2_cleaned.csv'
    data = pd.read_csv(pfadData)
    data.fillna(-1, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    X = data.drop('Gesamt_MEASURE_FAIL_CODE', axis=1)
    Y = data['Gesamt_MEASURE_FAIL_CODE']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

def MLPModell():
    global model  # Globalen Zugriff deklarieren
    model = Sequential()
    model.add(Dense(units=389, activation='relu', input_shape=(389,)))
    # model.add(Dropout(0.1))  # Optional, falls benötigt
    model.add(Dense(1, activation='sigmoid'))

    #optimizer = Adam(learning_rate=0.0001)  # Optional, falls benötigt
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model, x_train, y_train, x_test, y_test  # Globalen Zugriff deklarieren
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=2000, validation_data=(x_test, y_test), 
                        verbose=1, callbacks=[early_stopping])
    return history  # Rückgabe des Trainingsverlaufs (optional)

def MLPVorhersagen():
    global model, x_test  # Globalen Zugriff deklarieren
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    return y_pred  # Vorhersagen zurückgeben
