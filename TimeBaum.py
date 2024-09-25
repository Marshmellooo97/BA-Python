import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Globale Variablen
X = None
Y = None
x_train = None
x_test = None
y_train = None
y_test = None
modelBaum = None

def BaumDatenvorbereitung():
    global X, Y, x_train, x_test, y_train, y_test  # Globalen Zugriff deklarieren

    # Daten laden
    pfadData = '../Testdaten/final_df2_cleaned.csv'
    data = pd.read_csv(pfadData)
    data = data.apply(pd.to_numeric, errors='coerce')

    X = data.drop('Gesamt_MEASURE_FAIL_CODE', axis=1)
    Y = data['Gesamt_MEASURE_FAIL_CODE']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

def BaumModell():
    global modelBaum  # Globalen Zugriff deklarieren
    # Entscheidungsbaum-Modell trainieren
    modelBaum = DecisionTreeClassifier(random_state=101, min_samples_leaf=40)
    
def BaumModellTrainieren():    
    modelBaum.fit(x_train, y_train)

def BaumVorhersagen():
    global modelBaum  # Globalen Zugriff deklarieren
    # Vorhersagen
    y_pred = modelBaum.predict(x_test)
    return y_pred  # Vorhersagen zurückgeben
