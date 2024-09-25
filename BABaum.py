import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden
pfadData = '../Testdaten/final_df2_cleaned.csv'
data = pd.read_csv(pfadData)
data = data.apply(pd.to_numeric, errors='coerce')

X = data.drop('Gesamt_MEASURE_FAIL_CODE', axis=1)
Y = data['Gesamt_MEASURE_FAIL_CODE']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

# Entscheidungsbaum-Modell trainieren
modelBaum = DecisionTreeClassifier(random_state=101, min_samples_leaf=40)
modelBaum.fit(x_train, y_train)

# Vorhersagen
y_pred = modelBaum.predict(x_test)

# Modell bewerten
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Konfusionsmatrix visualisieren
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Entscheidungsbaum visualisieren
plt.figure(figsize=(4, 2))
plot_tree(modelBaum, 
           filled=True,          # Fülle die Knoten mit Farbe
           feature_names=X.columns,  # Verwende die Spaltennamen für die Merkmale
           class_names=['Fail', 'No Fail'],  # Benenne die Klassen (hier anpassen, je nach deinen Daten)
           rounded=True,        # Abgerundete Ecken der Knoten
           fontsize=1)        # Schriftgröße
plt.title("Entscheidungsbaum")
plt.show()