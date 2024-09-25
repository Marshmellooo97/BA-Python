import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

model = Sequential()
model.add(Dense(units=389, activation='relu', input_shape=(389,)))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

#optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), 
                    verbose=1, callbacks=[early_stopping])

print(model.summary())


pd.DataFrame(model.history.history).plot()
plt.show()

y_pred = (model.predict(x_test) > 0.5).astype("int32")  # Umwandlung der Wahrscheinlichkeiten in Klassen

# Modell bewerten
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Confusion Matrix berechnen
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Summery mlp erstellen 
# Bäume erstellen 
# Zeit messen mit Ivan reden    Gesammt ausfür dauer, Nur datenvorbereitung 1 und 2, Training , Predict, Einzelausfürung von Predict 
# Docker 