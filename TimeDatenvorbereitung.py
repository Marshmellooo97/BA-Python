import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def TimeDatenvorbereitung():
   
    # Einlesen der Daten und Entfernen unnötiger Spalten
    pfad_data = "../Testdaten/KiData.csv"
    data = pd.read_csv(pfad_data, delimiter=";", encoding="ISO-8859-1", quotechar="\"")
    data = data.drop(columns=['UPPER_LIMIT', 'LOWER_LIMIT', 'NOMINAL', 'TOLERANCE', 'PART_NUMBER', 'STATION_NUMBER', 'WORKORDER_ID', 'STATION_DESC', 'WORKORDER_DESC'])

    # Berechnung des Gesamt_MEASURE_FAIL_CODE pro BOOKING_ID
    measure_fail_code_per_booking = data.groupby('BOOKING_ID')['MEASURE_FAIL_CODE'].apply(lambda x: 1 if (x == 1).any() else 0).reset_index()

    # Filtern der MEASURE_NAMEs, die mindestens 1000 Einträge haben
    valid_measure_names = data['MEASURE_NAME'].value_counts()[lambda x: x >= 1000].index.tolist()

    # Filterung der Daten auf valide MEASURE_NAMEs
    data_filtered = data[data['MEASURE_NAME'].isin(valid_measure_names)]

    # Korrektur der Dezimaltrennzeichen und Überprüfung der Konvertierbarkeit zu numerischen Werten
    data_filtered['MEASURE_VALUE'] = data_filtered['MEASURE_VALUE'].str.replace(",", ".")
    data_filtered['CanConvertToNumeric'] = pd.to_numeric(data_filtered['MEASURE_VALUE'], errors='coerce').notna()

    # Identifizierung nicht-numerischer MEASURE_VALUEs und deren Anzahl pro MEASURE_NAME
    invalid_measure_values = data_filtered[~data_filtered['CanConvertToNumeric']][['MEASURE_NAME', 'MEASURE_VALUE']].drop_duplicates()

    invalid_measure_values_count = invalid_measure_values.groupby('MEASURE_NAME').agg(
        Count=('MEASURE_VALUE', 'nunique'),
        Variants=('MEASURE_VALUE', lambda x: list(x.unique()))
    ).reset_index()

    # Filtern der MEASURE_NAMEs nach Anzahl der Varianten (zwischen 2 und 20)
    valid_measure_names = invalid_measure_values_count[(invalid_measure_values_count['Count'] >= 2) & (invalid_measure_values_count['Count'] <= 20)]['MEASURE_NAME']
    invalid_measure_names = invalid_measure_values_count[(invalid_measure_values_count['Count'] < 2) | (invalid_measure_values_count['Count'] > 20)]['MEASURE_NAME']

    # Filterung der Daten auf valide MEASURE_NAMEs
    data_filtered2 = data_filtered[~data_filtered['MEASURE_NAME'].isin(invalid_measure_names)]

    # Pivotieren der Daten, Hinzufügen von Zusatzinformationen und Zusammenführen
    data_wide = data_filtered2.pivot(index='BOOKING_ID', columns='MEASURE_NAME', values='MEASURE_VALUE').reset_index()

    # Ändern der Aggregation, um den ersten eindeutigen Wert anstelle einer Liste zu extrahieren
    additional_info = data_filtered2.groupby('BOOKING_ID').agg(
        STATION_ID=('STATION_ID', lambda x: x.unique()[0] if len(x.unique()) > 0 else np.nan),
        WORKORDER_NUMBER=('WORKORDER_NUMBER', lambda x: x.unique()[0] if len(x.unique()) > 0 else np.nan),
        SEQUENCE_NUMBER=('SEQUENCE_NUMBER', lambda x: x.unique()[0] if len(x.unique()) > 0 else np.nan),
        BOOK_DATE=('BOOK_DATE', lambda x: x.unique()[0] if len(x.unique()) > 0 else np.nan),
        MEASURE_TYPE=('MEASURE_TYPE', lambda x: list(x.unique()))
    ).reset_index()

    final_df = pd.merge(data_wide, measure_fail_code_per_booking, on='BOOKING_ID', how='left')
    final_df = pd.merge(final_df, additional_info, on='BOOKING_ID', how='left')
    final_df = final_df.rename(columns={'MEASURE_FAIL_CODE': 'Gesamt_MEASURE_FAIL_CODE'})

    # One-Hot-Encoding
    columns_to_encode = ['ComputerName', 'DMM_SN', 'Messergebnis DMC']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(final_df[columns_to_encode].fillna(''))
    one_hot_encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    final_df = pd.concat([final_df, one_hot_encoded_df], axis=1)
    #final_df.fillna(0, inplace=True)

    # Anpassung der DataFrame-Filterung
    final_df2 = final_df.copy()
    final_df2['D'] = final_df2['MEASURE_TYPE'].apply(lambda x: 1 if 'D' in x else 0)
    final_df2['T'] = final_df2['MEASURE_TYPE'].apply(lambda x: 1 if 'T' in x else 0)
    final_df2 = final_df2.drop(columns=['MEASURE_TYPE', 'ComputerName_', 'DMM_SN_', 'Messergebnis DMC_', 'ComputerName', 'DMM_SN', 'Messergebnis DMC'])

    final_df2['WORKORDER_NUMBER'] = final_df2['WORKORDER_NUMBER'].str.slice(0, -3).astype(float)

    final_df2 = final_df2.drop(columns=['BOOK_DATE', 'BOOKING_ID'])
    final_df2 = final_df2.apply(pd.to_numeric, errors='coerce')

    # Speichern des DataFrames als CSV
    final_df2.to_csv("/home/justin.simon/repos/BA/Testdaten/final_df2_cleaned.csv", index=False)

    # Bestätigung ausgeben
    print("DataFrame wurde in '/home/justin.simon/repos/BA/Testdaten/final_df2_cleaned.csv' gespeichert.")
