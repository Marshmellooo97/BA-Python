import pandas as pd
from TimeDatenvorbereitung import TimeDatenvorbereitung
from TimeBaum import BaumDatenvorbereitung, BaumModell, BaumModellTrainieren, BaumVorhersagen
from TimeMLP import MLPDatenvorbereitung, MLPModell, MLPModellTrainieren, MLPVorhersagen
from memory_profiler import memory_usage
import time

# Funktion zur Messung von CPU-Zeit, Real-Zeit und Speicherverbrauch
def measure_performance(func):
    # Zeitmessung mit time.process_time() für die CPU-Zeit
    start_cpu_time = time.process_time()
    
    # Zeitmessung mit time.time() für die reale Zeit (Wandzeit)
    start_real_time = time.time()

    # Funktion ausführen und Speicherverbrauch messen
    mem_usage = memory_usage((func,), interval=0.1, timeout=1)

    # Zeit nach der Ausführung messen
    end_cpu_time = time.process_time()
    end_real_time = time.time()

    # Gesamtlaufzeit berechnen
    total_cpu_time = end_cpu_time - start_cpu_time
    total_real_time = end_real_time - start_real_time

    return total_cpu_time, total_real_time, max(mem_usage)  # Maximalen Speicherverbrauch zurückgeben

if __name__ == "__main__":
    # Liste der Funktionen, die ausgeführt und gemessen werden sollen
    functions_to_run = [
        (TimeDatenvorbereitung, "TimeDatenvorbereitung"),
        (BaumDatenvorbereitung, "BaumDatenvorbereitung"),
        (BaumModell, "BaumModell"),
        (BaumModellTrainieren, "BaumModellTrainieren"),
        (BaumVorhersagen, "BaumVorhersagen"),
        (MLPDatenvorbereitung, "MLPDatenvorbereitung"),
        (MLPModell, "MLPModell"),
        (MLPModellTrainieren, "MLPModellTrainieren"),
        (MLPVorhersagen, "MLPVorhersagen")
    ]

    # DataFrame zur Speicherung der Ergebnisse erstellen
    results = []

    # Schleife über die Funktionen und deren Performance messen
    for func, name in functions_to_run:
        total_cpu_time, total_real_time, max_memory = measure_performance(func)  # Hier wird das Ergebnis nicht verwendet, kann aber gespeichert werden
        
        results.append({
            "Funktionsname": name,
            "CPU Zeit (Sekunden)": total_cpu_time,
            "Reale Zeit (Sekunden)": total_real_time,  # Reale Zeit hinzufügen
            "Speicherverbrauch (MB)": max_memory  # Speicherverbrauch in MB
        })

    # DataFrame aus den Ergebnissen erstellen
    results_df = pd.DataFrame(results)

    # Ausgabe des DataFrames
    print(results_df)

    # Optional: Ergebnisse in eine CSV-Datei speichern
    results_df.to_csv("performance_results.csv", index=False)
