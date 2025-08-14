import csv
import os
import psutil

def log_experiment_result(csv_path, result_dict):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)

def get_system_metrics():
    return {
        "cpu_usage": round(psutil.cpu_percent(), 2),
        "memory_used": round(psutil.virtual_memory().used / (1024 * 1024), 2),
        "cpu_cores": psutil.cpu_count(logical=True),
    }
