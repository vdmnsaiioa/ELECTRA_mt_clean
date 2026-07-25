import json
import csv

# Input and output file paths
input_json = "/home/energy/s234633/electra_MM/data_splits/qm9_g4mp2_energies_atomization.json"
output_csv = "/home/energy/s234633/ELECTRA/data_splits/energies.csv"

# Load JSON data
with open(input_json, 'r') as f:
    data = json.load(f)

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'value'])  # header

    for key, value in data.items():
        writer.writerow([key, value])

print(f"CSV file '{output_csv}' created successfully.")
