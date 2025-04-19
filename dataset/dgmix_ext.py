import csv

input_file = "data.csv"
output_file = "dgmix_values.txt"

dgmix_index = 29  # 30th column, 0-based index

with open(input_file, newline='') as csvfile, open(output_file, "w") as outfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) > dgmix_index:
            val = row[dgmix_index].strip()
            if val and val != "0":
                outfile.write(val + "\n")

print(f"Extracted dGmix values written to {output_file}")