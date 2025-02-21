import requests
import re
import csv
from datetime import datetime, timedelta

response = requests.get('https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/202502/dst2502.for.request')

print(response)

data = response.text

# Process each line
parsed_data = []
for line in data.split("\n"):
    # Extract header (first non-space word)
    header_match = re.match(r"(\S+)", line)
    header = header_match.group(1) if header_match else None

    # Extract numbers (handling negative and very large numbers)
    numbers = re.findall(r"-?\d+", line)
    numbers = [int(n) for n in numbers]

    # Filter out invalid numbers (e.g., extremely large values like 999999999999...)
    filtered_numbers = [n if abs(n) < 999999999 else None for n in numbers]

    # Store results
    parsed_data.append({"header": header, "values": filtered_numbers})

outdata = []
header = []
for entry in parsed_data:
    if len(entry['values']) < 29:
        break
    header = entry['values'][:2]
    outdata = entry['values'][4:29]

# Starting timestamp

start_time = datetime(2000 + header[0] // 100, header[0] % 100, header[1], 0, 0, 0)

# Create CSV file
csv_filename = "new_train_1.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ds", "y"])  # Header

    # Generate timestamps and write data
    for i, value in enumerate(outdata):
        timestamp = start_time + timedelta(hours=i)
        writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], value])

print(f"CSV file '{csv_filename}' has been created successfully.")