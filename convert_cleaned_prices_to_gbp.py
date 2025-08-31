import csv
import os
import re

INR_TO_GBP = 0.0095
CLEANED_FILE = 'artifacts/data_cleaned.csv'

PRICE_REGEX = re.compile(r'[\d,]+(?:\.\d+)?')

def inr_to_gbp(price_str):
    if not price_str or price_str.lower() == 'na':
        return price_str
    match = PRICE_REGEX.search(price_str.replace('₹', ''))
    if not match:
        return price_str
    inr = float(match.group(0).replace(',', ''))
    gbp = round(inr * INR_TO_GBP, 2)
    return f'£{gbp:,.2f}'

path = os.path.join(os.path.dirname(__file__), CLEANED_FILE)
with open(path, 'r', encoding='utf-8') as f:
    reader = list(csv.reader(f))
header = reader[0]
rows = reader[1:]

# Find price columns
price_cols = [i for i, col in enumerate(header) if 'Price' in col or 'MRP' in col]

for row in rows:
    for idx in price_cols:
        row[idx] = inr_to_gbp(row[idx])

with open(path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print('Prices in artifacts/data_cleaned.csv converted to GBP.')
