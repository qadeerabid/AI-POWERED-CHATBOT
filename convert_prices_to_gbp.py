import csv
import os
import re

# Conversion rate
INR_TO_GBP = 0.0095

# Files to process
DATA_FILES = [
    'data/data_shirts.csv',
    'data/data_sarees.csv',
    'data/data_watches.csv',
]

# Regex to extract numbers from price fields
PRICE_REGEX = re.compile(r'[\d,]+(?:\.\d+)?')

# Convert INR string (e.g., '₹1,999') to GBP string (e.g., '£19.99')
def inr_to_gbp(price_str):
    if not price_str or price_str.lower() == 'na':
        return price_str
    match = PRICE_REGEX.search(price_str.replace('₹', ''))
    if not match:
        return price_str
    inr = float(match.group(0).replace(',', ''))
    gbp = round(inr * INR_TO_GBP, 2)
    return f'£{gbp:,.2f}'

for file in DATA_FILES:
    path = os.path.join(os.path.dirname(__file__), file)
    with open(path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]
    # Find price columns
    price_cols = [i for i, col in enumerate(header) if 'Price' in col or 'MRP' in col]
    # Convert prices
    for row in rows:
        for idx in price_cols:
            row[idx] = inr_to_gbp(row[idx])
    # Write back
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

print('All prices converted to GBP and files updated.')
