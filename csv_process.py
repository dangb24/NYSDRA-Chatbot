
import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('misc_data/policy_test_data.csv') as f:
    reader = csv.reader(f, delimiter = "\t")
    next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
            
            
            
with open('processed_csv/csv__test_processed.txt', 'w') as f:
    for row, row2 in zip(columns[5], columns[6]):
        f.write(row + " | " + row2 + '\n')


#and then do it again but for the train file because that one is a little different
with open('misc_data/policy_train_data.csv') as f:
    reader = csv.reader(f, delimiter = "\t")
    next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
            
            
            
with open('processed_csv/csv_train_processed.txt', 'w') as f:
    for row, row2 in zip(columns[5], columns[6]):
        f.write(row + " | " + row2 + '\n')



