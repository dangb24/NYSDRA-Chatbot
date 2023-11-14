# import pandas as pd
# df = pd.read_csv("policy_test_data.csv")
# questions = df['Query']
# responses = df.Segment



import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

# with open('policy_test_data.csv') as f:
#     reader = csv.DictReader(f) # read rows into a dictionary format
#     for row in reader: # read a row as {column1: value1, column2: value2,...}
#         for (k,v) in row.items(): # go over each column name and value 
#             columns[k].append(v) # append the value into the appropriate list
#                                  # based on column name k

# print(columns['Query'])

with open('policy_test_data.csv') as f:
    reader = csv.reader(f, delimiter = "\t")
    next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
            
            
            
with open('processed_csv/csv__test_processed.txt', 'w') as f:
    for row, row2 in zip(columns[5], columns[6]):
        f.write(row + " | " + row2 + '\n')
# for col in columns:
#     print(columns[col])
#     print()
#     print()



#col 5 is the questions
#col 6 is the answer
print(columns[5][1])
print(columns[6][1])
print()
print()



#and then do it again but for the train file because that one is a little different
with open('policy_train_data.csv') as f:
    reader = csv.reader(f, delimiter = "\t")
    next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
            
            
            
with open('processed_csv/csv_train_processed.txt', 'w') as f:
    for row, row2 in zip(columns[5], columns[6]):
        f.write(row + " | " + row2 + '\n')
# for col in columns:
#     print(columns[col])
#     print()
#     print()



#col 5 is the questions
#col 6 is the answer
print(columns[5][1])
print(columns[6][1])
print()
print()


