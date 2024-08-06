import csv

hypothesis = ['%', '%', '%', '%', '%', '%']

with open("enjoysport.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = [row for row in csv_reader if row[-1].strip().upper() == "YES"]

print("\nThe given training examples are:")
for row in data:
    print(row)

print("\nThe positive examples are:")
for example in data:
    print(example)

if data:
    hypothesis = data[0][:-1]


for example in data[1:]:
    for index in range(len(hypothesis)):
        if hypothesis[index] != example[index]:
            hypothesis[index] = '?'

print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
print(hypothesis)
