import csv
hypo = ['?','?','?','?','?']
with open('/content/enjoysport (1).csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    data = []
    print("csv file is: ")
    for line in lines:
    print(line)
    if line[len(line)-1].upper()=='YES':
    data.append(line[:-1])
print("Positive eg are: ")
for line in data:
    print(line)
print("Steps of the candidate elimination algo are: ")
print(hypo)
hypo = data[0]
for i in range(len(data)):
    for j in range(len(data[0])):
    if hypo[j]!=data[i][j]:
    hypo[j] = '?'
    print(hypo)
print("The maximally specific Find-s hypothesis for the given training 
examples is ")
print(hypo)
