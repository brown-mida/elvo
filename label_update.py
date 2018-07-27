import csv
from random import shuffle

def update_labels():
    positives = {}
    with open('metadata_positives.csv', 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        for row in reader:
            if row[0] != 'Anon ID':
                positives[row[0]] = 1

    negatives = {}
    with open('metadata_negatives.csv', 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        for row in reader:
            if row[0] != 'Anon ID':
                negatives[row[0]] = 0

    positives.update(negatives)
    print(len(positives))

    with open('labels.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['ID', 'Label'])
        for id_, label in list(positives.items()):
            writer.writerow([id_, label])


if __name__ == '__main__':
    update_labels()
