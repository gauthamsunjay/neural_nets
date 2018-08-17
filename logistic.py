import sys
from scipy.io import arff

from helper import *
from nn import NN

if len(sys.argv) < 5:
    print "Usage: python logistic.py l e <train_file> <test_file>"

lr, epochs, train_file, test_file = sys.argv[1:]
data, meta = arff.loadarff(train_file)

stats = get_stats(data, meta)
records = []
for record in data:
    records.append(normalize(stats, record, meta))

num_inputs = records[0][0].shape[-1]
num_hidden = 0

nn = NN(num_inputs, num_hidden, 1, float(lr), int(epochs))
nn.train(records)

test_data, test_meta = arff.loadarff(test_file)
f1, correct = calculate_f1(nn, stats, test_data, test_meta)

print "%d\t%d" % (correct, len(test_data) - correct)
print "%.12f" % f1
