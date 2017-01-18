import re
import pprint

SCORE_PATTERN = re.compile("""{'minPoints': (?P<minPoints>\d+), 'filter_size': (?P<filter_size>\d+), 'step': (?P<step>\d+), 'outliers': (?P<outliers>\d+\.\d+), 'eps': (?P<eps>\d+)} (?P<f1>\d+.\d+) (?P<precision>\d+.\d+) (?P<recall>\d+.\d+)""")


def read_data(path):
    with open(path, 'r') as handle:
        return handle.read()


def index_values(content):
    indexed_values = {}
    for match in SCORE_PATTERN.finditer(content):
        parameters = (int(match.group('eps')), int(match.group('minPoints')), int(match.group('filter_size')), int(match.group('step')), float(match.group('outliers')),)
        score      = (float(match.group('f1')), float(match.group('precision')), float(match.group('recall')),)
        indexed_values[parameters] = score
    return indexed_values

    
def best(n=10):
    train_iv = {}
    test_iv = {}
    for i in range(4):
            train_iv.update(index_values(read_data(train % i)))
            test_iv.update(index_values(read_data(test % i)))
    top_train = sorted(zip(train_iv.values(), train_iv.keys()), reverse=True)
    top_test = sorted(zip(test_iv.values(), test_iv.keys()), reverse=True)
    pprint.pprint(top_train[:n])
    print ''
    pprint.pprint(top_test[:n])
    best_train = top_train[0][1]
    print
    for index, ele in enumerate(top_test):
        if best_train == ele[1]:
            print index
            print ele
            break


train = 'f1_train_fold_3_par_%d.txt'
test = 'f1_test_fold_3_par_%d.txt'

