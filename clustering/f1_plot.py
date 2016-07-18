__author__ = 'mgoetz'

import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import re
plt.style.use('ggplot')

SCORE_PATTERN = re.compile("""{'minPoints': (?P<minPoints>\d+), 'filter_size': (?P<filter_size>\d+), 'step': (?P<step>\d+), 'outliers': (?P<outliers>\d+), 'eps': (?P<eps>\d+)} (?P<f1>\d+.\d+) (?P<precision>\d+.\d+) (?P<recall>\d+.\d+)""")

def read_data(path):
    with open(path, 'r') as handle:
        return handle.read()

def index_values(content):
    indexed_values = {}

    for match in SCORE_PATTERN.finditer(content):
        parameters = (int(match.group('eps')), int(match.group('minPoints')), int(match.group('filter_size')), int(match.group('step')), int(match.group('outliers')),)
        score      = (float(match.group('f1')), float(match.group('precision')), float(match.group('recall')),)
        indexed_values[parameters] = score

    return indexed_values

def flatten_eps_min(indexed_values):
    flattened = {}

    for key, score in indexed_values.items():
        parameters = (key[0], key[1],)
        flattened.setdefault(parameters, 0.0)
        flattened[parameters] = max(flattened[parameters], score[0])

    return flattened

def f1_surface_plot(best_values):
    m = map(lambda x: x[0], best_values.keys())
    e = map(lambda y: y[1], best_values.keys())
    m_min = min(m)
    m_max = max(m)
    e_min = min(e)
    e_max = max(e)

    m_coords, e_coords = np.meshgrid(range(m_min, m_max + 1, 2), range(e_min, e_max + 1, 2))
    f = np.array([best_values.get((x,y), 0) for x, y in zip(np.ravel(m_coords), np.ravel(e_coords))]).reshape(m_coords.shape)

    fig = plt.figure()
    ax  = fig.gca(projection='3d', axisbg='white')
    ax.set_xlabel('minPoints')
    ax.set_ylabel('$\epsilon$')
    ax.set_zlabel('F1 score')
    ax.plot_surface(m_coords, e_coords, f, rstride=2, cstride=2, cmap=matplotlib.cm.coolwarm, antialiased=True, alpha=0.7)

    ax.view_init(elev=15, azim=-120)

if __name__ == '__main__':
    # best_values_0 = index_values(read_data('f1_train_fold_0.txt'))
    # best_values_1 = index_values(read_data('f1_train_fold_1.txt'))
    # best_values_2 = index_values(read_data('f1_train_fold_2.txt'))
    #
    # averaged = {}
    # for parameter in best_values_2:
    #     averaged[parameter] = best_values_2[parameter]###np.mean([best_values_0[parameter], best_values_1[parameter], best_values_2[parameter]])
    #
    # top = sorted(zip(averaged.values(), averaged.keys()), reverse=True)[:10]
    # import pprint; pprint.pprint(top)

    # best_values_0 = index_values(read_data('f1_test_fold_0.txt'))
    # best_values_1 = index_values(read_data('f1_test_fold_1.txt'))
    best_values_2 = index_values(read_data('f1_test_fold_2.txt'))

    averaged = {}
    for parameter in best_values_2:
        averaged[parameter] = best_values_2[parameter]#np.mean([best_values_0[parameter], best_values_1[parameter], best_values_2[parameter]])

    top = sorted(zip(averaged.values(), averaged.keys()), reverse=True)[:10]
    import pprint; pprint.pprint(top)

    # best_values = flatten_eps_min(best_values_2)
    # f1_surface_plot(best_values)
    # plt.savefig('f1_train.svg', format='svg', dpi=1200, bbox_inches='tight')
    #
    # best_values_2 = index_values(read_data('f1_test_fold_2.txt'))
    # best_values = flatten_eps_min(best_values_2)
    # f1_surface_plot(best_values)
    # plt.savefig('f1_test.svg', format='svg', dpi=1200, bbox_inches='tight')
    #
    # plt.show()
