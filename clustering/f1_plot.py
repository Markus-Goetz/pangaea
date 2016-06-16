__author__ = 'mgoetz'

import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import re
plt.style.use('ggplot')

SCORE_PATTERN = re.compile("""{'minPoints': (?P<minPoints>\d+), 'filter_size': \d+, 'step': \d+, 'outliers': \d+, 'eps': (?P<eps>\d+)} (?P<f1>\d+.\d+) \d+.\d+ \d+.\d+""")

def read_data(path):
    with open(path, 'r') as handle:
        return handle.read()

def find_best_values(content):
    best_values = {}
    for match in SCORE_PATTERN.finditer(content):
        parameter = (int(match.group('minPoints')), int(match.group('eps')),)
        score     = float(match.group('f1'))

        best_values.setdefault(parameter, 0.0)
        best_values[parameter] = max(best_values[parameter], score)

    return best_values

def f1_surface_plot(best_values):
    m = map(lambda x: x[0], best_values.keys())
    e = map(lambda y: y[0], best_values.keys())
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

    ax.view_init(elev=15, azim=150)
    #plt.show()
    plt.savefig('f1.svg', format='svg', dpi=1200, bbox_inches='tight')

if __name__ == '__main__':
    best_values = find_best_values(read_data('f1_scores.txt'))
    f1_surface_plot(best_values)
