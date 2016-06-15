# -*- coding: utf-8 -*-

__author__ = 'mgoetz'

import bokeh
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import datetime as dt
import numpy as np
import pandas as pd
import scipy.signal
import sklearn.cluster
import sklearn.grid_search

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.dates
plt.style.use('ggplot')

FILE      = 'preprocessed.tab'
PLOT      = 'clustering.html'

TIME        = 'datetime'
FORMATTED   = 'formatted'
VARS        = ['temperature', 'salinity', 'oxygen']
FILTER      = 'filter'
OUTLIER     = 'outlier'
FILTER_VARS = ['%s_%s' % (var, FILTER,) for var in VARS]
STEP        = 'step'
STEP_VARS   = ['%s_%s' % (var, STEP,) for var in VARS]
FS_VARS     = ['%s_%s_%s' % (var, FILTER, STEP,) for var in VARS]
OUT_VARS    = ['%s_%s' % (var, OUTLIER) for var in VARS]
LABEL       = 'labels'
DATE_FORM   = '%Y-%m-%d %H:%M:%S'
UNITS       = [u'in \u00B0C', u'', u'in \u00B5mol/l']

WIDTH      = 620
HEIGHT     = 300
ALPHA      = 0.5

MEASURES_PER_DAY = 96
STEP_DIFFERENCE  = 1
OUTLIERS         = 1
EPS              = 28
MIN_POINTS       = 42
DELTA            = dt.timedelta(days=2)

EVENTS = pd.to_datetime([
    '15-09-2011 00:00:00',
    '01-10-2011 00:00:00',
    '07-10-2011 00:00:00',
    '18-10-2011 00:00:00',
    '10-12-2011 00:00:00',
    '25-12-2011 00:00:00',
    '20-01-2012 00:00:00',
    '26-02-2012 12:00:00',
    '15-03-2012 12:00:00',
    '21-03-2012 00:00:00',
    '06-04-2012 00:00:00',
    '24-07-2012 00:00:00',
    '04-01-2013 12:00:00',
    '14-02-2013 00:00:00',
    '19-03-2013 00:00:00',
    '23-04-2013 00:00:00'
], format='%d-%m-%Y %H:%M:%S')

def read_data(path):
    data            = pd.read_csv(path, sep='\t')
    data[TIME]      = pd.to_datetime(data[TIME], format=DATE_FORM)
    data[FORMATTED] = data[TIME].dt.strftime(DATE_FORM)

    return data

def filter_data(data, variables, filter_size=MEASURES_PER_DAY, **kwargs):
    filter = np.ones((filter_size,)) / filter_size

    for i, var in enumerate(variables):
        data[FILTER_VARS[i]] = scipy.signal.lfilter(filter, 1, data[var])

    return data[filter_size + 1:].reset_index(drop=True)

def one_step_differences(data, variables, step=STEP_DIFFERENCE, **kwargs):
    for var in variables:
        series        = data[var]
        difference    = series.shift(step) - series
        for i in range(step):
            difference[i] = 0
        data['%s_%s' % (var, STEP,)] = difference

    return data

def cluster_outliers_combined(data, variables, outliers=OUTLIERS, outlier_vars=OUT_VARS, eps=EPS, minPoints=MIN_POINTS, clusters=LABEL, **kwargs):
    outlier_points = np.full((len(data),), False, dtype=np.bool)
    data[clusters] = pd.Series([-1] * len(data))

    for index, var in enumerate(variables):
        stddev = outliers * data[var].std()
        data[outlier_vars[index]] = np.absolute(data[var]) > stddev
        outlier_points |= data[outlier_vars[index]]

    dbscan   = sklearn.cluster.DBSCAN(eps=eps, min_samples=minPoints)
    reshaped = outlier_points.nonzero()[0]
    if len(reshaped) == 0:
        return data
    dbscan.fit(reshaped.reshape([len(reshaped), -1]))

    for i in range(0, dbscan.labels_.max() + 1):
        label_index = (dbscan.labels_ == i)
        found       = reshaped[label_index]
        data.loc[found, clusters] = i

    return data

# def make_source(data, variables):
#     return bokeh.models.ColumnDataSource({var : data[var] for var in variables})
#
# def plot_variables(data, time, variables, formatted=FORMATTED, type='datetime'):
#     plots = []
#     for index, var in enumerate(variables):
#         source = make_source(data, [time, var, formatted])
#         hover  = bokeh.models.HoverTool(tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
#         figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', 'save', hover])
#
#         figure.line(time, var, source=source, color=bokeh.palettes.Spectral11[index], alpha=ALPHA)
#         figure.xaxis.axis_label = 'time'
#         figure.yaxis.axis_label = u'%s %s' % (var.split('_')[0], UNITS[index],)
#
#         plots.append(figure)
#     return plots

def plot_matplot_variables(data, time, variables, formatted=FORMATTED, type='datetime'):
    for index, var in enumerate(variables):
        fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
        ax  = fig.add_subplot(111)
        ax.set_xlabel('time')
        ax.set_ylabel(u'%s %s' % (var.split('_')[0], UNITS[index],))
        ax.plot(data[time], data[var], color=bokeh.palettes.Spectral11[index])

        fig.autofmt_xdate()
        ax.autoscale_view()
        fig.savefig(var + '.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_matplot_standard_deviations(data, time, variables, outliers=OUTLIERS, formatted=FORMATTED, type='datetime'):
    min_time = data[time].iloc[0]
    max_time = data[time].iloc[-1]

    for index, var in enumerate(variables):
        stddev = data[var].std()

        fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
        ax  = fig.add_subplot(111)
        ax.set_xlabel('time')
        ax.set_ylabel(u'$\Delta$\u00B9 %s %s' % (var.split('_')[0], UNITS[index],))
        ax.plot(data[time], data[var], color=bokeh.palettes.Spectral11[index])

        for c in range(-3,4):
            if c == 0: continue
            ax.plot([min_time, max_time], [c * stddev, c * stddev], color='#888888', linestyle='--', lw=0.7, dashes=[4,2])

        fig.autofmt_xdate()
        ax.autoscale_view()
        fig.savefig(var + '.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_matplot_clusters(data, time, variables, outliers=OUT_VARS, label=LABEL, formatted=FORMATTED, type='datetime'):
    fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel(u'$\Delta$\u00B9 outliers')
    plt.yticks(range(3), [var.split('_')[0] for var in variables], rotation='vertical', fontsize=7)

    clusters = set(data[LABEL])
    for cluster in clusters:
        if cluster < 0: continue
        clu = data[TIME][data[LABEL] == cluster]
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (matplotlib.dates.date2num(clu.min()), 0,),
                matplotlib.dates.date2num(clu.max()) - matplotlib.dates.date2num(clu.min()), 2,
                alpha=0.6, facecolor='#cc3300', linewidth=0, edgecolor='#ffffff'#colors[cluster]
            )
        )

    for index, var in enumerate(variables):
        out = data[outliers[index]]

        x_out = np.array(data[time][out])
        y_out = np.full(len(x_out), index)
        ax.scatter(x_out, y_out, color=bokeh.palettes.Spectral11[index], s=3)

    fig.autofmt_xdate()
    ax.autoscale_view()
    fig.savefig('outliers.svg', format='svg', dpi=1200, bbox_inches='tight')

#
# def plot_standard_deviations(data, time, variables, outliers=OUTLIERS, formatted=FORMATTED, type='datetime'):
#     plots = []
#
#     for index, var in enumerate(variables):
#         source = make_source(data, [time, var, formatted])
#         stddev = data[var].std()
#
#         color  = bokeh.palettes.Spectral11[index]
#         hover  = bokeh.models.HoverTool(names=['data'], tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
#         figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', hover])
#
#         figure.xaxis.axis_label = 'time'
#         figure.yaxis.axis_label = var
#         figure.line(time, var, name='data', source=source, color=color, alpha=ALPHA)
#         for c in range(-outliers, outliers + 1):
#             if c == 0: continue
#             figure.line(x=[data[time].iloc[0], data[time].iloc[-1]], y=[c * stddev, c * stddev], line_dash='4 2', color='black', alpha=ALPHA)
#
#         plots.append(figure)
#     return plots
#
# def plot_clusters(data, time, variables, outliers=OUTLIERS, clusters=LABEL, formatted=FORMATTED, type='datetime', **kwargs):
#     plots = []
#     for index, var in enumerate(variables):
#         source = make_source(data, [time, var, formatted])
#         hover  = bokeh.models.HoverTool(tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
#         figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', hover])
#         figure.xaxis.axis_label = 'time'
#         figure.yaxis.axis_label = 'clusters - %d stddevs' % outliers
#
#         figure.line(time, var, source=source, color=bokeh.palettes.Spectral11[index], alpha=ALPHA)
#         data_min = data[var].min()
#         data_max = data[var].max()
#
#         for event in EVENTS:
#             figure.line([event, event], [data_min, data_max], color='black', alpha=ALPHA, line_dash='4 2')
#
#         for i in range(0, data[clusters].max() + 1):
#             label_index = (data[clusters] == i)
#             time_scale = data[time][label_index]
#             time_min = time_scale.min()
#             time_max = time_scale.max()
#
#             figure.patch(
#                 [time_min, time_min, time_max, time_max],
#                 [data_min, data_max, data_max, data_min],
#                 color='red', alpha=ALPHA, line_width=1
#             )
#
#         plots.append(figure)
#
#     return plots
#
# def plot_fft(data, variables):
#     plots = []
#
#     import scipy.fftpack
#
#     for var in variables:
#         #fft = np.fft.rfft(data[var]).real * (1/np.sqrt(2*np.pi))
#         fft = 2.0 / len(data) * np.abs(scipy.fftpack.fft(data[var]))
#         x = np.array(range(0, len(data))) - len(data)
#         figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize'])
#         figure.xaxis.axis_label = 'frequency'
#         figure.yaxis.axis_label = 'fft'
#         figure.line(x, fft, alpha=ALPHA)
#
#         plots.append(figure)
#
#     return plots
#
# def precision(data, time=TIME, events=EVENTS, clusters=LABEL, tolerance=DELTA):
#     known = {event for event in events}
#     total_clusters = data[clusters].max()
#
#     for i in range(total_clusters + 1):
#         cluster_index = (data[clusters] == i)
#         found = data.loc[cluster_index, time]
#
#         found_min = found.min() - tolerance
#         found_max = found.max() + tolerance
#
#         for event in known:
#             if found_min <= event and event <= found_max:
#                 known.remove(event)
#                 break
#
#     return (len(events) - len(known)) / float(total_clusters) if total_clusters > 0 else 0.0
#
# def recall(data, time=TIME, events=EVENTS, clusters=LABEL, tolerance=DELTA):
#     known = {event for event in events}
#     total_clusters = data[clusters].max()
#
#     for i in range(total_clusters + 1):
#         cluster_index = (data[clusters] == i)
#         found = data.loc[cluster_index, time]
#
#         found_min = found.min() - tolerance
#         found_max = found.max() + tolerance
#
#         for event in known:
#             if found_min <= event and event <= found_max:
#                 known.remove(event)
#                 break
#
#     return (len(events) - len(known)) / float(len(events)) if len(events) > 0 else 0.0
#
# def f1(precision, recall):
#     return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

if __name__ == '__main__':
    plots = []
    # grid  = sklearn.grid_search.ParameterGrid({
    #     'step':        range(1,  6),
    #     'eps':         range(12, 97, 2),
    #     'minPoints':   range(6,  193, 2),
    #     'outliers':    range(1,  4),
    #     'filter_size': range(24, 193, 12)
    # })
    #
    # handle = open('f1_one_step.txt', 'w')
    # for parameters in grid:
    #     if parameters['minPoints'] > parameters['eps'] * 2: continue
    #
    #     data = read_data(FILE)
    #     data = filter_data(data, VARS, **parameters)
    #     data = one_step_differences(data, FILTER_VARS, **parameters)
    #     data = cluster_outliers_combined(data, FS_VARS, **par.reshape(-1)ameters)
    #
    #     prec = precision(data)
    #     reca = recall(data)
    #     f = f1(prec, reca)
    #     text = "%s %f %f %f\n" % (parameters, f, prec, reca,)
    #     print text,
    #     handle.write(text)
    # handle.close()

    data = read_data(FILE)
    data = filter_data(data, VARS)
    data = one_step_differences(data, FILTER_VARS)
    data = cluster_outliers_combined(data, FS_VARS)

    #plot_matplot_variables(data, TIME, VARS)
    #plot_matplot_variables(data, TIME, FILTER_VARS)
    #plot_matplot_standard_deviations(data, TIME, FS_VARS)
    plot_matplot_clusters(data, TIME, FS_VARS)

    # plots.append(plot_variables(data, TIME, VARS))
    # plots.append(plot_fft(data, VARS))
    # plots.append(plot_variables(data, TIME, FILTER_VARS))
    # plots.append(plot_standard_deviations(data, TIME, FS_VARS))
    # plots.append(plot_clusters(data, TIME, FILTER_VARS))

    # bokeh.plotting.output_file('foobar.html', title='Koljoefjord outlier detection')
    # bokeh.plotting.show(bokeh.plotting.gridplot(plots))
