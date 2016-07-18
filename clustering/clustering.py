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

FILE        = '../data/preprocessed.tab'
PLOT        = 'clustering.html'
TIME        = 'datetime'
FORMATTED   = 'formatted'
VARIABLES   = ['temperature', 'salinity', 'oxygen']
FILTER      = 'filter'
OUTLIER     = 'outlier'
FILTER_VARS = ['%s_%s' % (var, FILTER,) for var in VARIABLES]
STEP        = 'step'
STEP_VARS   = ['%s_%s' % (var, STEP,) for var in VARIABLES]
FS_VARS     = ['%s_%s_%s' % (var, FILTER, STEP,) for var in VARIABLES]
OUT_VARS    = ['%s_%s' % (var, OUTLIER) for var in VARIABLES]
LABEL       = 'labels'
DATE_FORM   = '%Y-%m-%d %H:%M:%S'
UNITS       = [u'in \u00B0C', u'', u'in % air saturation']

WIDTH      = 620
HEIGHT     = 300
ALPHA      = 0.5

MEASURES_PER_DAY = 96
STEP_DIFFERENCE  = 2
OUTLIERS         = 1
EPS              = 20
MIN_POINTS       = 38
DELTA            = dt.timedelta(days=1)

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
    '18-07-2012 02:40:00',
    '24-07-2012 00:00:00',
    '11-09-2012 10:20:00',
    '04-01-2013 12:00:00',
    '14-02-2013 00:00:00',
    '19-03-2013 00:00:00',
    '21-04-2013 00:20:00',
    '24-04-2013 00:20:00',
    '02-05-2013 13:00:00',
    '07-05-2013 23:00:00',
    '16-07-2013 17:40:00',
    '21-08-2013 22:00:00',
    '22-09-2013 01:20:00',
    '24-09-2013 06:20:00',
    '26-09-2013 23:00:00',
    '29-09-2013 14:40:00',
    '03-10-2013 10:00:00',
    '07-10-2013 08:00:00',
    '11-10-2013 10:20:00',
    '16-10-2013 10:40:00',
    '22-10-2013 02:00:00',
    '06-12-2013 21:40:00', # begin fouling
    '14-12-2013 08:00:00',
    '16-12-2013 06:40:00',
    '18-12-2013 12:00:00',
    '25-12-2013 20:00:00',
    '01-01-2014 01:20:00',
    '18-02-2014 13:40:00', # end fouling
    '06-03-2014 08:40:00',
    '26-03-2014 18:00:00',
    '02-04-2013 11:40:00',
    '05-06-2014 10:40:00', # question
    '25-06-2014 13:00:00',
    '02-09-2014 10:40:00', # oxygen sensor redeploy
    '24-09-2014 03:20:00',
    '28-09-2014 23:00:00',
    '08-10-2014 22:20:00',
    '18-10-2014 03:40:00',
    '29-10-2014 08:40:00',
    '11-12-2014 18:00:00',
    '17-12-2014 16:20:00',
    '21-12-2014 14:00:00',
    '24-12-2014 22:40:00',
    '30-12-2014 10:20:00',
    '02-01-2015 18:40:00',
    '19-01-2015 00:00:00',
    '11-03-2015 07:20:00',
    '23-03-2015 19:20:00',
    '28-03-2015 08:20:00',
    '10-04-2015 00:00:00',
    '12-04-2015 19:40:00',
    '09-05-2015 15:40:00',
    '14-05-2015 23:40:00',
    '19-05-2015 09:00:00',
    '27-05-2015 07:40:00',
    '30-05-2015 08:20:00',
    '03-06-2015 15:40:00',
    '08-07-2015 00:20:00'
], format='%d-%m-%Y %H:%M:%S')

def read_data(path):
    data            = pd.read_csv(path, sep='\t')
    data[TIME]      = pd.to_datetime(data[TIME], format=DATE_FORM)
    data[FORMATTED] = data[TIME].dt.strftime(DATE_FORM)

    return data

def filter_data(data, variables, filter_size=MEASURES_PER_DAY, **kwargs):
    filter = np.ones((filter_size,)) / filter_size

    for i, var in enumerate(variables):
        data[FILTER_VARS[i]] = scipy.signal.lfilter(filter, 1, scipy.signal.medfilt(data[var]))

    return data[filter_size + 1:].reset_index(drop=True)

def one_step_differences(data, variables, step=STEP_DIFFERENCE, **kwargs):
    for var in variables:
        series     = data[var]
        difference = series.shift(step) - series

        for i in range(step):
            difference[i] = 0
        data['%s_%s' % (var, STEP,)] = difference

    return data

def split(data, test_fold=2, folds=3, events=EVENTS):
    items = len(data)
    fraction = 1.0 / folds
    start, end = int(max(0, test_fold * fraction) * items), int(min((test_fold + 1) * fraction, 1.0) * items)

    test_index = np.full((len(data),), False, dtype=np.bool)
    test_index[start:end] = True
    train, test = data[~test_index].reset_index(drop=True), data[test_index].reset_index(drop=True)
    test_event_index = (test[TIME].min() <= events) & (events <= test[TIME].max())

    return train, test, events[~test_event_index], events[test_event_index]

def cluster_outliers_combined(data, variables, outliers=OUTLIERS, outlier_vars=OUT_VARS, eps=EPS, minPoints=MIN_POINTS, clusters=LABEL, **kwargs):
    outlier_points = np.full((len(data),), False, dtype=np.bool)
    data[clusters] = np.full((len(data),), -1, dtype=np.int)

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
                alpha=0.6, facecolor='#cc3300', linewidth=0, edgecolor='#ffffff'
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

def make_source(data, variables):
    return bokeh.models.ColumnDataSource({var : data[var] for var in variables})

def plot_variables(data, time, variables, formatted=FORMATTED, type='datetime'):
    plots = []
    for index, var in enumerate(variables):
        source = make_source(data, [time, var, formatted])
        hover  = bokeh.models.HoverTool(tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
        figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', 'save', hover])

        figure.line(time, var, source=source, color=bokeh.palettes.Spectral11[index], alpha=ALPHA)
        figure.xaxis.axis_label = 'time'
        figure.yaxis.axis_label = u'%s %s' % (var.split('_')[0], UNITS[index],)

        plots.append(figure)
    return plots

def plot_standard_deviations(data, time, variables, outliers=OUTLIERS, formatted=FORMATTED, type='datetime'):
    plots = []

    for index, var in enumerate(variables):
        source = make_source(data, [time, var, formatted])
        stddev = data[var].std()

        color  = bokeh.palettes.Spectral11[index]
        hover  = bokeh.models.HoverTool(names=['data'], tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
        figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', hover])

        figure.xaxis.axis_label = 'time'
        figure.yaxis.axis_label = var
        figure.line(time, var, name='data', source=source, color=color, alpha=ALPHA)
        for c in range(-outliers, outliers + 1):
            if c == 0: continue
            figure.line(x=[data[time].iloc[0], data[time].iloc[-1]], y=[c * stddev, c * stddev], line_dash='4 2', color='black', alpha=ALPHA)

        plots.append(figure)
    return plots

def plot_clusters(data, time, variables, outliers=OUTLIERS, clusters=LABEL, formatted=FORMATTED, type='datetime', **kwargs):
    plots = []
    for index, var in enumerate(variables):
        source = make_source(data, [time, var, formatted])
        hover  = bokeh.models.HoverTool(tooltips =[('index', '$index'), ('value', '@formatted -> $y')])
        figure = bokeh.plotting.figure(width=WIDTH, height=HEIGHT, x_axis_type=type, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize', hover])
        figure.xaxis.axis_label = 'time'
        figure.yaxis.axis_label = 'clusters - %d stddevs' % outliers

        figure.line(time, var, source=source, color=bokeh.palettes.Spectral11[index], alpha=ALPHA)
        data_min = data[var].min()
        data_max = data[var].max()

        for event in EVENTS:
            figure.line([event, event], [data_min, data_max], color='black', alpha=ALPHA, line_dash='4 2')

        for i in range(0, data[clusters].max() + 1):
            label_index = (data[clusters] == i)
            time_scale = data[time][label_index]
            time_min = time_scale.min()
            time_max = time_scale.max()

            figure.patch(
                [time_min, time_min, time_max, time_max],
                [data_min, data_max, data_max, data_min],
                color='red', alpha=ALPHA, line_width=1
            )

        plots.append(figure)

    return plots

def precision(data, events=EVENTS, time=TIME, clusters=LABEL, tolerance=DELTA):
    total_clusters = data[clusters].max()

    true_positives = 0
    false_positives = 0

    for i in range(total_clusters + 1):
        cluster_index = (data[clusters] == i)
        found = data.loc[cluster_index, time]

        found_min = found.min() - tolerance
        found_max = found.max() + tolerance

        for event in events:
            if found_min <= event and event <= found_max:
                true_positives += 1
                break
        else:
            false_positives += 1

    return true_positives / float(true_positives + false_positives) if true_positives + false_positives > 0 else 0.0

def recall(data, events=EVENTS, time=TIME, clusters=LABEL, tolerance=DELTA):
    found_events = set()
    total_clusters = data[clusters].max()
    true_positives = 0

    for i in range(total_clusters + 1):
        cluster_index = (data[clusters] == i)
        found = data.loc[cluster_index, time]

        found_min = found.min() - tolerance
        found_max = found.max() + tolerance

        for event in events:
            if found_min <= event and event <= found_max:
                found_events.add(event)
                true_positives += 1
                break

    false_negatives = len(set(events) - found_events)

    return true_positives / float(true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0

def f1(precision, recall):
    return (2 * precision * recall) / float(precision + recall) if precision + recall > 0 else 0.0

if __name__ == '__main__':
    # Grid search, uncomment for full model search

    # grid  = sklearn.grid_search.ParameterGrid({
    #     'step':        range(1,  4),
    #     'eps':         range(12, 49, 2),
    #     'minPoints':   range(6,  97, 2),
    #     'outliers':    range(1,  4),
    #     'filter_size': range(24, 97, 12)
    # })
    #
    # fold  = 1
    # folds = 3
    #
    # train_handle = open('f1_train_fold_%d.txt' % fold, 'w')
    # test_handle  = open('f1_test_fold_%d.txt' % fold, 'w')
    # total        = len(grid)
    #
    # for index, parameters in enumerate(grid):
    #     print '%d/%d' % (index, total,)
    #     if parameters['minPoints'] > parameters['eps'] * 2: continue
    #
    #     data = read_data(FILE)
    #     data = filter_data(data, VARS, **parameters)
    #     data = one_step_differences(data, FILTER_VARS, **parameters)
    #     train, test, train_events, test_events = split(data, test_fold=fold, folds=folds)
    #
    #     train           = cluster_outliers_combined(train, FS_VARS, **parameters)
    #     train_precision = precision(train, train_events)
    #     train_recall    = recall(train, train_events)
    #     train_f1        = f1(train_precision, train_recall)
    #     train_text = '%s %f %f %f\n' % (parameters, train_f1, train_precision, train_recall,)
    #     print 'train', train_text,
    #     train_handle.write(train_text)
    #
    #     test           = cluster_outliers_combined(test, FS_VARS, **parameters)
    #     test_precision = precision(test, test_events)
    #     test_recall    = recall(test, test_events)
    #     test_f1        = f1(test_precision, test_recall)
    #     test_text = '%s %f %f %f\n' % (parameters, test_f1, test_precision, test_recall,)
    #     print 'test', test_text,
    #     test_handle.write(test_text)
    #
    # train_handle.close()
    # test_handle.close()

    data = read_data(FILE)
    data = filter_data(data, VARIABLES)
    data = one_step_differences(data, FILTER_VARS)
    data = cluster_outliers_combined(data, FS_VARS)

    # matplot to generate standablone SVGs

    # plot_matplot_variables(data, TIME, VARS)
    # plot_matplot_variables(data, TIME, FILTER_VARS)
    # plot_matplot_standard_deviations(data, TIME, FS_VARS)
    # plot_matplot_clusters(data, TIME, FS_VARS)

    # interactive bokeh plots

    plots = []
    plots.append(plot_variables(data, TIME, VARIABLES))
    plots.append(plot_variables(data, TIME, FILTER_VARS))
    plots.append(plot_standard_deviations(data, TIME, FS_VARS))
    plots.append(plot_clusters(data, TIME, FILTER_VARS))
    bokeh.plotting.output_file(PLOT, title='Koljoefjord outlier detection')
    bokeh.plotting.show(bokeh.plotting.gridplot(plots))
