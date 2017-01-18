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

from mpi4py import MPI

FILE        = '../data/preprocessed.tab'
EVENTS_FILE = '../data/events.tab'
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
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
UNITS       = [u'in \u00B0C', u'', u'in % air saturation']

WIDTH  = 620
HEIGHT = 300
ALPHA  = 0.5

MEASURES_PER_DAY = 96
STEP_DIFFERENCE  = 1
OUTLIERS         = 1
EPS              = 22
MIN_POINTS       = 42
DELTA            = dt.timedelta(days=2)

def read_data(path):
    data            = pd.read_csv(path, sep='\t')
    data[TIME]      = pd.to_datetime(data[TIME], format=DATE_FORMAT)
    data[FORMATTED] = data[TIME].dt.strftime(DATE_FORMAT)

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

def split(data, events, test_fold=2, folds=3):
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

        if var == 'salinity':
            # do not resize the field of view to include all the outliers in salinity, nasty exception, sorry!
            ax.set_ylim([20, 29])
        ax.plot(data[time], data[var], color=bokeh.palettes.Spectral11[index], lw=1.5)

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
        ax.plot(data[time], data[var], color=bokeh.palettes.Spectral11[index], lw=1.5)

        for c in range(-3,4):
            if c == 0: continue
            ax.plot([min_time, max_time], [c * stddev, c * stddev], color='#888888', linestyle='--', lw=0.7, dashes=[4,2])

        fig.autofmt_xdate()
        ax.autoscale_view()
        fig.savefig(var + '.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_matplot_clusters(train, test, time, variables, outliers=OUT_VARS, label=LABEL, formatted=FORMATTED, type='datetime'):
    fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel(u'$\Delta$\u00B9 outliers')
    plt.yticks(range(3), [var.split('_')[0] for var in variables], rotation='vertical', fontsize=7)

    clusters = set(train[label])
    for cluster in clusters:
        if cluster < 0: continue
        clu = train[time][train[label] == cluster]
        x = matplotlib.dates.date2num(clu.min() + (clu.max() - clu.min()) / 2.0)
        plt.axvline(x, color='#cc3300', lw=0.5)

    clusters = set(test[label])
    for cluster in clusters:
        if cluster < 0: continue
        clu = test[time][test[label] == cluster]
        x = matplotlib.dates.date2num(clu.min() + (clu.max() - clu.min()) / 2.0)
        plt.axvline(x, color='#cc3300', lw=0.5)

    for index, var in enumerate(variables):
        out = train[outliers[index]]
        x_out = np.array(train[time][out])
        y_out = np.full(len(x_out), index)
        ax.scatter(x_out, y_out, color=bokeh.palettes.Spectral11[index], s=3)

    for index, var in enumerate(variables):
        out = test[outliers[index]]
        x_out = np.array(test[time][out])
        y_out = np.full(len(x_out), index)
        ax.scatter(x_out, y_out, color=bokeh.palettes.Spectral11[index], s=3)

    fig.autofmt_xdate()
    ax.autoscale_view()
    fig.savefig('outliers.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_matplot_variables_and_events(data, events, time, variables):
    for index, var in enumerate(variables):
        fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
        ax  = fig.add_subplot(111)
        ax.set_xlabel('time')
        ax.set_ylabel(u'$\Delta$\u00B9 %s %s' % (var.split('_')[0], UNITS[index],))

        for i in range(len(events)):
            event_time = events.iloc[i][TIME]
            plt.axvline(event_time,color='#888888', linestyle='--', lw=0.7, dashes=[4,2])
        ax.plot(data[time], data[var], color=bokeh.palettes.Spectral11[index], lw=1.5)

        fig.autofmt_xdate()
        ax.autoscale_view()

        fig.savefig(var + '_events.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_matplot_predictions(train, test, train_events, test_events, time, variables):
    for index, var in enumerate(variables):
        ### train set
        fig = plt.figure(figsize=(2 * 5.0, 2 * 2.0))
        ax = fig.add_subplot(111)
        ax.set_xlabel('time')
        ax.set_ylabel(u'%s %s' % (var.split('_')[0], UNITS[index],))
        # plot train events
        for i in range(len(train_events)):
            plt.axvline(train_events.iloc[i], color='#888888', linestyle='--', lw=0.5, dashes=[4,2])

        # plot test events
        for i in range(len(test_events)):
            plt.axvline(test_events.iloc[i], color='#888888', linestyle='--', lw=0.5, dashes=[4,2])

        ax.plot(train[time], train[var], color=bokeh.palettes.Spectral11[index])
        ax.plot(test[time],  test[var], color=bokeh.palettes.Spectral11[index])
        ax.set_xlim([train[time].min(), test[time].max()])

        y_s, y_t = ax.get_ylim()[0], ax.get_ylim()[1]
        y = y_s + (y_t - y_s) * 0.05
        y_g = y_s + (y_t - y_s) * 0.95
        s_base =  40
        s = 0.24 * s_base
        s_g = 0.76 * s_base

        train_clusters = train[LABEL].max()
        for i in range(train_clusters + 1):
            cluster_index = (train[LABEL] == i)
            found = train.loc[cluster_index, time]

            found_min = found.min() - DELTA
            found_max = found.max() + DELTA
            x = matplotlib.dates.date2num(found.min() + (found.max() - found.min()) / 2.0)

            for event in train_events:
                if found_min <= event and event <= found_max:
                    plt.scatter([x], [y_g], s=s_g, color='#32cd32', alpha=1)
                    break
            else:
                x = matplotlib.dates.date2num(found.min() + (found.max() - found.min()) / 2.0)
                plt.scatter([x], [y], s=s, color='#cc3300', alpha=1)

        test_clusters = test[LABEL].max()
        for i in range(test_clusters + 1):
            cluster_index = (test[LABEL] == i)
            found = test.loc[cluster_index, time]

            found_min = found.min() - DELTA
            found_max = found.max() + DELTA
            x = matplotlib.dates.date2num(found.min() + (found.max() - found.min()) / 2.0)

            for event in test_events:
                if found_min <= event and event <= found_max:
                    plt.scatter([x], [y_g], s=s_g, color='#32cd32', alpha=1)
                    break
            else:
                x = matplotlib.dates.date2num(found.min() + (found.max() - found.min()) / 2.0)
                plt.scatter([x], [y], s=s, color='#cc3300', alpha=1)

        fig.autofmt_xdate()
        ax.autoscale_view()

        # division train/test
        plt.axvspan(test[time].min(), test[time].max(), color='y', alpha=0.1, lw=0)
        plt.figtext(0.635, 0.92, 'validation', size='small')
        plt.figtext(0.730, 0.92, 'test', size='small')

        fig.savefig(var + '_prediction.svg', format='svg', dpi=1200, bbox_inches='tight')

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

def plot_clusters(data, events, time, variables, outliers=OUTLIERS, clusters=LABEL, formatted=FORMATTED, type='datetime', **kwargs):
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

        for event in events:
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

def precision(data, events, time=TIME, clusters=LABEL, tolerance=DELTA):
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

def recall(data, events, time=TIME, clusters=LABEL, tolerance=DELTA):
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
    events_complete = read_data(EVENTS_FILE)
    events = events_complete[TIME]

    # Grid search, uncomment for full model search
    # grid  = sklearn.grid_search.ParameterGrid({
    #     'step':        range(1,  4),
    #     'eps':         range(12, 61, 2),
    #     'minPoints':   range(6,  121, 2),
    #     'filter_size': range(24, 121, 12),
    #     'outliers':    np.arange(0.25, 2.25, 0.25)
    # })
    #
    # fold  = 3
    # folds = 4
    #
    # comm = MPI.COMM_WORLD
    # rank = comm.rank
    # size = comm.size
    #
    # train_handle = open('f1_train_fold_%d_par_%d.txt' % (fold, rank,), 'w')
    # test_handle  = open('f1_test_fold_%d_par_%d.txt' % (fold, rank,), 'w')
    # total        = len(grid)
    #
    # offset = float(total) / size * rank
    # end = float(total) / size * (rank + 1)
    # raw_data = read_data(FILE)
    #
    # for index, parameters in enumerate(grid):
    #     if index < offset: continue
    #     if index > end: break
    #
    #     print '%d/%d' % (index, total,)
    #     if parameters['minPoints'] > parameters['eps'] * 2: continue
    #
    #     data = raw_data.copy(deep=True)
    #     data = filter_data(data, VARIABLES, **parameters)
    #     data = one_step_differences(data, FILTER_VARS, **parameters)
    #     train, test, train_events, test_events = split(data, events, test_fold=fold, folds=folds)
    #
    #     train           = cluster_outliers_combined(train, FS_VARS, **parameters)
    #     train_precision = precision(train, train_events)
    #     train_recall    = recall(train, train_events)
    #     train_f1        = f1(train_precision, train_recall)
    #     train_text = '%s %f %f %f\n' % (parameters, train_f1, train_precision, train_recall,)
    #     train_handle.write(train_text)
    #
    #     test           = cluster_outliers_combined(test, FS_VARS, **parameters)
    #     test_precision = precision(test, test_events)
    #     test_recall    = recall(test, test_events)
    #     test_f1        = f1(test_precision, test_recall)
    #     test_text = '%s %f %f %f\n' % (parameters, test_f1, test_precision, test_recall,)
    #     test_handle.write(test_text)
    #
    # train_handle.close()
    # test_handle.close()

    # Single model with above set parameters

    data = read_data(FILE)
    data = filter_data(data, VARIABLES)
    data = one_step_differences(data, FILTER_VARS)
    train, test, train_events, test_events = split(data, events, test_fold=3, folds=4)
    train = cluster_outliers_combined(train, FS_VARS)
    test  = cluster_outliers_combined(test, FS_VARS)

    # matplot to generate standablone SVGs
    # plot_matplot_variables(data, TIME, VARIABLES)
    # plot_matplot_variables(data, TIME, FILTER_VARS)
    # plot_matplot_standard_deviations(data, TIME, FS_VARS)
    # plot_matplot_variables_and_events(data, events_complete, TIME, FILTER_VARS)
    plot_matplot_clusters(train, test, TIME, FS_VARS)
    plot_matplot_predictions(train, test, train_events, test_events, TIME, FILTER_VARS)

    # interactive bokeh plots
    # plots = []
    # plots.append(plot_variables(data, TIME, VARIABLES))
    # plots.append(plot_variables(data, TIME, FILTER_VARS))
    # plots.append(plot_standard_deviations(data, TIME, FS_VARS))
    # plots.append(plot_clusters(data, events, TIME, FILTER_VARS))
    # bokeh.plotting.output_file(PLOT, title='Koljoefjord outlier detection')
    # bokeh.plotting.show(bokeh.plotting.gridplot(plots))
