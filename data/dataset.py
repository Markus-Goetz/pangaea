import datetime as dt
import numpy as np
import pandas as pd

# Data 2011-2013
FILE_11_13         = '2011_2013.tab'
TIME               = 'datetime'
VARIABLES          = ['temperature', 'salinity', 'oxygen']
DROP_COLUMNS_11_13 = ['timestamp', 'timestamp_one', 'salinity_one', 'temperature_one', 'oxygen_one']
DATE_FORMAT_11_13  = '%Y-%m-%d %H:%M:%S'

# Data 2013-2015
FILES_13_15 = ['RCM_270_20130425_0930_T_SAL_O2_15m.txt',
               'RCM_270_20130605_1411_T_SAL_O2_15m.txt',
               'RCM_270_20130823_1548_T_SAL_O2_15m.txt',
               'RCM_270_20131212_1634_T_SAL_O2_15m.txt',
               'RCM_270_20140305_1343_T_SAL_O2_15m.txt',
               'RCM_270_20140604_1400_T_SAL_O2_15m.txt',
               'RCM_270_20140901_1517_T_SAL_O2_15m.txt',
               'RCM_270_20150211_1525_T_SAL_O2_15m.txt',
               'RCM_270_20150407_1709_T_SAL_O2_15m.txt']
COLUMNS_13_15     = ['datetime', 'oxygen', 'temperature', 'salinity']
DATE_FORMAT_13_15 = '%Y-%m-%dT%H:%M:%SZ'

EVENTS = pd.DataFrame([
    ['15-09-2011 00:00:00', False],
    ['01-10-2011 00:00:00', False],
    ['07-10-2011 00:00:00', False],
    ['18-10-2011 00:00:00', False],
    ['10-12-2011 00:00:00', False],
    ['25-12-2011 00:00:00', False],
    ['20-01-2012 00:00:00', False],
    ['26-02-2012 12:00:00', False],
    ['15-03-2012 12:00:00', False],
    ['21-03-2012 00:00:00', False],
    ['06-04-2012 00:00:00', False],
    ['18-07-2012 02:40:00', False],
    ['24-07-2012 00:00:00', False],
    ['11-09-2012 10:20:00', False],
    ['04-01-2013 12:00:00', False],
    ['14-02-2013 00:00:00', False],
    ['19-03-2013 00:00:00', False],
    ['21-04-2013 00:20:00', False],
    ['24-04-2013 00:20:00', False],
    ['02-05-2013 13:00:00', False],
    ['07-05-2013 23:00:00', False],
    ['16-07-2013 17:40:00', False],
    ['21-08-2013 22:00:00', False],
    ['22-09-2013 01:20:00', False],
    ['24-09-2013 06:20:00', False],
    ['26-09-2013 23:00:00', False],
    ['29-09-2013 14:40:00', False],
    ['03-10-2013 10:00:00', False],
    ['07-10-2013 08:00:00', False],
    ['11-10-2013 10:20:00', False],
    ['16-10-2013 10:40:00', False],
    ['22-10-2013 02:00:00', False],
    ['06-12-2013 21:40:00', True], # begin fouling
    ['14-12-2013 08:00:00', True],
    ['16-12-2013 06:40:00', True],
    ['18-12-2013 12:00:00', True],
    ['25-12-2013 20:00:00', True],
    ['01-01-2014 01:20:00', True],
    ['18-02-2014 13:40:00', True], # end fouling
    ['06-03-2014 08:40:00', False],
    ['26-03-2014 18:00:00', False],
    ['02-04-2013 11:40:00', False],
    ['05-06-2014 10:40:00', True], # sensor redeploy
    ['25-06-2014 13:00:00', False],
    ['02-09-2014 10:40:00', True], # oxygen sensor redeploy
    ['24-09-2014 03:20:00', False],
    ['28-09-2014 23:00:00', False],
    ['08-10-2014 22:20:00', False],
    ['18-10-2014 03:40:00', False],
    ['29-10-2014 08:40:00', False],
    ['11-12-2014 18:00:00', False],
    ['17-12-2014 16:20:00', False],
    ['21-12-2014 14:00:00', False],
    ['24-12-2014 22:40:00', False],
    ['30-12-2014 10:20:00', False],
    ['02-01-2015 18:40:00', False],
    ['19-01-2015 00:00:00', False],
    ['11-03-2015 07:20:00', False],
    ['23-03-2015 19:20:00', False],
    ['28-03-2015 08:20:00', False],
    ['10-04-2015 00:00:00', False],
    ['12-04-2015 19:40:00', False],
    ['09-05-2015 15:40:00', False],
    ['14-05-2015 23:40:00', False],
    ['19-05-2015 09:00:00', False],
    ['27-05-2015 07:40:00', False],
    ['30-05-2015 08:20:00', False],
    ['03-06-2015 15:40:00', False],
    ['08-07-2015 00:20:00', False]
], columns=[TIME, 'false_reading'])
EVENTS[TIME] = pd.to_datetime(EVENTS[TIME], format='%d-%m-%Y %H:%M:%S')
EVENTS_FILE  = 'events.tab'

TIME_DELTA = 30 #min

# Result
RESULT_FILE = 'preprocessed.tab'

def read_data(path):
    data = pd.read_csv(path, sep='\t')
    data[TIME] = pd.to_datetime(data[TIME], format=DATE_FORMAT_11_13)

    return data.drop(DROP_COLUMNS_11_13, axis=1)

def read_test(paths):
    test_data = []

    for path in paths:
        data = pd.read_csv(path, sep='\t', skiprows=2, names=COLUMNS_13_15)
        data[TIME] = pd.to_datetime(data[TIME], format=DATE_FORMAT_13_15)
        test_data.append(data)

    return pd.concat(test_data).reset_index(drop=True)[[TIME] + VARIABLES]

def fill_gaps(data):
    delta = dt.timedelta(minutes=TIME_DELTA)
    differences = (data[TIME] - data[TIME].shift(1))[1:]

    gaps_index = (differences > delta).nonzero()[0]
    offset = 0

    print 'gaps:'
    for gap in gaps_index:
        before  = data.loc[gap + offset]
        after   = data.loc[gap + offset + 1]
        n_elems = int((after[TIME] - before[TIME]) / delta)
        print before[TIME], after[TIME], n_elems

        fill_points = [before[TIME] + (i + 1) * delta for i in range(n_elems)]
        fill_points_unix = [time.value for time in fill_points]
        fill_frame = pd.DataFrame({TIME: fill_points})

        for var in VARIABLES:
            interpolated = np.interp(fill_points_unix, [before[TIME].value, after[TIME].value], [before[var], after[var]])
            fill_frame[var] = interpolated

        data = pd.concat((data[:gap + offset + 1], fill_frame, data[gap + offset + 1:],)).reset_index(drop=True)
        offset += n_elems

    return data

if __name__ == '__main__':
    data_11_13 = read_data(FILE_11_13)
    data_13_15 = read_test(FILES_13_15)
    data = pd.concat([data_11_13, data_13_15]).reset_index(drop=True)
    data = fill_gaps(data)
    data.to_csv(RESULT_FILE, sep='\t', index=False)
    EVENTS.to_csv(EVENTS_FILE, sep='\t', index=False)
