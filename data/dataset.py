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
