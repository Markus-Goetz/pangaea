import datetime as dt
import numpy as np
import pandas as pd

FILE = 'combined.tab'
VARS = ['temperature', 'salinity', 'oxygen']

TIME        = 'datetime'
DATE_FORM   = '%Y-%m-%d %H:%M:%S'

DROP_COLUMNS = ['timestamp', 'timestamp_one', 'salinity_one', 'temperature_one', 'oxygen_one']

def read_data(path):
    data = pd.read_csv(path, sep='\t')
    data[TIME] = pd.to_datetime(data[TIME], format=DATE_FORM)

    return data.drop(DROP_COLUMNS, axis=1)

def fill_gaps(data):
    delta = dt.timedelta(minutes=30)
    differences = (data[TIME] - data[TIME].shift(1))[1:]

    gaps_index = (differences > delta).nonzero()[0]
    offset = 0

    for gap in gaps_index:
        before  = data.loc[gap + offset]
        after   = data.loc[gap + offset + 1]
        n_elems = int((after[TIME] - before[TIME]) / delta)

        fill_points = [before[TIME] + (i + 1) * delta for i in range(n_elems)]
        fill_points_unix = [time.value for time in fill_points]
        fill_frame = pd.DataFrame({TIME: fill_points})

        for var in VARS:
            interpolated = np.interp(fill_points_unix, [before[TIME].value, after[TIME].value], [before[var], after[var]])
            fill_frame[var] = interpolated

        data = pd.concat((data[:gap + offset + 1], fill_frame, data[gap + offset + 1:],)).reset_index(drop=True)
        offset += n_elems

    return data

if __name__ == '__main__':
    data = read_data(FILE)
    data = fill_gaps(data)
    # data.to_csv('preprocessed.tab', sep='\t', index=False)
