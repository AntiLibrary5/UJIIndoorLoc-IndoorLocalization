# -----------------------------------------------------------
# pre-process input feature via scaling
# -----------------------------------------------------------

def invertRSS(data):
    # Input:
    # - data: a df of input RSS features in UJIndoorLoc format (-98:min, 0:max, 100:null)
    # Output:
    # - a df of RSS values (0:mull, 1:min, 98:max)

    outOfRange = 100 #null values to be replaced
    weakestSignal = -98 #replaces the null values
    # Change null value to new value and set all lower values to it
    data.replace(outOfRange, weakestSignal, inplace=True)
    data[data < weakestSignal] = weakestSignal
    return data + 98

def scaleRSS(data):
    # Input:
    # - data: a df of input RSS features in format (0:mull, 1:min, 98:max)
    # Output:
    # - a df of scaled RSS values [0,1]

    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    return data/data.values.max()
