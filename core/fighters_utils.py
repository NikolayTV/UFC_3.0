import pandas as pd
import ast
import numpy as np


### Замена пустых значений роста на размах рук
def replace_null_height_to_arm_span(row):
    if np.isnan(row['height']) and row['armSpan']:
        arm_span = row['armSpan']
        return arm_span
    return row['height']



### Замена пустых значений размаха рук на рост
def replace_null_arm_span_to_height(row):
    if np.isnan(row['armSpan']) and row['height']:
        height = row['height']
        return height
    return row['armSpan']

