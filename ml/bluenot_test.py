import numpy as np
import pandas as pd

import bluenot


def test_to_arrays():
    x_dict = {
        'a': np.array([1]),
        'b': np.array([2]),
        'c': np.array([3]),
    }
    y_df = pd.DataFrame(
        data=np.array([1, 2, 3]),
        index=['a', 'b', 'c'],
    )

    x_arr, y_arr = bluenot.to_arrays(x_dict, y_df)
    assert np.all(x_arr == y_arr)
