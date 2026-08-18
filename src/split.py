"""Fixes the out-of-sample split used by every downstream step.

Chosen once, up front, and never revisited after this module is written:
every model in this project is trained on the same rows before the split
date and evaluated on the same rows on or after it.
"""

import numpy as np

SPLIT_DATE = "20100101"


def split_dataset(dataset, split_date=SPLIT_DATE):
    """Splits a dataset dict from data.build_dataset into train/test masks.

    Only rows with a full feature window (`valid`) are eligible for either
    half, so GARCH, HAR-RV, naive persistence, and the ML model are all
    scored on the identical set of out-of-sample target days.
    """
    dates = np.array(dataset["dates"])
    valid = dataset["valid"]
    train_mask = valid & (dates < split_date)
    test_mask = valid & (dates >= split_date)
    return train_mask, test_mask
