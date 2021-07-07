#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path
from typing import List, Optional, Union

import tensorflow as tf


def from_rikai(
    uri_or_df: Union[str, Path, "pyspark.sql.DataFrame"],
    columns: Optional[List[str]] = None,
) -> tf.data.Dataset:
    """Rikai Tensorflow Dataset.

    Parameters
    ----------
    data_ref : str, Path, pyspark.sql.DataFrame
        URI to the data files or the dataframe
    columns : list of str, optional
        An optional list of column to load from parquet files.

    Note
    ----

    It does not support distributed dataset *yet*.

    Example
    -------


    """

    pass
