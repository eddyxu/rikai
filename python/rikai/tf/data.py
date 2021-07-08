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
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from rikai.mixin import ToNumpy
from rikai.parquet.dataset import Dataset

__all__ = ["from_rikai"]


def _schema_to_tf_spec(schema: Dict[str, Any]) -> tf.TypeSpec:
    print("Find schema", schema)
    if schema == "long":
        return tf.TensorSpec(shape=(), dtype=tf.int64)
    elif schema["type"] == "udt":
        cls = Dataset._find_udt(schema["pyClass"])
        print("Find class: ", cls, type(cls))
        # assert isinstance(
        #     cls, ToNumpy
        # ), f"We can only load ToNumpy class, but got {cls}"
        return tf.TensorSpec(shape=(None))
    else:
        raise ValueError("")


def _get_output_signature(dataset: Dataset) -> Tuple:
    spark_schema = dataset.spark_row_metadata
    assert spark_schema["type"] == "struct"
    print(spark_schema)
    return {
        field["name"]: _schema_to_tf_spec(field["type"])
        for field in spark_schema["fields"]
    }


def from_rikai(
    data_ref: Union[str, Path, "pyspark.sql.DataFrame"],
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

    Use Rikai dataset in Tensorflow>2.0

    >>> import rikai.tf
    >>>
    >>> dataset = (
    ...    rikai.tf.data
    ...    .from_rikai("dataset", columns=["image", "label"])
    ...    .batch(32)
    ...    .prefetch(tf.data.experimental.AUTOTUNE)
    ...    .map(my_transform)
    ... )
    """
    raw_data = Dataset(data_ref, columns=columns, convert_tensor=True)

    def gen_impl():
        for row in raw_data:
            yield row

    signature = _get_output_signature(raw_data)
    print(signature)

    return tf.data.Dataset.from_generator(
        gen_impl, output_signature=_get_output_signature(raw_data)
    )
