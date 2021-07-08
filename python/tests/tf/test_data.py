from pathlib import Path

from pyspark.sql import SparkSession
import numpy as np

from rikai.numpy import wrap
from rikai.types.vision import Image
import rikai.tf.data


def test_load_dataset(spark: SparkSession, tmp_path: Path):
    dataset_dir = tmp_path / "features"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True)

    expected = []
    data = []
    for i in range(1000):
        image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
        image_uri = asset_dir / f"{i}.png"

        array = wrap(np.random.random_sample((3, 4)))
        data.append(
            {
                "id": i,
                "array": array,
                "image": Image.from_array(image_data, image_uri),
            }
        )
        expected.append({"id": i, "array": array, "image": image_data})
    df = spark.createDataFrame(data)

    df.write.mode("overwrite").format("rikai").save(str(dataset_dir))

    dataset = rikai.tf.data.from_rikai(dataset_dir)
    for a in dataset.take(2):
        print(a)
