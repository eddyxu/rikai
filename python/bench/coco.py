#!/usr/bin/env python
#
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

import argparse

from pyspark.sql import SparkSession
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from rikai.contrib.datasets.coco import convert
from rikai.logging import logger
from rikai.torch.data import Dataset


def coco_convert(args):
    logger.info("Converting coco dataset into Rikai format: args=%s", args)

    spark = (
        SparkSession.builder.appName("spark-test")
        .config(
            "spark.jars.packages",
            ",".join(
                [
                    "ai.eto:rikai_2.12:0.0.3-SNAPSHOT",
                    "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.1.6",
                ]
            ),
        )
        .config("spark.executor.memory", "4g")
        .config(
            "fs.AbstractFileSystem.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
        )
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
        )
        .config(
            "rikai.sql.ml.registry.file.impl",
            "ai.eto.rikai.sql.model.fs.FileSystemRegistry",
        )
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .master("local[*]")
        .getOrCreate()
    )
    df = convert(spark, args.dataset)
    df.write.format("rikai").save(args.output)


class _Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for row in self.dataset:
            # print(row["image"])
            yield self.transform(row["image"])


def collate_fn(batch):
    return batch


def train(args):
    print(args.dataset)
    dataset = Dataset(args.dataset, columns=["image"])
    # print(next(iter(dataset)))

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    transform = T.Compose([T.ToPILImage(), T.Resize((640, 640)), T.ToTensor()])
    dataset = _Dataset(dataset, transform)

    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=4,
        collate_fn=collate_fn,
    )
    steps = 0
    for batch in loader:
        steps += 1
        if steps > args.steps:
            break


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="sub commands")

    parser_convert = subparsers.add_parser("datagen", help="data generation")
    parser_convert.add_argument("dataset", help="dataset root directory")
    parser_convert.add_argument("output", help="output directory")
    parser_convert.set_defaults(func=coco_convert)

    parser_train = subparsers.add_parser("train", help="train")
    parser_train.add_argument("dataset")
    parser_train.add_argument("--steps", default=1000, type=int)
    parser_train.add_argument("-w", "--num_workers", default=8, type=int)
    parser_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
