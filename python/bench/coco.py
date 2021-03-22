#!/usr/bin/env python
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

from rikai.contrib.datasets.coco import convert
from rikai.logging import logger


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


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="sub commands")
    parser_convert = subparsers.add_parser("convert", help="data generation")
    parser_convert.add_argument("dataset", help="dataset root directory")
    parser_convert.add_argument("output", help="output directory")
    parser_convert.set_defaults(func=coco_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
