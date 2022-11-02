#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Initializing basic_cleaning job in Weights & Biases")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Loading {args.input_artifact} from Weights & Biases")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info(f"DataFrame information\n{df.shape}\n{df.describe()}")
    logger.info(df.info())

    # Drop duplicates if any
    logger.info("Dropping duplicates from pandas DataFrame")
    df_shapes = [df.shape]
    df.drop_duplicates(inplace=True)
    df_shapes.append(df.shape)
    logger.info(
        f"The DataFrame has shape {df_shapes[0]} before dropping duplicates and shape {df_shapes[1]} after dropping."
    )

    # Convert `last_review` str to datetime
    logger.info("Convert last_review to datetime")
    df_types = [type(df.loc[0, "last_review"])]
    df["last_review"] = pd.to_datetime(df["last_review"])
    df_types.append(type(df.loc[0, "last_review"]))
    logger.info(
        f"The last_review column has type {df_types[0]} before converting and type {df_types[1]} after converting."
    )

    # Do not imput missing values
    # df['last_review'] = pd.to_datetime(df['last_review']).fillna(pd.to_datetime(df['last_review']).min())
    # df['reviews_per_month'].fillna(0, inplace=True)

    # Drop `price` outliers
    logger.info(f"Removing price outliers. Price must be in [{args.min_price},{args.max_price}]")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Drop rows that are not in the proper geolocation
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info(f"DataFrame information\n{df.shape}\n{df.describe()}")
    logger.info(df.info())

    # Save DataFrame to csv file
    logger.info(f"Saving DataFrame to csv file {args.output_artifact}")
    df.to_csv(args.output_artifact, index=False)

    logger.info(
        f"Uploading csv file {args.output_artifact} to Weights & Biases "
        f"with type {args.output_type} and description {args.output_description}"
    )
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    logger.info("Finishing basic_cleaning job in Weights & Biases")
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to be cleaned",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Min acceptable price. Used to remove outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Max acceptable price. Used to remove outliers",
        required=True,
    )

    args = parser.parse_args()

    go(args)
