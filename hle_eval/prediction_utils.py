#!/usr/bin/env python3

"""Utilities to analyze prediction results file.
"""

import json
import numpy as np
import argparse


def calculate_token_usage(predictions):
    # calculate distribution of token use.
    buckets = [25, 50, 75, 90, 95]
    tokens = [p["usage"]["total_tokens"] for p in predictions.values()]
    percentiles = np.percentile(tokens, buckets)
    print("Total tokens distribution")
    for perc, bucket in zip(percentiles, buckets):
        print(f"  {bucket}%: {perc:10.2f}")
    print(f"  avg: {np.average(tokens):10.2f}")


def main(args):
    with open(args.prediction_result, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    calculate_token_usage(predictions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prediction_result", "-p", required=True, help="prediction result JSON file")
    args = p.parse_args()
    main(args)
