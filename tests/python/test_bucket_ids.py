import knowhere
import numpy as np
import json
from bfloat16 import bfloat16
from loguru import logger
import threading
import os
import glob

num_threads = 32
num_retry = 30
dim = 128
num_train = 10_000
num_query = 10
nprobe = 12

knowhere.SetBuildThreadPool(2)
knowhere.SetSearchThreadPool(2)
version = knowhere.GetCurrentVersion()

indexFile = "test.index"

indexType = "IVF_FLAT"
config = {
    "metric_type": "L2",
    "nlist": 66,
    "nprobe": nprobe,
    "record_bucket_stats": True,
    "bucket_stats_file": "test_bucket_stats.csv",
    "return_visited_buckets": True,
}


def train_and_dump():
    train_vectors = np.random.rand(num_train, dim).astype(np.float32)
    train_data = knowhere.ArrayToDataSet(train_vectors)
    index = knowhere.CreateIndex(indexType, version)
    index.Build(train_data, json.dumps(config))
    indexBinarySet = knowhere.GetBinarySet()
    index.Serialize(indexBinarySet)
    knowhere.Dump(indexBinarySet, indexFile)


def load_index():
    index = knowhere.CreateIndex(indexType, version)
    indexBinarySet = knowhere.GetBinarySet()
    knowhere.Load(indexBinarySet, indexFile)
    index.Deserialize(indexBinarySet)
    return index


def search_test(index):
    query_vectors = np.random.rand(num_query, dim).astype(np.float32)
    query = knowhere.ArrayToDataSet(query_vectors)
    bitset = knowhere.GetNullBitSetView()
    res, _ = index.Search(query, json.dumps(config), bitset)

    ids, dists = knowhere.DataSetToArray(res)
    bucket_ids, bucket_distances = knowhere.BucketsInfoToArray(res, nprobe)

    print(f"Bucket IDs shape: {bucket_ids.shape}")
    print(f"Bucket Distances shape: {bucket_distances.shape}")

    config["nprobe"] = nprobe * 2
    res, _ = index.Search(query, json.dumps(config), bitset)
    bucket_ids_2, bucket_distances_2 = knowhere.BucketsInfoToArray(res, nprobe * 2)

    print(f"Bucket IDs 2 shape: {bucket_ids_2.shape}")
    print(f"Bucket Distances 2 shape: {bucket_distances_2.shape}")

    print(bucket_ids)
    print(bucket_ids_2)
    print(bucket_distances)
    print(bucket_distances_2)

    return ids, dists, bucket_ids, bucket_distances


def main():
    train_and_dump()
    index = load_index()
    ids, dists, bucket_ids, bucket_distances = search_test(index)


if __name__ == "__main__":
    main()
