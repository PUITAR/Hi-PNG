dataset = 'sift-128-euclidean'
import sys, os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_collect.binary import fvecs_read, fvecs_save
from data_collect.interval import create_interval
import numpy as np
from os import system

# Original data loading
base_data = fvecs_read(f'../data/{dataset}.train.fvecs')
query_data = fvecs_read(f"../data/{dataset}.test.fvecs")

distribution1 = "uniform-0.0-1000.0"
distribution2 = "uniform-0.0-100.0"

# Create downsampled versions
for i in range(5):
    print(f"Process {i}")
    # Calculate the sample size (1/(2^i) of original)
    sample_size = int(len(base_data) / (2 ** i))

    # Randomly sample without replacement
    base_data_i = base_data[np.random.choice(len(base_data), size=sample_size, replace=False)]

    # Save the sampled data (you can choose to save it or just keep it in memory)
    # For example, saving back to fvecs format:
    fvecs_save(f'../data/{dataset}.{i}.train.fvecs', base_data_i)
    fvecs_save(f"../data/{dataset}.{i}.test.fvecs", query_data)
    
    fvecs_save(f'../data/{dataset}.{i}.{distribution1}.train.itv', create_interval(fvecs_read(f'../data/{dataset}.{i}.train.fvecs').shape[0], "uniform", left=0, right=1000, rf=True))
    fvecs_save(f'../data/{dataset}.{i}.{distribution2}.test.itv', create_interval(fvecs_read(f'../data/{dataset}.{i}.test.fvecs').shape[0], "uniform", left=0, right=100, rf=True))

    system(f"./../bin/groundtruth \
              ../data/{dataset}.{i}.train.fvecs \
              ../data/{dataset}.{i}.test.fvecs \
              ../data/{dataset}.{i}.{distribution1}.train.itv \
              ../data/{dataset}.{i}.{distribution2}.test.itv \
              ../data/{dataset}.{i}.{distribution1}.{distribution2}.gt \
              100 48")

topk = 10
num_threads_build = 48
num_threads_search = 1
perf_output = "perf/scale"
index_type = 'hnsw'

index_param_1, index_param_2 = [32, 128], [10000, 32, 128]
search_params_1, search_params_2 = [[ef, iter] for ef in range(10, 200, 20) for iter in range(1, 10, 2)], \
    [[ef, iter, sim/100] for ef in range(10, 30, 4) for iter in range(1, 10, 1) for sim in range(0, 101, 20)]

from anns import DataSet, IntervalSet, GroundTruth, Timer, HiPNG, PostFilterGraph
import json

timer = Timer()

if not os.path.exists(perf_output):
    os.makedirs(perf_output)

for i in range(5):
    print(f"Benchmark for {i}")

    base_data = DataSet(f'../data/{dataset}.{i}.train.fvecs')
    query_data = DataSet(f'../data/{dataset}.{i}.test.fvecs')
    base_attrs = IntervalSet(f'../data/{dataset}.{i}.{distribution1}.train.itv')
    query_attrs = IntervalSet(f'../data/{dataset}.{i}.{distribution2}.test.itv')
    groundtruth = GroundTruth(f"../data/{dataset}.{i}.{distribution1}.{distribution2}.gt")
    
    # Benchmark for PostFilterGraph
    model = PostFilterGraph(index_type, index_param_1, "euclidean")
    timer.reset()
    timer.start()
    model.build(base_data, base_attrs, num_threads_build)
    timer.stop()
    build_time_sec = timer.get()
    model.get_comparison_and_clear()
    performance = []
    for search_param in tqdm(search_params_1):
        timer.reset()
        timer.start()
        _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads_search)
        timer.stop()
        search_time_sec = timer.get()
        comparison = model.get_comparison_and_clear()
        performance.append({
            "name": "HNSW",
            "index_param": index_param_1,
            "index_size": model.index_size(),
            "build_time": build_time_sec,
            "k": topk,
            "search_param": search_param,
            "search_time": search_time_sec,
            "qps": query_data.size() / search_time_sec,
            "comparison": comparison,
            "cps": comparison / search_time_sec,
            "recall": groundtruth.recall(topk, knn)})
        # Save results
        json.dump(performance, open(os.path.join(perf_output, f"{dataset}.{i}.HNSW.json"), 'w') ,indent=2)
        
    # Benchmark for Hi-PNG
    model = HiPNG(index_type, index_param_2, "euclidean")
    timer.reset()
    timer.start()
    model.build(base_data, base_attrs, num_threads_build)
    timer.stop()
    build_time_sec = timer.get()
    model.get_comparison_and_clear()
    performance = []
    for search_param in tqdm(search_params_2):
        timer.reset()
        timer.start()
        _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads_search)
        timer.stop()
        search_time_sec = timer.get()
        comparison = model.get_comparison_and_clear()
        performance.append({
            "name": "Hi-PNG-HNSW",
            "index_param": index_param_2,
            "index_size": model.index_size(),
            "build_time": build_time_sec,
            "k": topk,
            "search_param": search_param,
            "search_time": search_time_sec,
            "qps": query_data.size() / search_time_sec,
            "comparison": comparison,
            "cps": comparison / search_time_sec,
            "recall": groundtruth.recall(topk, knn)})
        # Save results
        json.dump(performance, open(os.path.join(perf_output, f"{dataset}.{i}.Hi-PNG-HNSW.json"), 'w'), indent=2)
 