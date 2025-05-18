import sys
from os import system
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anns import DataSet, IntervalSet, GroundTruth, Timer, HiPNG, PostFilterGraph
import os
from tqdm import tqdm
import json
import ray, pyqtree, rtree

from data_collect.binary import fvecs_read
import numpy as np


#################################### Data Generation ####################################
dataset = 'sift-128-euclidean'
distribution1 = "uniform-0.0-1000.0"
distribution2 = "uniform-0.0-100.0"
data_directory = "../data"
topk = 10
num_threads_build = 48
num_threads_search = 1
perf_output = "perf/sp"

# system(f"""
#   cd .. && \
#   python create_dataset.py --dataset {dataset} \
#     --train_distr uniform --train_left 0 --train_right 1000 \
#     --test_distr uniform --test_left 0 --test_right 100 \
#     --k 100 --num_threads 48
#   """)

# if not os.path.exists(perf_output):
#     os.makedirs(perf_output)

# ################################ Benchmark for PF-HNS and Hi-PNG ################################
# # Load dataset
# base_data = DataSet(os.path.join(data_directory, f"{dataset}.train.fvecs"))
# query_data = DataSet(os.path.join(data_directory, f"{dataset}.test.fvecs"))
# base_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
# query_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))
# groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

# index_type = "hnsw"

# index_param_1, index_param_2 = [32, 128], [10000, 32, 128]
# search_params_1, search_params_2 = [[ef, iter] for ef in range(10, 200, 20) for iter in range(1, 10, 2)], \
#     [[ef, iter, sim/100] for ef in range(10, 30, 4) for iter in range(1, 8, 2) for sim in range(0, 101, 20)]

# timer = Timer()

# # Benchmark for PostFilterGraph
# model = PostFilterGraph(index_type, index_param_1, "euclidean")
# timer.reset()
# timer.start()
# model.build(base_data, base_attrs, num_threads_build)
# timer.stop()
# build_time_sec = timer.get()
# model.get_comparison_and_clear()
# performance = []
# for search_param in tqdm(search_params_1):
#     timer.reset()
#     timer.start()
#     _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads_search)
#     timer.stop()
#     search_time_sec = timer.get()
#     comparison = model.get_comparison_and_clear()
#     performance.append({
#         "name": "HNSW",
#         "index_param": index_param_1,
#         "index_size": model.index_size(),
#         "build_time": build_time_sec,
#         "k": topk,
#         "search_param": search_param,
#         "search_time": search_time_sec,
#         "qps": query_data.size() / search_time_sec,
#         "comparison": comparison,
#         "cps": comparison / search_time_sec,
#         "recall": groundtruth.recall(topk, knn)})
#     # Save results
#     json.dump(performance, open(os.path.join(perf_output, f"{dataset}.HNSW.json"), 'w') ,indent=2)
    
# # Benchmark for Hi-PNG
# model = HiPNG(index_type, index_param_2, "euclidean")
# timer.reset()
# timer.start()
# model.build(base_data, base_attrs, num_threads_build)
# timer.stop()
# build_time_sec = timer.get()
# model.get_comparison_and_clear()
# performance = []
# for search_param in tqdm(search_params_2):
#     timer.reset()
#     timer.start()
#     _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads_search)
#     timer.stop()
#     search_time_sec = timer.get()
#     comparison = model.get_comparison_and_clear()
#     performance.append({
#         "name": "Hi-PNG-HNSW",
#         "index_param": index_param_2,
#         "index_size": model.index_size(),
#         "build_time": build_time_sec,
#         "k": topk,
#         "search_param": search_param,
#         "search_time": search_time_sec,
#         "qps": query_data.size() / search_time_sec,
#         "comparison": comparison,
#         "cps": comparison / search_time_sec,
#         "recall": groundtruth.recall(topk, knn)})
#     # Save results
#     json.dump(performance, open(os.path.join(perf_output, f"{dataset}.Hi-PNG-HNSW.json"), 'w'), indent=2)

################################ Benchmark for QuadTree ################################

base_data = fvecs_read(os.path.join(data_directory, f"{dataset}.train.fvecs"))
query_data = fvecs_read(os.path.join(data_directory, f"{dataset}.test.fvecs"))[:100]
base_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
query_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))

groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

imin, imax = base_attrs.min(), base_attrs.max()

timer = Timer()
timer_filter = Timer()

# Build QuadTree index
timer.start()
spindex = pyqtree.Index(bbox=(imin, imin, imax, imax))
for i, (l, r) in enumerate(base_attrs):
    spindex.insert(item=i, bbox=(l, l, r, r))
timer.stop()
build_time_sec = timer.get()

def batch_query(queries, attrs, topk, spindex, base_data, base_attrs):
    results = []
    for q, (l, r) in zip(queries, attrs):
        timer_filter.start()
        region = (l, l, r, r)
        inter = spindex.intersect(region)
        idx = np.array([id for id in inter if (base_attrs[id][0] >= l and base_attrs[id][1] <= r)])
        timer_filter.stop()
        knn = []
        if len(idx) > 0:
            X = base_data[idx]
            dists = np.sum((X - q)**2, axis=1)
            actual_topk = min(len(dists), topk)
            topk_idx = np.argpartition(dists, actual_topk-1)[:actual_topk]
            knn = idx[topk_idx].tolist() 
        knn = knn + [0x3f3f3f3f] * max(0, topk-len(knn))
        results.append(knn)
    return results

timer.reset()
# Process all queries in single thread
print(158)
timer.start()
results = batch_query(query_data, query_attrs, topk, spindex, base_data, base_attrs)
timer.stop()
search_time_sec = timer.get()
search_time_filter = timer_filter.get()

print(f"filter time: {search_time_filter}")
print(f"search time: {search_time_sec}")

# Combine results
knn = [n for knn in results for n in knn]
recall = groundtruth.recall(topk, knn)
qps = len(query_data) / search_time_sec

performance = {
    "name": "QuadTree",
    "index_size": None,  # QuadTree size could be added if needed
    "build_time": build_time_sec,
    "k": topk,
    "search_time": search_time_sec,
    "qps": qps,
    "recall": recall
}
json.dump([performance], open(os.path.join(perf_output, f"{dataset}.QuadTree.json"), 'w'), indent=2)

################################ Benchmark for R-Tree ################################
base_data = fvecs_read(os.path.join(data_directory, f"{dataset}.train.fvecs"))
query_data = fvecs_read(os.path.join(data_directory, f"{dataset}.test.fvecs"))[:100]
base_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
query_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))
groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

timer = Timer()

# Build R-Tree index with optimized parameters
timer.start()
# Create R-Tree with proper properties
rt_index = rtree.index.Index()
for i, (l, r) in enumerate(base_attrs):
    # Insert with proper bounding box format: (minx, miny, maxx, maxy)
    assert l <= r
    rt_index.insert(i, (l, r, l, r))

timer.stop()
build_time_sec = timer.get()
timer_filter = Timer()

def batch_query(queries, attrs, topk, rt_index, base_data, base_attrs):
    results = []
    for q, (l,r) in zip(queries, attrs):
        timer_filter.start()
        inter = list(rt_index.intersection((l, l, r, r), objects=False))
        idx = np.array([id for id in inter if (base_attrs[id][0] >= l and base_attrs[id][1] <= r)])
        timer_filter.stop()
        knn = []
        if len(idx) > 0:
            X = base_data[idx]
            dists = np.sum((X - q)**2, axis=1)
            actual_topk = min(len(dists), topk)
            topk_idx = np.argpartition(dists, actual_topk-1)[:actual_topk]
            knn = idx[topk_idx].tolist() 
        knn = knn + [0x3f3f3f3f] * max(0, topk-len(knn))
        results.append(knn)
    return results

print(224)
timer.reset()
# Process all queries in single thread
timer.start()
results = batch_query(query_data, query_attrs, topk, rt_index, base_data, base_attrs)
timer.stop()
search_time_sec = timer.get()
search_time_filter = timer_filter.get()
print(f"filter time: {search_time_filter}")
print(f"search time: {search_time_sec}")

# Combine results
knn = [n for knn in results for n in knn]
recall = groundtruth.recall(topk, knn)
qps = len(query_data) / search_time_sec
performance = {
    "name": "RTree",
    "index_size": None,  # R-Tree provides size information
    "build_time": build_time_sec,
    "k": topk,
    "search_time": search_time_sec,
    "qps": qps,
    "recall": recall
}
json.dump([performance], open(os.path.join(perf_output, f"{dataset}.RTree.json"), 'w'), indent=2)