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
from function import create_dataset

#################################### Data Generation ####################################
dataset = 'covid19-epitope-prediction-euclidean'

create_dataset(dataset)

distribution1 = ""
distribution2 = ""
data_directory = "../data"
topk = 10
num_threads_build = 48
num_threads_search = 1
perf_output = "perf/covid19"

if not os.path.exists(perf_output):
    os.makedirs(perf_output)

################################ Benchmark for PF-HNS and Hi-PNG ################################
# Load dataset
base_data = DataSet(os.path.join(data_directory, f"{dataset}.train.fvecs"))
query_data = DataSet(os.path.join(data_directory, f"{dataset}.test.fvecs"))
base_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
query_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))
groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

index_type = "hnsw"

index_param_1, index_param_2 = [32, 128], [100, 32, 128]
search_params_1, search_params_2 = [[ef, iter] for ef in range(10, 200, 20) for iter in range(1, 10, 2)], \
    [[ef, iter, sim/100] for ef in range(10, 30, 4) for iter in range(1, 10, 1) for sim in range(0, 101, 20)]

timer = Timer()

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
    json.dump(performance, open(os.path.join(perf_output, f"{dataset}.HNSW.json"), 'w') ,indent=2)
    
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
    json.dump(performance, open(os.path.join(perf_output, f"{dataset}.Hi-PNG-HNSW.json"), 'w'), indent=2)

################################ Benchmark for QuadTree ################################
base_data = fvecs_read(os.path.join(data_directory, f"{dataset}.train.fvecs"))
query_data = fvecs_read(os.path.join(data_directory, f"{dataset}.test.fvecs"))
base_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
query_attrs = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))
groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

imin, imax = base_attrs.min(), base_attrs.max()

timer = Timer()

# initialize ray
ray.init(num_cpus=num_threads_search)

# Build QuadTree index
timer.start()
spindex = pyqtree.Index(bbox=(imin, imin, imax, imax))
for i, (l, r) in enumerate(base_attrs):
    spindex.insert(item=i, bbox=(l,r,l,r))
timer.stop()
build_time_sec = timer.get()

@ray.remote
def batch_query(queries, attrs, topk, spindex, base_data, base_attrs):
    results = []
    for q, attr in zip(queries, attrs):
        region = (attr[0], attr[0], attr[1], attr[1])
        inter = spindex.intersect(region)
        idx = np.array([id for id in inter if (base_attrs[id][0] >= attr[0] and base_attrs[id][1] <= attr[1])])
        knn = []
        if len(idx) > 0:
            X = base_data[idx]
            dists = np.sum((X - q)**2, axis=1)
            actual_topk = min(len(dists), topk)
            topk_idx = np.argpartition(dists, actual_topk-1)[:actual_topk]  # Select indices with smallest distances
            knn = idx[topk_idx].tolist() 
        knn = knn + [0x3f3f3f3f] * max(0, topk-len(knn))
        results.append(knn)
    return results

# Split the queries and attributes into num_threads chunks
chunk_size = (len(query_data) + num_threads_search - 1) // num_threads_search
query_chunks = [query_data[i:i+chunk_size] for i in range(0, len(query_data), chunk_size)]
attr_chunks = [query_attrs[i:i+chunk_size] for i in range(0, len(query_attrs), chunk_size)]

spindex = ray.put(spindex)
base_data = ray.put(base_data)
base_attrs = ray.put(base_attrs)

timer.reset()
# Process chunks in parallel
timer.start()
result_chunks = ray.get([
    batch_query.remote(q_chunk, a_chunk, topk, spindex, base_data, base_attrs)
    for q_chunk, a_chunk in zip(query_chunks, attr_chunks)
])
timer.stop()
search_time_sec = timer.get()

# Combine results
knn = [n for chunk in result_chunks for knn in chunk for n in knn]
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

ray.shutdown()

################################ Benchmark for R-Tree ################################
base_data = fvecs_read(os.path.join(data_directory, f"{dataset}.train.fvecs"))
query_data = fvecs_read(os.path.join(data_directory, f"{dataset}.test.fvecs"))
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

def batch_query(queries, attrs, topk, rt_index, base_data, base_attrs):
    results = []
    for q, (l,r) in zip(queries, attrs):
        inter = list(rt_index.intersection((l, l, r, r), objects=False))
        idx = np.array([id for id in inter if (base_attrs[id][0] >= l and base_attrs[id][1] <= r)])
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
timer.start()
results = batch_query(query_data, query_attrs, topk, rt_index, base_data, base_attrs)
timer.stop()
search_time_sec = timer.get()

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
