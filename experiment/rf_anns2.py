"""
Rebuttal: RF-ANNS
"""

datasets = [
  'sift-128-euclidean', 
]

from function import create_dataset
from os import system
import sys, os
sys.path.append("../data_collect")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_collect.binary import fvecs_read, ibin_save

for dataset in datasets:
    try:
        # create dataset
        system(f"""
        cd .. && \
        python create_dataset.py --dataset {dataset} \
            --train_distr uniform --train_left 0 --train_right 1000 \
            --test_distr uniform --test_left 0 --test_right 100 \
            --k 100 --num_threads 48
        """)
    except Exception as e:
        print(e)


from anns import DataSet, IntervalSet, GroundTruth, Timer, HiPNG, PostFilterGraph
import os
from tqdm import tqdm
import json
import numpy as np

topk = 10
num_threads = 48
perf_output = "perf/rf_anns2"
index_type = "hnsw"
distribution1 = "uniform-0.0-1000.0"
distribution2 = "uniform-0.0-100.0"
data_directory = "../data"

base_interval = fvecs_read(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
base_interval = base_interval.astype(np.int32)
a1 = base_interval[:,0].reshape(-1, 1)
a2 = base_interval[:,1].reshape(-1, 1)
ibin_save(f"../data/{dataset}.iRangeGraph.a1", a1)
ibin_save(f"../data/{dataset}.iRangeGraph.a2", a2)

if not os.path.exists(perf_output):
    os.makedirs(perf_output)

index_param_1, index_param_2 = [32, 128], [10000, 32, 128]
search_params_1, search_params_2 = [[ef, iter] for ef in range(10, 200, 20) for iter in range(1, 10, 2)], \
    [[ef, iter, sim/100] for ef in range(10, 30, 4) for iter in range(1, 10, 1) for sim in range(0, 101, 20)]

timer = Timer()

# for dataset in datasets:
#     print(f"Running dataset: {dataset}")

#     # Load dataset
#     base_data = DataSet(os.path.join(data_directory, f"{dataset}.train.fvecs"))
#     query_data = DataSet(os.path.join(data_directory, f"{dataset}.test.fvecs"))
#     base_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution1}.train.itv"))
#     query_attrs = IntervalSet(os.path.join(data_directory, f"{dataset}.{distribution2}.test.itv"))
#     groundtruth = GroundTruth(os.path.join(data_directory, f"{dataset}.{distribution1}.{distribution2}.gt"))

#     # Benchmark for PostFilterGraph
#     model = PostFilterGraph(index_type, index_param_1, "euclidean")
#     timer.reset()
#     timer.start()
#     model.build(base_data, base_attrs, num_threads)
#     timer.stop()
#     build_time_sec = timer.get()
#     model.get_comparison_and_clear()
#     performance = []
#     for search_param in tqdm(search_params_1):
#         timer.reset()
#         timer.start()
#         _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads)
#         timer.stop()
#         search_time_sec = timer.get()
#         comparison = model.get_comparison_and_clear()
#         performance.append({
#             "name": "HNSW",
#             "index_param": index_param_1,
#             "index_size": model.index_size(),
#             "build_time": build_time_sec,
#             "k": topk,
#             "search_param": search_param,
#             "search_time": search_time_sec,
#             "qps": query_data.size() / search_time_sec,
#             "comparison": comparison,
#             "cps": comparison / search_time_sec,
#             "recall": groundtruth.recall(topk, knn)})
#         # Save results
#         json.dump(performance, open(os.path.join(perf_output, f"{dataset}.HNSW.json"), 'w') ,indent=2)
        
#     # Benchmark for Hi-PNG
#     model = HiPNG(index_type, index_param_2, "euclidean")
#     timer.reset()
#     timer.start()
#     model.build(base_data, base_attrs, num_threads)
#     timer.stop()
#     build_time_sec = timer.get()
#     model.get_comparison_and_clear()
#     performance = []
#     for search_param in tqdm(search_params_2):
#         timer.reset()
#         timer.start()
#         _, knn = model.search(query_data, query_attrs, topk, search_param, num_threads)
#         timer.stop()
#         search_time_sec = timer.get()
#         comparison = model.get_comparison_and_clear()
#         performance.append({
#             "name": "Hi-PNG-HNSW",
#             "index_param": index_param_2,
#             "index_size": model.index_size(),
#             "build_time": build_time_sec,
#             "k": topk,
#             "search_param": search_param,
#             "search_time": search_time_sec,
#             "qps": query_data.size() / search_time_sec,
#             "comparison": comparison,
#             "cps": comparison / search_time_sec,
#             "recall": groundtruth.recall(topk, knn)})
#         # Save results
#         json.dump(performance, open(os.path.join(perf_output, f"{dataset}.Hi-PNG-HNSW.json"), 'w'), indent=2)


"""
Rebuttal: iRangeGraph
"""

datasets = [
    'sift-128-euclidean',
]

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from data_collect.binary import fvecs_read, ivecs_read, fbin_save, ibin_save
from tqdm import tqdm

# Change `vecs` type data into `bin` type data
for dataset in tqdm(datasets):
    try:
        # create dataset
        if os.path.exists(f"../data/{dataset}.train.fbin") and os.path.exists(f"../data/{dataset}.test.fbin"):
            print(f"Dataset {dataset} already exists.")
            continue
        base_data = fvecs_read(f"../data/{dataset}.train.fvecs")
        query_data = ivecs_read(f"../data/{dataset}.test.fvecs")
        fbin_save(f"../data/{dataset}.train.fbin", base_data)
        ibin_save(f"../data/{dataset}.test.fbin", query_data)
    except Exception as e:
        print(e)

M = 32
ef_construction = 128
num_threads = 48

for dataset in datasets:
    print(f"Running dataset: {dataset}")

    if not os.path.exists(f"../data/{dataset}.iRangeGraph.index"):
        os.system(f"""
        ./RF-ANNS/iRangeGraph/build/tests/buildindex \
            --data_path ../data/{dataset}.train.fbin \
            --index_file ../data/{dataset}.iRangeGraph.index \
            --M {M} \
            --ef_construction {ef_construction} \
            --threads {num_threads}
        """)

    os.system(f"""
    ./RF-ANNS/iRangeGraph/build/tests/search_multi \
        --data_path ../data/{dataset}.train.fbin \
        --query_path ../data/{dataset}.test.fbin \
        --range_saveprefix ../data/{dataset}.iRangeGraph.range \
        --groundtruth_saveprefix ../data/{dataset}.iRangeGraph.groundtruth \
        --index_file ../data/{dataset}.iRangeGraph.index \
        --result_saveprefix perf/rf_anns2/{dataset}.iRangeGraph.result \
        --attribute1 ../data/{dataset}.iRangeGraph.a1 \
        --attribute2 ../data/{dataset}.iRangeGraph.a2\
        --M {M}
    """)