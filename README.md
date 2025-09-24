# *Hi-PNG*: Efficient Interval-Filtering ANNS via Hierarchical Interval Partition Navigating Graph

[![DOI](https://zenodo.org/badge/929221163.svg)](https://doi.org/10.5281/zenodo.15540999)

This is the official implementation of the paper [*Hi-PNG*: Efficient Interval-Filtering ANNS via Hierarchical Interval Partition Navigating Graph](README.md).

## Requirements

- C++ 17
- Python
- OpenMP
  
```shell
# python requirements
pip install -r requirements.txt
```

## Build

Our experiment use `CMake` to compile, and use `pybind11` to bind `C++` code into `Python` package. To achieve this, u can use the code as follows:
```shell
mkdir -p build && cd build
cmake ..
make -j
```

## Dataset

Benchmark datasets in this paper are from [*ann-benchmark*](https://github.com/erikbern/ann-benchmarks), SIFT1M, GIST1M, GloVe, MNIST, DEEP1M, DBpedia-OpenAI and real-world data [UCF-Crime](https://www.kaggle.com/api/v1/datasets/download/odins0n/ucf-crime-dataset), [S\&P 500](https://www.kaggle.com/api/v1/datasets/download/footballjoe789/us-stock-dataset). Data is embedded into vector space for similarity search. Intervals for SIFT1M, GIST1M, GloVe, MNIST, DEEP1M, UCF-Crime, and DBpedia-OpenAI are generated from a distribution in $\mathbb{R}^2$, while S\&P 500 intervals are based on 2024 price ranges.
All dataset should be in the format of `fvecs`, `ivecs`, `fbin` and `ibin`.
You don't need to worry about any reproducibility issues with the data, as you can use the ``create_dataset.py`` script to generate the dataset.

```shell
mkdir -p data
python create_dataset.py -h
python create_dataset.py --dataset sift-128-euclidean \
  --train_distr uniform --train_left 0 --train_right 1000 \
  --test_distr uniform --test_left 0 --test_right 1000 \
  --k 100 --num_threads 48
```

## Experiments

All experimental sections are corresponding to a code file in ``experiments``.


| 🔬 Experiment | ✍ Code |
|--|--|
| Overall Experiment | ``overall.py`` | 
| Impact of Factors<br> (1) threads<br> (2) interval distribution<br> (3) query interval fraction | (1) ``impact_thread.py``<br> (2) ``overall.py``<br> (3) ``impact_rho.py`` |
| Impact of Parameters<br> (1) max iteration<br> (2) overlap threshold | (1) ``overall.py``<br> (2) ``overall.py`` |
| Effect of Partition Point Selection | ``balance.py`` |

## Reference
```
@inproceedings{10.1145/3711896.3736997,
author = {Yang, Ming and Cai, Yuzheng and Zheng, Weiguo},
title = {Hi-PNG: Efficient Interval-Filtering ANNS via Hierarchical Interval Partition Navigating Graph},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3736997},
doi = {10.1145/3711896.3736997},
abstract = {Approximate nearest neighbor search (ANNS) is widely used to retrieve similar vectors from high-dimensional data. However, many real-world applications require additional interval filtering based on numerical constraints, such as stock price ranges. In this paper, we introduce interval-filtering ANNS (IF-ANNS), a novel yet general retrieval task where both base and query vectors are associated with numerical intervals. The goal is to retrieve nearest neighbors whose intervals are fully contained within the query interval. To efficiently address this problem, we propose Hierarchical Interval Partitioning Navigating Graph (Hi-PNG), a framework that hierarchically partitions the interval space to efficiently locate the search space, eliminating unnecessary computations and achieving significant speedups. We provide a theoretical analysis of the search space, demonstrating the superiority of our approach. Extensive experiments on eight datasets, including commonly used benchmarks and real-world stock price data, show that Hi-PNG outperforms four state-of-the-art graph-based ANNS baselines, achieving up to 15\texttimes{} acceleration while maintaining high precision. These results highlight Hi-PNG's effectiveness in solving the IF-ANNS problem. All codes and data generation are available at https://github.com/PUITAR/Hi-PNG.git.},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {3518–3529},
numpages = {12},
keywords = {approximate nearest neighbor search, hierarchical interval partition, interval-filtering},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
