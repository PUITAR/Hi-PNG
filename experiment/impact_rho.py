dataset = 'sift-128-euclidean'

rhoes = [2.0**(0), 2**(-1), 2**(-2), 2**(-3), 2**(-4)]

k = 10
num_threads = 48
perf_output = "perf/factor/rho"
distribution = "uniform-0.0-1.0"

def impact_rho(index, rho):
  pf_bp = {
    'hnsw': [[M, efc] for M in [32] for efc in [128]],
    'vamana': [[R, efc, alpha] for R in [32] for efc in [128] for alpha in [1.2]],
    'nsg': [[R, efc] for R in [32] for efc in [128]],
    'hcnng': [[T, Ls, s] for T in [15] for Ls in [1000] for s in [3]]
  }[index]
  # PosterFilter Query Parameters
  pf_qp = [[ef, iter] for ef in range(10, 256, 10) for iter in range(1, 10, 2)]
  # IPNG Build Parameters
  qt_bp = {
    'hnsw': [[ls, M, efc] for ls in [10000] for M in [32] for efc in [128]],
    'vamana': [[ls, R, efc, alpha] for ls in [10000] for R in [32] for efc in [64] for alpha in [1.2]],
    'nsg': [[ls, R, efc] for ls in [10000] for R in [32] for efc in [128]],
    'hcnng': [[ls, T, Ls, s] for ls in [10000] for T in [15] for Ls in [1000] for s in [3]]
  }[index]
  # IPNG Query Parameters
  qt_qp = [[ef, iter, sim/100] for ef in range(10, 30, 2) for iter in range(1, 10, 2) for sim in range(30, 101, 5)]

  from os import system
  
  system(f"""
  cd .. && \
  HF_ENDPOINT=https://hf-mirror.com python create_dataset.py --dataset {dataset} \
    --train_distr uniform --train_left 0 --train_right 1 \
    --test_distr uniform --test_left 0 --test_right {rho} \
    --k 100 --num_threads 48
  """)

  def do_grid_search(method):
    bp = pf_bp if method == "pf" else qt_bp
    qp = pf_qp if method == "pf" else qt_qp
    import os
    from grid_search import GridSearch
    # file paths
    data_dir = "../data"
    base_path = os.path.join(data_dir, f"{dataset}.train.fvecs")
    query_path = os.path.join(data_dir, f"{dataset}.test.fvecs")
    base_attrs_path = os.path.join(data_dir, f"{dataset}.{distribution}.train.itv")
    query_attrs_path = os.path.join(data_dir, f"{dataset}.uniform-0.0-{rho}.test.itv")
    gt_path = os.path.join(data_dir, f"{dataset}.{distribution}.uniform-0.0-{rho}.gt")
    name = f"{method}.{index}.{dataset}.{distribution}.uniform-0.0-{rho}.k{k}"
    # run grid search
    GridSearch(
      name = name,
      base_path = base_path,
      query_path = query_path,
      base_attrs_path = base_attrs_path,
      query_attrs_path = query_attrs_path,
      groundtruth_path = gt_path,
      k = k,
      build_parameters = bp,
      query_parameters = qp
    ).run(
      perf_output, num_threads)
    return os.path.join(perf_output, f"{name}.json")

  do_grid_search('pf')
  do_grid_search('qt')

for index in ['hnsw', 'vamana', 'nsg', 'hcnng']:
  for rho in rhoes:
    impact_rho(index, rho)