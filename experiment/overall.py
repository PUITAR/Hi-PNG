datasets = [
  'sift-128-euclidean', 
  'glove-200-angular',
  'gist-960-euclidean',
  'fashion-mnist-784-euclidean', 
  'deep-image-96-angular',
  'ucf-crime-4096-euclidean',
  'dbpedia-openai-1000k-angular',
  'us-stock-384-euclidean',
]

from function import create_dataset

for dataset in datasets:
  try:
    # download dataset
    create_dataset(dataset)
  except Exception as e:
    print(e)

def overall(dataset, indice):
  from function import plot_qps_recall_curve

  k = 10
  num_threads = 48
  perf_output = "perf/overall"
  figure_output = "figure/overall"
  
  # Framework parameters
  # Posterfilter Build Parameters
  pf_bp = {
    'hnsw': [[M, efc] for M in [32] for efc in [128]],
    'vamana': [[R, efc, alpha] for R in [32] for efc in [128] for alpha in [1.2]],
    'nsg': [[R, efc] for R in [32] for efc in [128]],
    'hcnng': [[T, Ls, s] for T in [15] for Ls in [1000] for s in [3]]
  }[indice]
  # PosterFilter Query Parameters
  # pf_qp = [[ef, iter] for ef in range(10, 256, 10) for iter in range(1, 10, 2)]
  pf_qp = [[ef, iter] for ef in range(10, 200, 20) for iter in range(1, 10, 2)]
  # HiPNG Build Parameters
  qt_bp = {
    'hnsw': [[ls, M, efc] for ls in [10000] for M in [32] for efc in [128]],
    'vamana': [[ls, R, efc, alpha] for ls in [10000] for R in [32] for efc in [64] for alpha in [1.2]],
    'nsg': [[ls, R, efc] for ls in [10000] for R in [32] for efc in [128]],
    'hcnng': [[ls, T, Ls, s] for ls in [10000] for T in [15] for Ls in [1000] for s in [3]]
  }[indice]
  # HiPNG Query Parameters
  # qt_qp = [[ef, iter, sim/100] for ef in range(10, 30, 2) for iter in range(1, 10, 2) for sim in range(30, 101, 5)]
  qt_qp = [[ef, iter, sim/100] for ef in range(10, 30, 4) for iter in range(1, 10, 2) for sim in range(0, 101, 20)]
  
  if dataset == 'us-stock-384-euclidean':
    distribution = ""
  else:
    distribution = "uniform-0.0-1000.0"
    # distribution = "normal-0.0-1.0"
    # distribution = "poisson-100.0"

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
    query_attrs_path = os.path.join(data_dir, f"{dataset}.{distribution}.test.itv")
    gt_path = os.path.join(data_dir, f"{dataset}.{distribution}.{distribution}.gt")
    name = f"{method}.{indice}.{dataset}.{distribution}.{distribution}.k{k}"
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
  
  # Run 
  plot_qps_recall_curve(
    do_grid_search('pf'), do_grid_search('qt'), 
    # f"{figure_output}/{dataset}.{distribution}.{distribution}.{indice}.png", 
    underlying_graph=indice
  )
  

indices = [
  'hnsw', 
  'vamana', 
  'nsg', 
  'hcnng'
]

for dataset in datasets:
  for index in indices:
    overall(dataset, index)


