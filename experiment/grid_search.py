from anns import DataSet, IntervalSet, GroundTruth, Timer, QuadTree, PostFilterGraph
import os


''' A grid search base class '''
class GridSearch:
  def __init__(self, 
               name: str, base_path: str, query_path: str, base_attrs_path: str, query_attrs_path: str, groundtruth_path: str,
               k: int, build_parameters: list[list[float]], query_parameters: list[list[float]],
               index_snapshot_dir = None):
    self.name = name
    self.base_data = DataSet(base_path)
    self.query_data = DataSet(query_path)
    self.base_attrs = IntervalSet(base_attrs_path)
    self.query_attrs = IntervalSet(query_attrs_path)
    self.groundtruth = GroundTruth(groundtruth_path)
    self.k = k
    self.build_parameters = build_parameters
    self.query_parameters = query_parameters
    self.index_snapshot_dir = index_snapshot_dir
    self.model = None
    # determine underlying graph type
    if "hnsw" in self.name:
      self.underlying_graph = "hnsw"
    elif "vamana" in self.name:
      self.underlying_graph = "vamana"
    elif "nsg" in self.name:
      self.underlying_graph = "nsg"
    elif "hcnng" in self.name:
      self.underlying_graph = "hcnng"
    else:
      raise ValueError(f"Unknown graph type: {self.name}")
    # determine metric
    if "euclidean" in self.name:
      self.metric = "euclidean"
    elif "hamming" in self.name:
      self.metric = "hamming"
    elif "angular" in self.name:
      self.metric = "angular"
    elif "dot" in self.name:
      self.metric = "dot"
    else:
      raise ValueError(f"Unknown metric: {self.name}")
    
  def build(self, bp, num_threads):
    # build model
    if "pf" in self.name:
      self.model = PostFilterGraph(self.underlying_graph, bp, self.metric)
    elif "qt" in self.name:
      self.model = QuadTree(self.underlying_graph, bp, self.metric)
    else:
      raise ValueError(f"Unknown index type: {self.name}")
    self.model.build(self.base_data, self.base_attrs, num_threads) 

  def query(self, qp, num_threads):
    return self.model.search(self.query_data, self.query_attrs, self.k, qp, num_threads)

  # run the grid search
  def run(self, output_dir: str, num_threads = 1): 
    from tqdm import tqdm
    import json
    gs_results = []
    timer = Timer()
    for bp in self.build_parameters:
      print(f"Building {self.name} with parameters {bp}")
      timer.reset()
      timer.start()
      self.build(bp, num_threads)
      timer.stop()
      build_time_sec = timer.get()
      self.model.get_comparison_and_clear()
      if self.index_snapshot_dir:
        if not os.path.exists(self.index_snapshot_dir):
          os.makedirs(self.index_snapshot_dir)
        self.model.save(os.path.join(self.index_snapshot_dir, f"{self.name}-{bp}.idx"))
      for qp in tqdm(self.query_parameters):
        timer.reset()
        timer.start()
        _, knn = self.query(qp, num_threads)
        timer.stop()
        query_time_sec = timer.get()
        comparison = self.model.get_comparison_and_clear()
        gs_results.append({
          "name": self.name,
          "build_parameters": bp,
          "build_time": build_time_sec,
          "index_size": self.model.index_size(),
          "k": self.k,
          "query_parameters": qp,
          "query_time": query_time_sec,
          "qps": self.query_data.size() / query_time_sec,
          "comparison": comparison,
          "cps": comparison / query_time_sec,
          "recall": self.groundtruth.recall(self.k, knn)})
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        with open(os.path.join(output_dir, f"{self.name}.json"), 'w') as f:
          json.dump(gs_results, f, indent=2)

  def run2(self, output_dir: str, num_threads_list): 
    from tqdm import tqdm
    import json
    timer = Timer()
    for bp in self.build_parameters:
      print(f"Building {self.name} with parameters {bp}")
      timer.reset()
      timer.start()
      self.build(bp, 48)
      timer.stop()
      build_time_sec = timer.get()
      self.model.get_comparison_and_clear()
      for num_threads in num_threads_list:
        gs_results = [] 
        for qp in tqdm(self.query_parameters):
          timer.reset()
          timer.start()
          _, knn = self.query(qp, num_threads)
          timer.stop()
          query_time_sec = timer.get()
          comparison = self.model.get_comparison_and_clear()
          gs_results.append({
            "name": self.name,
            "build_parameters": bp,
            "index_size": self.model.index_size(),
            "build_time": build_time_sec,
            "k": self.k,
            "query_parameters": qp,
            "query_time": query_time_sec,
            "qps": self.query_data.size() / query_time_sec,
            "comparison": comparison,
            "cps": comparison / query_time_sec,
            "recall": self.groundtruth.recall(self.k, knn)})
          if not os.path.exists(output_dir):
            os.makedirs(output_dir)
          with open(os.path.join(output_dir, f"{self.name}.t{num_threads}.json"), 'w') as f:
            json.dump(gs_results, f, indent=2)


