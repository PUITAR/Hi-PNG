#include <interval.hpp>
#include <iostream>
#include <pf_graph.hpp>
#include <utils/binary.hpp>
#include <utils/timer.hpp>
#include <utils/recall.hpp>

using namespace std;
using namespace anns::interval;
using namespace anns;
using namespace anns::utils;

using interval::PostFilterGraph;

using data_t = float;

int main(int argc, char** argv) {
  DataSetWrapper<data_t> base("data/sift-128-euclidean.train.fvecs"), 
  query("data/sift-128-euclidean.test.fvecs");
  IntervalSetWrapper base_attr("data/sift-128-euclidean.uniform-0.0-1.0.train.itv"), 
  query_attr("data/sift-128-euclidean.uniform-0.0-1.0.test.itv");
  utils::GroundTruth gt("data/sift-128-euclidean.uniform-0.0-1.0.uniform-0.0-1.0.gt");
  const size_t k = 10;
  utils::Timer timer;
  PostFilterGraph<data_t, metrics::euclidean> index("hnsw", {32, 128});
  index.set_num_threads(44); 
  timer.start();
  index.build(base, base_attr);
  timer.stop();
  cout << "Build time: " << timer.get() << endl;

  while (true)
  {
    float ef, iter;
    cin >> ef >> iter;
    timer.reset();
    timer.start();
    auto [_, knn] = index.search(query, query_attr, k, {ef, iter});
    timer.stop();
    cout << "Search time: " << timer.get() << endl;
    cout << "Recall: " << gt.recall(k, knn) << endl;
  }

  return 0;
}