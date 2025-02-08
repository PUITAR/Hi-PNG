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
  DataSetWrapper<data_t> base("data/nytimes-256-angular.train.fvecs"), 
  query("data/nytimes-256-angular.test.fvecs");
  IntervalSetWrapper base_attr("data/nytimes-256-angular.poisson-100.0.train.itv"), 
  query_attr("data/nytimes-256-angular.poisson-100.0.test.itv");
  utils::GroundTruth gt("data/nytimes-256-angular.poisson-100.0.poisson-100.0.gt");
  const size_t k = 10;
  utils::Timer timer;
  cout << 1 << endl;
  PostFilterGraph<data_t, metrics::cosine> index("hcnng", {15, 100, 3});
  index.set_num_threads(1); 
  timer.start();
  cout << 2 << endl;
  index.build(base, base_attr);
  cout << 3 << endl;
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