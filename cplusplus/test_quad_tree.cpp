#include <iostream>
#include <quad_tree.hpp>
#include <utils/binary.hpp>
#include <utils/timer.hpp>
#include <utils/recall.hpp>

using namespace std;
using namespace anns::interval;
using namespace anns;

using interval::HiPNG;

using data_t = float;

int main(int argc, char **argv)
{
  DataSetWrapper<data_t> base("data/sift-128-euclidean.train.fvecs"), 
  query("data/sift-128-euclidean.test.fvecs");
  IntervalSetWrapper base_attr("data/sift-128-euclidean.uniform-0.0-1000.0.train.itv"), 
  query_attr("data/sift-128-euclidean.uniform-0.0-1000.0.test.itv");
  utils::GroundTruth gt("data/sift-128-euclidean.uniform-0.0-1000.0.uniform-0.0-1000.0.gt");
  const size_t k = 10;
  utils::Timer timer;
  std::cout << "leaf-size: ";
  float ls;
  cin >> ls;
  HiPNG<data_t, metrics::euclidean> index("hnsw", {ls, 32, 128});

  index.set_num_threads(48);
  timer.start();
  index.build(base, base_attr);
  timer.stop();
  cout << "Build time: " << timer.get() << endl;
  index.get_comparison_and_clear();

  // index.debug();

  // while (true)
  // {
  //   size_t k;
  //   float ef, mit, r;
  //   cout << "k, ef, mit, r: ";
  //   std::cin >> k >> ef >> mit >> r;
  //   timer.reset();
  //   timer.start();
  //   auto [_, knn] = index.search(query, query_attr, k, {ef, mit, r});
  //   timer.stop();
  //   cout << "Search time: " << timer.get() << endl;
  //   cout << "Recall: " << gt.recall(k, knn) << endl;
  //   // for (size_t j = 0; j < 20; j++)
  //   // {
  //   //   printf("query interval: [%.4f, %.4f]\n", query_attr[j].first, query_attr[j].second);
  //   //   std::cout << "groundtruth: ";
  //   //   for (size_t p = 0; p < k; p++)
  //   //   {
  //   //     std::cout << gt.data_[j * k + p] << " ";
  //   //   }
  //   //   std::cout << std::endl;
  //   //   std::cout << "answers: ";
  //   //   for (size_t p = 0; p < k; p++)
  //   //   {
  //   //     std::cout << knn[j * k + p] << " ";
  //   //   }
  //   //   std::cout << std::endl;
  //   // }
  // }

  return 0;
}