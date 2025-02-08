/**
 * @author Ming Yang
 * @date 2024/10/18
 */
#include <anns.hpp>
#include <interval.hpp>
#include <distance.hpp>
#include <utils/binary.hpp>
#include <utils/resize.hpp>
#include <utils/timer.hpp>

#include <vector>
#include <cstring>
#include <iostream>
#include <queue>

const static char help[] =
    "The Program needs exact 7 parameters   \n\
    (1) Base Vector File Name               \n\
    (2) Query Vector File Name              \n\
    (3) Base Interval File Name             \n\
    (4) Query Interval File Name            \n\
    (5) Ground Truth File Name              \n\
    (6) The number of nearest neighbors     \n\
    (7) The number of threads               \n";

int main(int argc, char **argv)
{
  if (argc != 1 + 7)
  {
    std::cerr << help;
    exit(-1);
  }

  using data_t = float; // current only support float
  using label_t = float;
  using namespace anns::interval;
  using namespace anns;
  using namespace anns::utils;

  float (*distance)(const data_t *, const data_t *, size_t) = nullptr;
  if (strstr(argv[1], "euclidean"))
  {
    distance = anns::metrics::euclidean;
  }
  else if (strstr(argv[1], "hamming"))
  {
    distance = anns::metrics::hamming;
  }
  else if (strstr(argv[1], "angular"))
  {
    distance = anns::metrics::cosine;
  }
  else
  {
    std::cerr << "Unknown Metric" << std::endl;
    exit(-1);
  }

  DataSetWrapper<data_t> base(argv[1]), query(argv[2]);
  IntervalSetWrapper base_attr(argv[3]), query_attr(argv[4]);
  size_t k = std::atoi(argv[6]);
  size_t nt = std::atoi(argv[7]);
  std::vector<int> gt(query.num_ * k, MAGIC_ID);

  Timer timer;
  timer.start();

#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
  for (size_t i = 0; i < query.num_; i++)
  {
    const data_t *q = query[i];
    const auto &qitv = query_attr[i];
    std::priority_queue<std::pair<float, size_t>> pq;
    for (size_t j = 0; j < base.num_; j++)
    {
      if (Interval::conquer(qitv, base_attr[j]))
      {
        pq.emplace(distance(base[j], q, base.dim_), j);
        if (pq.size() > k)
          pq.pop();
      }
    }
    while (pq.size() < k)
    {
      pq.emplace(MAGIC_DIST, MAGIC_ID);
    }
    for (int j = k - 1; j >= 0; j--)
    {
      gt[i * k + j] = pq.top().second;
      pq.pop();
    }
  }

  timer.stop();
  std::cout << "Time: " << timer.get() << " s" << std::endl;
  write_to_file(gt, {query.num_, k}, argv[5]);

  return 0;
}