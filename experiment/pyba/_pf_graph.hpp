#pragma once

#include <pf_graph.hpp>

#include "_data.hpp"

using namespace anns;

namespace pyba
{

  class PostFilterGraph
  {
  private:
    std::unique_ptr<interval::IntervalBaseIndex<float>> index_{nullptr};

  public:
    PostFilterGraph(const std::string &underlying_graph, const std::vector<float> &params, const std::string &metric)
    {
      if (metric == "euclidean")
        index_ = std::make_unique<interval::PostFilterGraph<float, metrics::euclidean>>(underlying_graph, params);
      else if (metric == "angular")
        index_ = std::make_unique<interval::PostFilterGraph<float, metrics::cosine>>(underlying_graph, params);
      else if (metric == "dot")
        index_ = std::make_unique<interval::PostFilterGraph<float, metrics::inner_product>>(underlying_graph, params);
      else if (metric == "hamming")
        index_ = std::make_unique<interval::PostFilterGraph<float, metrics::hamming>>(underlying_graph, params);
      else
        throw std::runtime_error("Unsupported metric");
    }

    void build(const pyba::DataSet &base, const pyba::IntervalSet &attrs, size_t num_threads = 1)
    {
      index_->set_num_threads(num_threads);
      index_->build(base, attrs);
      index_->set_num_threads(1);
    }

    /* in pybind 11, there is never allowed two functions have the same name */
    res_t search(const DataSet &query, const IntervalSet &attrs, 
                 size_t k, const std::vector<float> &params, size_t num_threads = 1)
    {
      index_->set_num_threads(num_threads);
      auto ret = index_->search(query, attrs, k, params);
      index_->set_num_threads(1);
      return ret;
    }

    size_t index_size() const
    {
      return index_->index_size();
    }

    size_t get_comparison_and_clear() 
    {
      // assert(index_ && "Index not built");
      return index_->get_comparison_and_clear();
    }
  };

}