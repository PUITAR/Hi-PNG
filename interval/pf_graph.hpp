#pragma once

#include <graph/hcnng.hpp>
#include <graph/hnsw.hpp>
#include <graph/nsg.hpp>
#include <graph/vamana.hpp>

#include "interval.hpp"
#include <anns.hpp>

using namespace anns::graph;

namespace anns
{

  namespace interval
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class PostFilterGraph : public IntervalBaseIndex<data_t>
    {
    protected:
      std::unique_ptr<Base<data_t>> graph_{nullptr};
      IntervalSet ft_;
      size_t num_threads_{1};

    public:
      PostFilterGraph(const std::string &name, const std::vector<float> &params)
      {
        if (name == "hnsw" && params.size() == 2)
          graph_ = std::make_unique<graph::HNSW<data_t, distance>>(params[0], params[1]);
        else if (name == "nsg" && params.size() == 2)
          graph_ = std::make_unique<graph::NSG<data_t, distance>>(params[0], params[1]);
        else if (name == "vamana" && params.size() == 3)
          graph_ = std::make_unique<graph::Vamana<data_t, distance>>(params[0], params[1], params[2]);
        else if (name == "hcnng" && params.size() == 3)
          graph_ = std::make_unique<graph::HCNNG<data_t, distance>>(params[0], params[1], params[2]);
        else
          throw std::runtime_error("Graph Parameters Setting ERROR");
      }

      // PostFilterGraph(const std::string &name, const DataSet<data_t> &base, const IntervalSet &ft, const std::string &fname)
      // {
      //   if (name == "hnsw")
      //     graph_ = std::make_unique<graph::HNSW<data_t, distance>>(base, fname);
      //   else if (name == "nsg")
      //     graph_ = std::make_unique<graph::NSG<data_t, distance>>(base, fname);
      //   else if (name == "vamana")
      //     graph_ = std::make_unique<graph::Vamana<data_t, distance>>(base, fname);
      //   else if (name == "hcnng")
      //     graph_ = std::make_unique<graph::HCNNG<data_t, distance>>(base, fname);
      //   else
      //     throw std::runtime_error("Graph Parameters Setting ERROR");
      //   ft_ = ft;
      // }

      void build(const DataSet<data_t> &base, const IntervalSet &ft) override
      {
        assert(base.num_ == ft.num_ && graph_);
        ft_ = ft;
        graph_->set_num_threads(num_threads_);
        graph_->build(base);
        graph_->set_num_threads(1);
      }

      inline res_t search(const DataSet<data_t> &query, size_t k, size_t ef) 
      {
        if (graph_ == nullptr)
        {
          throw std::runtime_error("Graph not initialized");
        }
        return graph_->search(query, k, ef);
      }

      res_t search(const DataSet<data_t> &query, const IntervalSet &attrs,
                   size_t k, const std::vector<float> &params) override
      {
        assert(query.num_ == attrs.num_ && graph_);
        size_t ef = params[0], max_iteration = params[1];
        knn_t knn(query.num_ * k, MAGIC_ID);
        dis_t dis(query.num_ * k, MAGIC_DIST);
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < query.num_; i++)
        {
          size_t actual_k = k, iteration = 0;
          std::priority_queue<std::pair<float, int>> hp;
          do
          {
            hp = std::move(std::priority_queue<std::pair<float, int>>());
            auto cd = graph_->search(query[i], actual_k, std::max(ef, actual_k));
            while (cd.size())
            {
              if (Interval::conquer(attrs[i], ft_[cd.top().second]))
              {
                hp.emplace(cd.top());
                if (hp.size() > k)
                  hp.pop();
              }
              cd.pop();
            }
            actual_k *= 2;
            iteration++;
          } while (hp.size() < k && iteration < max_iteration);
          for (size_t j = 0; hp.size(); j++)
          {
            std::tie(dis[i * k + j], knn[i * k + j]) = hp.top();
            hp.pop();
          }
        }
        return {dis, knn};
      }

      size_t index_size() const override
      {
        if (graph_)
          return graph_->index_size();
        else
          return 0;
      }

      // void save(const std::string &fname) const override
      // {
      //   if (graph_)
      //     graph_->save(fname);
      // }

      size_t get_comparison_and_clear()  override
      {
        if (graph_)
          return graph_->get_comparison_and_clear();
        else
          return 0;
      }

      size_t get_num_threads() const  override
      {
        return num_threads_;
      }

      void set_num_threads(size_t num_threads)  override
      {
        num_threads_ = num_threads;
      }
    };

  }

}
