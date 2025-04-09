#pragma once

#include <pf_graph.hpp>
#include <anns.hpp>
#include <utils/timer.hpp>
#include <set>

namespace anns
{

  namespace interval
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class HiPNG : public IntervalBaseIndex<data_t>
    {
#define PFG interval::PostFilterGraph<data_t, distance>
    protected:
      struct Node
      {
        int c0_{MAGIC_ID};                    // child 0
        int c1_{MAGIC_ID};                    // child 1
        int c2_{MAGIC_ID};                    // child 2
        int c3_{MAGIC_ID};                    // child 3
        Interval pit_;                        // positive interval
        Interval nit_;                        // negative interval
        std::vector<int> act_;                // active vector
        std::shared_ptr<PFG> index_{nullptr}; // post-filter graph
      };

      std::vector<Node> nodes_;         // tree nodes
      size_t ls_{0};                    // leaf size
      std::string name_;                // name of under-lying graph
      std::vector<float> index_params_; // parameters of under-lying graph
      DataSet<data_t> base_;            // base dataset
      IntervalSet attrs_;               // attributes
      size_t num_threads_{1};           // number of threads

    public:
      HiPNG(const std::string &name, const std::vector<float> &params) : name_(name)
      {
        ls_ = params[0];
        // max_it_construction_ = params[1];
        index_params_ = std::vector<float>(params.begin() + 1, params.end());
      }

      void build(const DataSet<data_t> &base, const IntervalSet &attrs) override
      {
        assert(base.num_ == attrs.num_ && "base and its must have the same number of intervals");
        base_ = base;
        attrs_ = attrs;
        // setup the root node
        std::vector<int> act;
        act.reserve(base_.num_);
        float mi = +std::numeric_limits<float>::max(),
              ma = -std::numeric_limits<float>::max();
        for (int id = 0; id < base_.num_; id++)
        {
          act.emplace_back(id);
          mi = std::min(mi, attrs_[id].first);
          ma = std::max(ma, attrs_[id].second);
        }
        nodes_.emplace_back(Node());
        nodes_[0].pit_ = {mi, ma};
        nodes_[0].nit_ = {ma, mi};
        nodes_[0].act_ = std::move(act);
        // build nodes
        std::queue<int> que;
        que.push(0);
        while (que.size())
        {
          int nid = que.front();
          que.pop();
          // build graph
          auto hasher = [nid, this](int id) -> int
          {
            return this->nodes_[nid].act_[id];
          };
          Node &n = nodes_[nid];
          n.index_ = std::make_shared<PFG>(name_, index_params_);
          n.index_->set_num_threads(num_threads_);
          n.index_->build(
              {base_.data_, n.act_.size(), base_.dim_, hasher},
              {attrs_.data_, n.act_.size(), hasher});
          n.index_->set_num_threads(1);
          // split node
          if (ls_ < n.act_.size())
          {
            std::vector<int> act0, act1, act2, act3;
            Node n0, n1, n2, n3;
            // Interval e = evenly_select(n);
            Interval e = balanced_select(n);
            n0.pit_ = n.pit_;
            n0.nit_ = {e.first, e.second};
            n1.pit_ = {n0.nit_.first, n0.pit_.second};
            n1.nit_ = {n.nit_.first, n0.nit_.second};
            n2.pit_ = {n0.pit_.first, n0.nit_.second};
            n2.nit_ = {n0.nit_.first, n.nit_.second};
            n3.pit_ = n0.nit_;
            n3.nit_ = n.nit_;
            for (int id : n.act_)
            {
              const auto &a = attrs_[id];
              if (a.first < n0.nit_.first && a.second > n0.nit_.second)
                act0.emplace_back(id);
              else if (a.first < n1.nit_.first && a.second > n1.nit_.second)
                act1.emplace_back(id);
              else if (a.first < n2.nit_.first && a.second > n2.nit_.second)
                act2.emplace_back(id);
              else // a.first < n3.nit_.first && a.second > n3.nit_.second
                act3.emplace_back(id);
            }
            /* Reference n may colipse after emplace_back, so use nid instead of referenc n */
            if (act0.size())
            {
              nodes_.emplace_back(n0);
              nodes_.back().act_ = std::move(act0);
              que.push(nodes_[nid].c0_ = nodes_.size() - 1);
            }
            if (act1.size())
            {
              nodes_.emplace_back(n1);
              nodes_.back().act_ = std::move(act1);
              que.push(nodes_[nid].c1_ = nodes_.size() - 1);
            }
            if (act2.size())
            {
              nodes_.emplace_back(n2);
              nodes_.back().act_ = std::move(act2);
              que.push(nodes_[nid].c2_ = nodes_.size() - 1);
            }
            if (act3.size())
            {
              nodes_.emplace_back(n3);
              nodes_.back().act_ = std::move(act3);
              que.push(nodes_[nid].c3_ = nodes_.size() - 1);
            }
          }
        }
      }

      res_t search(const DataSet<data_t> &query, const IntervalSet &attrs,
                   size_t k, const std::vector<float> &params) override
      {
        assert(query.dim_ == base_.dim_ && query.num_ == attrs.num_ &&
               "query and attrs must have the same dimension and number of intervals");
        assert(params.size() == 3 && "params must have 3 elements");
        assert(params[0] >= k && "k must be less than or equal to ef");
        size_t ef = params[0], max_iteration = params[1];
        float sim = params[2];
        knn_t knn(query.num_ * k, MAGIC_ID);
        dis_t dis(query.num_ * k, MAGIC_DIST);
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic, 1)
        for (size_t i = 0; i < query.num_; i++)
        {
          const data_t *q = query[i];
          const Interval &a = attrs[i];
          std::queue<int> que;
          std::priority_queue<std::pair<float, int>> candidates;
          // float ss = Interval::length(a) <= 0.1 ? sim: 0;
          auto search_node = [&](const Node &n, bool pf)
          {
            knn_t knn_i;
            dis_t dis_i;
            if (pf)
            {
              std::tie(dis_i, knn_i) = n.index_->search({q, 1, query.dim_}, {&a, 1}, k, {ef, max_iteration});
            }
            else
            {
              std::tie(dis_i, knn_i) = n.index_->search({q, 1, query.dim_}, k, ef);
            }
            for (size_t j = 0; j < knn_i.size(); j++)
            {
              if (knn_i[j] != MAGIC_ID)
              {
                candidates.emplace(dis_i[j], n.act_[knn_i[j]]);
                if (candidates.size() > k)
                  candidates.pop();
              }
            }
          };
          // search the tree
          que.push(0);
          while (que.size())
          {
            const Node &n = nodes_[que.front()];
            que.pop();
            if (conquer(a, n))
              search_node(n, 0);
            else if ((n.c0_ == MAGIC_ID && n.c1_ == MAGIC_ID && n.c2_ == MAGIC_ID && n.c3_ == MAGIC_ID) || similarity(a, n) > sim)
              search_node(n, 1);
            else // not leaf and not cover
            {
              if (n.c0_ != MAGIC_ID && similarity(a, nodes_[n.c0_]) > 0)
                que.push(n.c0_);
              if (n.c1_ != MAGIC_ID && similarity(a, nodes_[n.c1_]) > 0)
                que.push(n.c1_);
              if (n.c2_ != MAGIC_ID && similarity(a, nodes_[n.c2_]) > 0)
                que.push(n.c2_);
              if (n.c3_ != MAGIC_ID && similarity(a, nodes_[n.c3_]) > 0)
                que.push(n.c3_);
            }
          }
          // collect results
          for (size_t j = 0; candidates.size(); j++)
          {
            std::tie(dis[i * k + j], knn[i * k + j]) = candidates.top();
            candidates.pop();
          }
        }
        return {dis, knn};
      }

      size_t get_num_threads() const override
      {
        return num_threads_;
      }

      void set_num_threads(size_t num_threads) override
      {
        num_threads_ = num_threads;
      }

      size_t get_comparison_and_clear() override
      {
        size_t cmp = 0;
        for (auto &n : nodes_)
        {
          cmp += n.index_->get_comparison_and_clear();
        }
        return cmp;
      }

      size_t index_size() const override
      {
        size_t sz = 0;
        for (const auto &n : nodes_)
        {
          sz += n.index_->index_size() + sizeof(Node) + n.act_.size() * sizeof(int);
        }
        return sz;
      }

    protected:
      static inline bool conquer(const Interval &query, const Node &node)
      {
        return Interval::conquer(query, node.pit_);
      }

      static inline float similarity(const Interval &query, const Node &node)
      {
        if (Interval::conquer(query, node.nit_))
        {
          float area0 = (node.nit_.first - node.pit_.first) * (node.pit_.second - node.nit_.second);
          float area1 = (node.nit_.first - std::max(node.pit_.first, query.first)) * (std::min(node.pit_.second, query.second) - node.nit_.second);
          return area1 / area0;
        }
        return 0;
      }

      Interval balanced_select(const Node &node)
      {
        // std::cout << 304 << std::endl;
        size_t nsample = ls_;
        std::vector<Interval> check;
        for (size_t i = 0; i < nsample; i++) 
        {
          check.emplace_back(attrs_[node.act_[rand() % node.act_.size()]]);
        }
        // std::cout << 311 << std::endl;
        std::sort(check.begin(), check.end());
        std::vector<int> upper_left(nsample);
        std::vector<int> upper_right(nsample);
        std::vector<int> lower_left(nsample);
        std::vector<int> lower_right(nsample);
        std::set<float> st;
        // std::cout << 312 << std::endl;
        for (int i = 0; i < nsample; i++)
        {
          auto [_, t] = check[i];
          st.insert(t);
          upper_left[i] = std::distance(std::lower_bound(st.begin(), st.end(), t), st.end());
          lower_left[i] = st.size() - upper_left[i];
        }
        // std::cout << 326 << std::endl;
        st.clear();
        for (int i = nsample - 1; i >= 0; i--)
        {
          auto [_, t] = check[i];
          st.insert(t);
          upper_right[i] = std::distance(std::lower_bound(st.begin(), st.end(), t), st.end());
          lower_right[i] = st.size() - upper_right[i];
        }
        // std::cout << 335 << std::endl;
        int best_score = std::numeric_limits<int>::max();
        Interval best;
        for (size_t i = 0; i < nsample; i++)
        {
          int score = abs(upper_left[i] - upper_right[i]) + abs(upper_left[i] - lower_right[i])  + 
                      abs(lower_left[i] - upper_right[i]) + abs(lower_left[i] - lower_right[i])  + 
                      abs(upper_left[i] - lower_left[i])  + abs(upper_right[i] - lower_right[i]) ;
          if (score < best_score)
          {
            best_score = score;
            best = check[i];
          }
        }
        // std::cout << 349 << std::endl;
        return best;
      }
      
      Interval evenly_select(const Node &n)
      {
        return {(n.pit_.first + n.nit_.first) / 2, (n.pit_.second + n.nit_.second) / 2};
      }

    };

  }

}