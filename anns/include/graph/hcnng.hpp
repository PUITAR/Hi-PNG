#pragma once

#include <distance.hpp>
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <memory>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <mutex>
#include <omp.h>
#include <algorithm>
#include <memory>

#include <utils/binary.hpp>

#include <base.hpp>

namespace anns
{

  namespace graph
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class HCNNG : public Base<data_t>
    {
      class DisjointSet
      {
      public:
        DisjointSet(size_t size)
        {
          parent.resize(size);
          for (size_t i = 0; i < size; ++i)
            parent[i] = i;
        }

        int find(int x)
        {
          if (parent[x] != x)
            parent[x] = find(parent[x]);
          return parent[x];
        }

        void union_set(int x, int y)
        {
          parent[find(x)] = find(y);
        }

        std::vector<int> parent;
      };

      struct Edge
      {
        int src;
        int dst;
        float weight;

        bool operator<(const Edge &other) const
        {
          return this->weight < other.weight;
        }
      };

    protected:
      size_t T_{0};
      size_t Ls_{0};
      size_t s_{0};
      std::vector<std::vector<int>> adj_memory_;
      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;

    public:
      __USE_BASE__

      HCNNG(size_t T, size_t Ls, size_t s)  : Ls_(Ls), s_(s), T_(T) {}

      HCNNG(const DataSet<data_t> &base, const std::string &filename) 
      {
        base_ = base;
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open())
        {
          throw std::runtime_error("Cannot open file for reading");
        }
        in.read(reinterpret_cast<char *>(&T_), sizeof(T_));
        in.read(reinterpret_cast<char *>(&Ls_), sizeof(Ls_));
        in.read(reinterpret_cast<char *>(&s_), sizeof(s_));
        adj_memory_.resize(base_.num_);
        for (auto &adj : adj_memory_)
        {
          size_t n;
          in.read(reinterpret_cast<char *>(&n), sizeof(n));
          adj.resize(n);
          in.read(reinterpret_cast<char *>(adj.data()), n * sizeof(int));
        }
        link_list_locks_.resize(base_.num_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
      }

      void save(const std::string &filename) const  override
      {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open())
        {
          throw std::runtime_error("Cannot open file for writing");
        }
        out.write(reinterpret_cast<const char *>(&T_), sizeof(T_));
        out.write(reinterpret_cast<const char *>(&Ls_), sizeof(Ls_));
        out.write(reinterpret_cast<const char *>(&s_), sizeof(s_));
        for (const auto &neighbors : adj_memory_)
        {
          size_t n = neighbors.size();
          const char *buffer = reinterpret_cast<const char *>(neighbors.data());
          out.write(reinterpret_cast<const char *>(&n), sizeof(n));
          out.write(buffer, neighbors.size() * sizeof(int));
        }
      }

      void build(const DataSet<data_t> &base)  override
      {
        base_ = base;
        adj_memory_.resize(base_.num_);
        link_list_locks_.resize(base_.num_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
        // std::cout << 139 << std::endl;
        // initialize graph data
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads_)
        for (int id = 0; id < base_.num_; id++)
        {
          adj_memory_[id].reserve(s_ * T_);
        }
        // std::cout << 146 << std::endl;
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < T_; i++)
        {
          auto idx_points = std::make_unique<std::vector<int>>(base_.num_);
          for (size_t j = 0; j < base_.num_; j++)
          {
            idx_points->at(j) = j;
          }
          create_clusters(*idx_points, 0, base_.num_ - 1, Ls_, s_);
        }
        // std::cout << 157 << std::endl;
      }

      res_t search(const DataSet<data_t> &query, size_t k, size_t ef) override
      {
        assert(query.dim_ == base_.dim_);
        knn_t knn(query.num_ * k, MAGIC_ID);
        dis_t dis(query.num_ * k, MAGIC_DIST);
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
        for (size_t i = 0; i < query.num_; i++)
        {
          auto r = search(query[i], k, ef);
          for (size_t j = 0; r.size(); j++)
          {
            std::tie(dis[i * k + j], knn[i * k + j]) = r.top();
            r.pop();
          }
        }
        return {dis, knn}; // auto RVO
      }

      size_t index_size() const  override
      {
        size_t sz = 0;
        for (int id = 0; id < base_.num_; id++)
        { // adj list
          sz += adj_memory_[id].size() * sizeof(int);
        }
        return sz;
      }

      /// @brief search the base layer (User call this funtion to do single query).
      /// @param data_point
      /// @param k
      /// @param ef
      /// @return a maxheap containing the knn results
      std::priority_queue<std::pair<float, int>> search(const data_t *data_point, size_t k, size_t ef) override
      {
        std::vector<bool> mass_visited(base_.num_, false);
        std::priority_queue<std::pair<float, int>> top_candidates;
        std::priority_queue<std::pair<float, int>> candidate_set;
        size_t comparison = 0;
        int ep = rand() % base_.num_;
        float dist = distance(data_point, base_[ep], base_.dim_);
        comparison++;
        top_candidates.emplace(dist, ep); // max heap
        candidate_set.emplace(-dist, ep); // min heap
        mass_visited[ep] = true;
        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();
          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;
          candidate_set.pop();
          int curr_node_id = curr_el_pair.second;
          std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
          const auto &neighbors = adj_memory_[curr_node_id];
          for (int neighbor_id : neighbors)
          {
            if (!mass_visited[neighbor_id])
            {
              mass_visited[neighbor_id] = true;
              float dd = distance(data_point, base_[neighbor_id], base_.dim_);
              comparison++;
              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dd || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dd, neighbor_id);
                top_candidates.emplace(dd, neighbor_id);
                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();
                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }
        comparison_.fetch_add(comparison);
        return top_candidates;
      }

    protected:
      std::vector<std::vector<Edge>> create_exact_mst(const std::vector<int> &idx_points, size_t left, size_t right, size_t s)
      {
        size_t num_points = right - left + 1;
        std::vector<Edge> full_graph;
        std::vector<std::vector<Edge>> mst(num_points);
        full_graph.reserve(num_points * (num_points - 1));

        // pick up all edges into full_graph
        for (size_t i = 0; i < num_points; i++)
        {
          for (size_t j = 0; j < num_points; j++)
          {
            if (i != j)
            {
              full_graph.emplace_back(
                  Edge{i, j, distance(base_[idx_points[left + i]], base_[idx_points[left + j]], base_.dim_)});
            }
          }
        }

        // Kruskal algorithm
        std::sort(full_graph.begin(), full_graph.end());
        DisjointSet ds(num_points);
        for (const auto &e : full_graph)
        {
          int src = e.src;
          int dst = e.dst;
          float weight = e.weight;
          if (ds.find(src) != ds.find(dst) && mst[src].size() < s && mst[dst].size() < s)
          {
            mst[src].emplace_back(e);
            mst[dst].emplace_back(Edge{dst, src, weight});
            ds.union_set(src, dst);
          }
        }

        return mst;
      }

      void create_clusters(std::vector<int> &idx_points, size_t left, size_t right, size_t Ls, size_t s)
      {
        // std::cout << 286 << std::endl;
        size_t num_points = right - left + 1;
        if (num_points <= Ls)
        {
          auto mst = create_exact_mst(idx_points, left, right, s);
          // Add edges to graph
          for (size_t i = 0; i < num_points; i++)
          {
            for (size_t j = 0; j < mst[i].size(); j++)
            {
              std::unique_lock<std::mutex> lock0(*link_list_locks_[idx_points[left + i]]);

              bool is_neighbor = false;
              auto &neigh0 = adj_memory_[idx_points[left + i]];

              for (const auto &nid0 : neigh0)
              {
                if (nid0 == idx_points[left + mst[i][j].dst])
                {
                  is_neighbor = true;
                  break;
                }
              }
              if (!is_neighbor)
              {
                neigh0.emplace_back(idx_points[left + mst[i][j].dst]);
              }
            }
          }
        }
        else
        {
          auto rand_int = [](size_t Min, size_t Max)
          {
            size_t sz = Max - Min + 1;
            return Min + (std::rand() % sz);
          };

          size_t x = rand_int(left, right);
          size_t y = -1;
          do
          {
            y = rand_int(left, right);
          } while (x == y);
          const data_t *vec_idx_left_p_x = base_[idx_points[x]];
          const data_t *vec_idx_left_p_y = base_[idx_points[y]];
          std::vector<int> ids_x_set, ids_y_set;
          ids_x_set.reserve(num_points);
          ids_y_set.reserve(num_points);

          for (size_t i = 0; i < num_points; i++)
          {
            const data_t *vec_idx_left_p_i = base_[idx_points[left + i]];

            float dist_x = distance(vec_idx_left_p_x, vec_idx_left_p_i, base_.dim_);
            float dist_y = distance(vec_idx_left_p_y, vec_idx_left_p_i, base_.dim_);

            if (dist_x < dist_y)
            {
              ids_x_set.emplace_back(idx_points[left + i]);
            }
            else
            {
              ids_y_set.emplace_back(idx_points[left + i]);
            }
          }

          assert(ids_x_set.size() + ids_y_set.size() == num_points);
          // reorder idx_points
          size_t i = 0, j = 0;
          while (i < ids_x_set.size())
          {
            idx_points[left + i] = ids_x_set[i];
            i++;
          }
          while (j < ids_y_set.size())
          {
            idx_points[left + i] = ids_y_set[j];
            j++;
            i++;
          }
          // std::cout << 376 << std::endl;
          create_clusters(idx_points, left, left + ids_x_set.size() - 1, Ls, s);
          create_clusters(idx_points, left + ids_x_set.size(), right, Ls, s);
        }
        // std::cout << 375 << std::endl;
      }

      void prune_neighbors(size_t max_neigh)
      {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (int id = 0; id < base_.num_; id++)
        {
          const data_t *vec_curid = base_[id];

          auto &neigh = adj_memory_[id];

          size_t new_size = std::min(neigh.size(), max_neigh);

          if (new_size == neigh.size())
            continue;

          std::vector<std::pair<float, int>> score;
          score.reserve(neigh.size());
          for (const auto &nid : neigh)
          {
            score.emplace_back(distance(base_[nid], vec_curid, base_.dim_), nid);
          }

          std::sort(score.begin(), score.end());
          score.resize(new_size);
          score.shrink_to_fit();
          neigh.resize(new_size);
          neigh.shrink_to_fit();

          for (size_t i = 0; i < new_size; i++)
          {
            neigh[i] = score[i].second;
          }
        }
      }
    };

  }

}
