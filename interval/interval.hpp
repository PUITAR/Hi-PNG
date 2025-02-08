#pragma once

#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>
#include <utils/binary.hpp>
#include <functional>
#include <anns.hpp>
#include <limits>

namespace anns
{
  namespace interval
  {

    class Interval : public std::pair<float, float>
    {
    public:
      Interval() = default;

      Interval(const Interval &other) = default;

      Interval(std::initializer_list<float> obj)
      {
        assert(obj.size() == 2 && "Initializer list must have exactly 2 elements");
        auto it = obj.begin();
        this->first = *it;
        this->second = *(++it);
      }

      inline static bool conquer(const Interval &a, const Interval &b) 
      {
        return a.first <= b.first && a.second >= b.second;
      }

      inline static Interval intersect(const Interval &a, const Interval &b) 
      {
        return {std::max(a.first, b.first), std::min(a.second, b.second)};
      }

      inline static float length(const Interval &a)
      {
        return a.second - a.first;
      }

    };

    const Interval INT_HULL{-std::numeric_limits<float>::max(), +std::numeric_limits<float>::max()};

    // using conquer_t = bool (*)(const Interval &, const Interval &);

    struct IntervalSet
    {
      const Interval *data_;
      size_t num_{0};
      std::function<size_t(size_t)> hash_{DEFAULT_HASH};

      inline const Interval &access(size_t id) const
      {
        return data_[hash_(id)];
      }

      inline const Interval &operator[](size_t id) const
      {
        return data_[hash_(id)];
      }
    };

    struct IntervalSetWrapper : public IntervalSet
    {
      using IntervalSet::data_;
      using IntervalSet::hash_;
      using IntervalSet::num_;
      std::vector<Interval> base_;

      IntervalSetWrapper(const std::string &fname)
      {
        load(fname);
      }

      void load(const std::string &fname)
      {
        using namespace utils;
        std::vector<float> buf;
        auto [n, d] = load_from_file(buf, fname);
        assert(d == 2 && "An interval must have exactly 2 elements [start, end].");
        num_ = n;
        base_.resize(num_);
        for (size_t i = 0; i < n; i++)
        {
          base_[i].first = buf[i * 2];
          base_[i].second = buf[i * 2 + 1];
        }
        data_ = base_.data();
      }
    };

    template <typename data_t>
    class IntervalBaseIndex
    {
    public:
      virtual void build(const DataSet<data_t> &base, const IntervalSet &attrs) = 0;
      virtual res_t search(const DataSet<data_t> &query, const IntervalSet &attrs,
                           size_t k, const std::vector<float> &params) = 0;
      virtual size_t get_num_threads() const = 0;
      virtual void set_num_threads(size_t num_threads) = 0;
      virtual size_t get_comparison_and_clear() = 0;
      virtual size_t index_size() const = 0;
      // virtual void save(const std::string &fname) const = 0;

    };

  }

}