#pragma once

#include <anns.hpp>
#include <interval.hpp>
#include <utils/recall.hpp>

#include <pybind11/pybind11.h>

using namespace anns;

namespace pyba
{
  class DataSet: public anns::DataSetWrapper<float>
  {
  public:
    size_t size() const
    {
      return num_;
    }

    size_t dimension() const
    {
      return dim_;
    }
  };

  class IntervalSet: public interval::IntervalSetWrapper
  {
  public:
    size_t size() const
    {
      return num_;
    }
  };

  class GroundTruth: public anns::utils::GroundTruth
  {
  public:
    size_t size() const
    {
      return num_;
    }

    size_t dimension() const
    {
      return dim_;
    }

    double recall(size_t k, const pybind11::list &knn)
    {
      std::vector<int> temp;
      for (auto id: knn)
      {
        temp.emplace_back(id.cast<int>());
      }
      return utils::recall(k, dim_, base_, temp);
    }

  };
  
}