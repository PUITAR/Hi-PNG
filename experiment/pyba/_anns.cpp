#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utils/timer.hpp>
#include "_data.hpp"
#include "_pf_graph.hpp"
#include "_quad_tree.hpp"

namespace py = pybind11;

namespace pyba
{

  PYBIND11_MODULE(anns, m)
  {
    m.doc() = "Python APIs for ANN search and Interval Research";

    /******************************* DataSet Interfaces *******************************/
    py::class_<DataSet>(m, "DataSet")
        .def(py::init<const std::string &>(), "create an empty dataset",
             py::arg("dataset file path"))
        .def("load", &DataSet::load, "load a dataset from a file",
             py::arg("vector dataset filename"), py::arg("use '.bin' format"))
        .def("size", &DataSet::size, "get the size of the dataset")
        .def("dimension", &DataSet::dimension, "get the dimension of the dataset");

    py::class_<IntervalSet>(m, "IntervalSet")
        .def(py::init<const std::string &>(), "create an empty interval set",
             py::arg("dataset file path"))
        .def("load", &IntervalSet::load, "load an interval set from a file",
             py::arg("interval set filename"))
        .def("size", &IntervalSet::size, "get the size of the interval set");

    py::class_<GroundTruth>(m, "GroundTruth")
        .def(py::init<const std::string &>(), "create an empty ground truth",
             py::arg("dataset file path"))
        .def("load", &GroundTruth::load, "load a ground truth from a file",
             py::arg("ground truth filename"), py::arg("use '.bin' format"))
        .def("recall", &GroundTruth::recall, "compute recall@k",
             py::arg("k"), py::arg("k-ANNS results"))
        .def("size", &GroundTruth::size, "get the size of the ground truth")
        .def("dimension", &GroundTruth::dimension, "get the max number of records of the ground truth");

    /******************************** Utils Interfaces ********************************/
    py::class_<anns::utils::Timer>(m, "Timer")
        .def(py::init<>(), "create a timer")
        .def("start", &anns::utils::Timer::start, "start the timer")
        .def("stop", &anns::utils::Timer::stop, "stop the timer")
        .def("reset", &anns::utils::Timer::reset, "reset the timer")
        .def("get", &anns::utils::Timer::get, "get the elapsed time");

    /************************** PosterFilterGraph Interfaces **************************/
    py::class_<PostFilterGraph>(m, "PostFilterGraph")
        .def(py::init<const std::string &, const std::vector<float> &, const std::string &>(), "create a PostFilterGraph from given parameters",
             py::arg("underlying graph type"), py::arg("responding graph parameters"), py::arg("vector space metric"))
        .def("build", &PostFilterGraph::build, "build the graph from given data",
             py::arg("base dataset"), py::arg("interval set"), py::arg("number of threads"))
        .def("search", &PostFilterGraph::search, "search k-ANNS with interval filtering",
             py::arg("query vector"), py::arg("interval set of the query"), py::arg("k"), py::arg("query parameters"), py::arg("number of threads"))
        .def("index_size", &PostFilterGraph::index_size, "get the bytes of the index")
        .def("get_comparison_and_clear", &PostFilterGraph::get_comparison_and_clear, "get the number of comparisons and clear the counter");

    /****************************** HiPNG Interfaces ******************************/
    py::class_<HiPNG>(m, "HiPNG")
        .def(py::init<const std::string &, const std::vector<float> &, const std::string &>(), "create a HiPNG from given parameters",
             py::arg("underlying graph type"), py::arg("build parameters {leaf size, underlying graph parameters...}"), py::arg("vector space metric"))
        .def("build", &HiPNG::build, "build the graph from given data",
             py::arg("base dataset"), py::arg("interval set"), py::arg("number of threads"))
        .def("search", &HiPNG::search, "search k-ANNS with interval filtering",
             py::arg("query vector"), py::arg("interval set of the query"), py::arg("k"), py::arg("query parameters"), py::arg("number of threads"))
        .def("index_size", &HiPNG::index_size, "get the bytes of the index")
        .def("get_comparison_and_clear", &HiPNG::get_comparison_and_clear, "get the number of comparisons and clear the counter");
  }

}