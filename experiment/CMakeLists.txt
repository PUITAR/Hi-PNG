find_package(pybind11 REQUIRED)

execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_include_dir
)
string(REPLACE "-I" "" pybind11_include_dir "${pybind11_include_dir}")
string(REPLACE " " ";" pybind11_include_dir "${pybind11_include_dir}")
foreach(path IN LISTS pybind11_include_dir)
  include_directories(${path})
endforeach()

set(experiment_dir ${CMAKE_SOURCE_DIR}/experiment)

file(GLOB_RECURSE module_files ${experiment_dir}/pyba/*.cpp)

add_library(anns_bindings STATIC ${module_files})
target_link_libraries(anns_bindings PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(anns_bindings PUBLIC 
  -Ofast 
  -march=native 
  -mtune=native 
  -lrt
  -fPIC
  -flto=auto
)

pybind11_add_module(anns ${module_files})

target_link_libraries(anns PRIVATE anns_bindings)

install(TARGETS anns
    LIBRARY DESTINATION ${experiment_dir}
)