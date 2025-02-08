set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

foreach(_target
  groundtruth # you should never delete this item
  test_postfilter_hnsw
  test_postfilter_hcnng
  test_quad_tree
)
  add_executable(${_target} "${_target}.cpp")
  target_compile_options(${_target} PUBLIC 
    -Ofast 
    -march=native 
    -mtune=native 
    -lrt
    -flto=auto
  )
  target_link_libraries(${_target} PUBLIC
    OpenMP::OpenMP_CXX
  )
endforeach()