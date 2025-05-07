# Find OneDNN ukernel library
find_package(dnnl CONFIG)
if (dnnl_FOUND)
  message(STATUS "Found OneDNN/DNNL")
  add_compile_definitions(ONEDNN_AVAILABLE)
  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
  # currently used only in triton_gen.cc and in ConvertDotToOneDNN
  include_directories(${dnnl_include})
else ()
  message(STATUS "Could NOT find OneDNN/DNNL")
endif()

# Find XSMM ukernel library
find_library(LIBXSMM xsmm
  PATHS ENV XSMM_LIBRARY_DIRS
)
find_path(LIBXSMM_INCLUDE_DIR
  NAMES libxsmm.h
  PATHS ENV XSMM_INCLUDE_DIRS
)
if (LIBXSMM AND LIBXSMM_INCLUDE_DIR)
  set(xsmm_FOUND True)
  message(STATUS "Found XSMM: " ${LIBXSMM})
  add_compile_definitions(XSMM_AVAILABLE)
  include_directories(${LIBXSMM_INCLUDE_DIR})
else ()
  message(STATUS "Could NOT find XSMM")
endif()

# Build Triton-GEN plugin
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonGEN ${CMAKE_CURRENT_SOURCE_DIR}/triton_gen.cc LINK_LIBS TritonGENToLLVM TritonGENTransforms)
  target_link_libraries(TritonGEN PUBLIC MLIRVectorToSCF MLIRAffineToStandard MLIRMathToLibm MLIRAMXToLLVMIRTranslation MLIRMemRefTransforms MLIRReconcileUnrealizedCasts PRIVATE Python3::Module pybind11::headers)
endif()

# Configure and build Triton-GEN runtime
set(TRITON_GEN_RUNTIME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/runtime/gen_runtime.cpp)
set(TRITON_GEN_RUNTIME_LIBS LLVMSupport)
if (dnnl_FOUND)
  set(TRITON_GEN_RUNTIME_SOURCES ${TRITON_GEN_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_onednn.cpp)
  set(TRITON_GEN_RUNTIME_LIBS ${TRITON_GEN_RUNTIME_LIBS} DNNL::dnnl)
endif()
if(xsmm_FOUND)
  set(TRITON_GEN_RUNTIME_SOURCES ${TRITON_GEN_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_xsmm.cpp)
  set(TRITON_GEN_RUNTIME_LIBS ${TRITON_GEN_RUNTIME_LIBS} ${LIBXSMM})
endif()

add_library(TritonGENRuntime SHARED ${TRITON_GEN_RUNTIME_SOURCES})
target_link_libraries(TritonGENRuntime PRIVATE ${TRITON_GEN_RUNTIME_LIBS})

