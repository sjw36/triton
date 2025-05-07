add_triton_library(TritonGENTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    ConvertDotOp/ConvertDotOpToUkernelOps.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonGENTransformsPassIncGen
)

if (dnnl_FOUND)
  target_link_libraries(TritonGENTransforms PRIVATE DNNL::dnnl)
endif()
