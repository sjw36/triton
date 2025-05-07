add_triton_library(TritonToTritonGEN
    ConvertAtomicOps.cpp
    ConvertControlFlowOps.cpp
    ConvertDebugOps.cpp
    ConvertDotOp.cpp
    ConvertElementwiseOps.cpp
    ConvertElemManipOps.cpp
    ConvertHistogramOp.cpp
    ScalarizeInterface.cpp
    ScalarizeUsingForOps.cpp
    ConvertMemoryOps.cpp
    ConvertPtrOps.cpp
    ConvertReductionOp.cpp
    ConvertScanOp.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToTritonGENPassIncGen
    ScalarizeInterfaceIncGen
    MLIRDialectUtils

    LINK_LIBS PUBLIC
    TritonGENIR
    MLIRVectorDialect
)
