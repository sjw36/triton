add_triton_library(TritonGENToLLVM
    AtomicOpsToLLVM.cpp
    DebugOpsToLLVM.cpp
    UkernelOpsToOneDNNLLVM.cpp
    UkernelOpsToXSMMLLVM.cpp
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    LowerMultiReduction.cpp
    MathToVecLib.cpp
    MemoryOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp

    DEPENDS
    TritonGENToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
