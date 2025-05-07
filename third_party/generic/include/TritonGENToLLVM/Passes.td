#ifndef TRITONGEN_CONVERSION_PASSES
#define TRITONGEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FuncOpToLLVM : Pass<"triton-gen-func-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert FuncOp to LLVM for GEN.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createFuncOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def MemoryOpToLLVM : Pass<"triton-gen-memory-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory operations to LLVM for GEN.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createMemoryOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def GetProgramIdOpToLLVM : Pass<"triton-gen-get-program-id-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton GetProgramId to LLVM for GEN.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createGetProgramIdOpToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect"];
}

def LowerMultiReduction : Pass<"triton-gen-lower-multi-reduction", "mlir::triton::FuncOp"> {
    let summary = "Convert multi-dimensional reductions.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createLowerMultiReductionPass()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect"];
}

def AtomicOpsToLLVM : Pass<"triton-gen-atomic-ops-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton atomic operations to LLVM.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createAtomicOpsToLLVMPass()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect"];
}

def DebugOpsToLLVM : Pass<"triton-gen-debug-ops-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton debug operations (prints and asserts) to LLVM.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createDebugOpsToLLVMPass()";

    let dependentDialects = ["mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect"];
}

def UkernelOpsToOneDNNLLVM : Pass<"triton-gen-ukernels-to-onednn-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ukernel operations to OneDNN LLVM runtime calls.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createUkernelOpsToOneDNNLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect"];
}

def UkernelOpsToXSMMLLVM : Pass<"triton-gen-ukernels-to-xsmm-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ukernel operations to XSMM LLVM runtime calls.";
    let description = [{}];
    let constructor = "mlir::triton::gen::createUkernelOpsToXSMMLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect"];
}

def MathToVecLib : Pass<"triton-gen-math-to-vec-lib", "mlir::ModuleOp"> {
    let summary = "Convert vector math operations to vector libm or sleef calls.";
    let description = [{
    }];
    let constructor = "mlir::triton::gen::createMathToVecLibPass()";

    let options = [
        Option<"lib", "lib",
               "mlir::triton::gen::VecLib", /*default*/"mlir::triton::gen::VecLib::Sleef",
               "Library to use for vector math (libsleef or libmvec).",
               [{::llvm::cl::values(
               clEnumValN(mlir::triton::gen::VecLib::Sleef, "sleef",
                "Use Sleef as mm lib"),
               clEnumValN(mlir::triton::gen::VecLib::Mvec, "mvec",
                "Use Mvec as mm lib")
              )}]>,
        ListOption<"gen_features", "gen_features", "std::string",
             "A list of available GEN features to choose proper vector functions">,
    ];

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::gen::TritonGENDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::func::FuncDialect",
                             "mlir::LLVM::LLVMDialect"];
}

#endif
