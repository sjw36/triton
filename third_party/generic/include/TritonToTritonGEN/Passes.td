#ifndef TRITONTOTRITONGEN_CONVERSION_PASSES
#define TRITONTOTRITONGEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemoryOps : Pass<"triton-gen-convert-memory-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory ops.";
    let description = [{

    }];

    let options = [
        Option<"useGatherScatter", "use-gather-scatter",
               "bool", /*default*/"false",
               "Use Gather or Scatter to lower memory ops.">,
    ];

    let constructor = "mlir::triton::gen::createConvertMemoryOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertElementwiseOps : Pass<"triton-gen-convert-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Convert elementwise ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertElementwiseOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertElemManipOps : Pass<"triton-gen-convert-elem-manip-ops", "mlir::ModuleOp"> {
    let summary = "Convert elements manipulation ops (transpose, shuffle, etc.).";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertElemManipOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertPtrOps : Pass<"triton-gen-convert-ptr-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton ops related to pointer arithmetics.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertPtrOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertDotOp : Pass<"triton-gen-convert-dot-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertDotOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertControlFlowOps : Pass<"triton-gen-convert-control-flow-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertControlFlowOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertHistogramOp : Pass<"triton-gen-convert-histogram-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton HistogramOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertHistogramOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect"];
}

def ConvertReductionOp : Pass<"triton-gen-convert-reduction", "mlir::ModuleOp"> {
    let summary = "Convert Triton ReduceOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertReductionOp()";

    let options = [
        Option<"useMultiDimReductionOp", "use-multidim-reduction-op",
               "bool", /*default*/"false",
               "Use vector::MultiDimReductionOp and its default lowering when possible.">,

        Option<"useReductionOp", "use-reduction-op",
               "bool", /*default*/"false",
                "Use vector::ReductionOp and its default lowering when possible.">,
    ];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertScanOp : Pass<"triton-gen-convert-scan", "mlir::ModuleOp"> {
    let summary = "Convert Triton ScanOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertScanOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertAtomicOps : Pass<"triton-gen-convert-atomic-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton atomic operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertAtomicOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ConvertDebugOps : Pass<"triton-gen-convert-debug-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton debug operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::gen::createConvertDebugOps()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}

def ScalarizeUsingForOp : Pass<"triton-gen-scalarize", "mlir::ModuleOp"> {
    let summary = "Insert Loops for ops, that are not vectorizable";
    let description = [{
        This pass is used to reduce compile time by generating loops for
        operations that cannot be handled as vectors, and simply increases
        the amount of IR without any further optimization.
    }];

    let options = [
        Option<"skipGatherScatter", "skip-gather-scatter",
               "bool", /*default*/"false",
               "Skip scalarizing gather/scatter ops.">,
    ];

    let constructor = "mlir::triton::gen::createScalarizeUsingForOpPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gen::TritonGENDialect"];
}


#endif
