#ifndef TRITONGEN_CONVERSION_TRITONGENTOLLVM_PASSES_H
#define TRITONGEN_CONVERSION_TRITONGENTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace gen {

enum class VecLib {
  Mvec,
  Sleef,
};

#define GEN_PASS_DECL
#include "generic/include/TritonGENToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createFuncOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createGetProgramIdOpToLLVMPass();
std::unique_ptr<OperationPass<triton::FuncOp>> createLowerMultiReductionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtomicOpsToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createDebugOpsToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToOneDNNLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToXSMMLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createMathToVecLibPass(VecLib lib = VecLib::Sleef,
                       std::set<std::string> gen_features = {});

#define GEN_PASS_REGISTRATION
#include "generic/include/TritonGENToLLVM/Passes.h.inc"

} // namespace gen
} // namespace triton

} // namespace mlir

#endif
