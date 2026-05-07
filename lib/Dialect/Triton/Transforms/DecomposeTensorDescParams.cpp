#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONDECOMPOSETENSORDESCRIPTORPARAMETERS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool hasATensorDescriptorType(TypeRange types) {
  return llvm::any_of(
      types, [](Type t) { return llvm::isa<triton::TensorDescType>(t); });
}

/**
 * @brief Decomposes `!tt.tensordesc` values at function boundaries into plain
 * SSA operands/results so call sites and signatures no longer carry the
 * aggregate type directly.
 *
 * @details For rank r, one descriptor maps to `1 + 2r` scalars: a pointer to
 * the block element type, `r` dynamic `i32` dim sizes (shape), and `r` dynamic
 * `i64` strides (derived from the descriptor's signless block tensor type).
 * Argument names and attributes (`tdesc.type`, `tdesc.shape`, `tdesc.stride`)
 * are populated via `FuncArgRenamer` so downstream tooling can still relate
 * fields to the original descriptor.
 *
 * The pass runs a partial conversion over the module: `tt.func`, `tt.call`, and
 * `tt.return` are rewritten when their types involve tensor descriptors, and
 * `scf` structural patterns propagate the new signature through regions.
 * `arith`, `scf`, and `tt` remain legal; materializations are enabled so that
 * wherever IR still needs a single `!tt.tensordesc` value, a source
 * materialization rebuilds it with `tt.make_tensor_descriptor` from the
 * decomposed pointer and shape/stride SSA values (including cases where the
 * descriptor is threaded through control flow rather than defined only by
 * `tt.make_tensor_descriptor`).
 *
 * Before conversion, each `tt.func` argument index is tagged with
 * `tt.user_index` for stable identification across the rewrite.
 */

class TritonDecomposeTensorDescriptorParametersPass
    : public impl::TritonDecomposeTensorDescriptorParametersBase<
          TritonDecomposeTensorDescriptorParametersPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();
    auto context = op->getContext();
    OpBuilder builder(context);
    for (auto func : op.getOps<triton::FuncOp>()) {
      for (int i = 0; i < func.getNumArguments(); i++) {
        func.setArgAttr(i, "tt.user_index", builder.getI32IntegerAttr(i));
      }
    }

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                           mlir::triton::TritonDialect>();
    target.addDynamicallyLegalOp<triton::CallOp>([](triton::CallOp op) {
      return !hasATensorDescriptorType(op->getOperandTypes()) &&
             !hasATensorDescriptorType(op->getResultTypes());
    });
    target.addDynamicallyLegalOp<triton::ReturnOp>([](triton::ReturnOp op) {
      return !hasATensorDescriptorType(op->getOperandTypes());
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    TypeConverter converter;
    converter.addConversion([](mlir::Type t) { return t; });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 32));
      out.insert(out.end(), tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      return mlir::success();
    });
    converter.addSourceMaterialization([](OpBuilder &builder,
                                          mlir::triton::TensorDescType type,
                                          ValueRange inputs, Location loc) {
      auto rank = type.getShape().size();
      assert(inputs.size() == 1 + 2 * rank);
      Value base = inputs[0];
      ValueRange shape = inputs.slice(1, rank);
      ValueRange strides = inputs.slice(1 + rank, rank);
      SmallVector<int32_t> blockShape =
          llvm::to_vector(llvm::map_range(type.getShape(), [](int64_t dim) {
            return static_cast<int32_t>(dim);
          }));
      auto desc = triton::MakeTensorDescOp::create(
          builder, loc, base, shape, strides, blockShape,
          type.getElementType().isSignedInteger(),
          triton::PaddingOption::PAD_ZERO);
      return desc.getResult();
    });

    FuncArgRenamer renamer(".");
    renamer.addRenamer([](mlir::triton::TensorDescType type,
                          FuncArgRenamer::SuffixList &out_suffix,
                          FuncArgRenamer::ArgAttrsList &out_attrs) {
      OpBuilder builder(type.getContext());
      auto tensorType = type.getSignlessBlockType();
      int dims = tensorType.getRank();
      out_suffix.push_back("ptr");
      // nv_tma_desc
      out_attrs.push_back(FuncArgRenamer::ArgAttrs{
          builder.getNamedAttr("tdesc.type", TypeAttr::get(type))});
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("shape." + std::to_string(i));
        out_attrs.push_back(FuncArgRenamer::ArgAttrs{
            builder.getNamedAttr("tdesc.shape", builder.getI32IntegerAttr(i))});
      }
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("stride." + std::to_string(i));
        out_attrs.push_back(FuncArgRenamer::ArgAttrs{builder.getNamedAttr(
            "tdesc.stride", builder.getI32IntegerAttr(i))});
      }
      return success();
    });

    RewritePatternSet patterns(context);
    triton::populateFunctionTypeConversions(converter, renamer, patterns);
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    ConversionConfig config;
    config.buildMaterializations = true;

    if (failed(applyPartialConversion(op, target, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
