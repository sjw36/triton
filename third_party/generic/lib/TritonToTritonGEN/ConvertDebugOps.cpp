#include "TypeConverter.h"

#include "generic/include/TritonToTritonGEN/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDEBUGOPS
#include "generic/include/TritonToTritonGEN/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gen;

namespace {

class DebugOpsConversionTarget : public ConversionTarget {
public:
  explicit DebugOpsConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonGENDialect>();

    addLegalOp<arith::ConstantOp>();
    addLegalOp<memref::AllocOp>();
    addLegalOp<memref::DeallocOp>();
    addLegalOp<memref::CastOp>();

    addIllegalOp<triton::PrintOp>();
    addIllegalOp<triton::AssertOp>();
  }
};

struct PrintOpConversion : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // It lowers to triton_gen.print after converting tensor types to vectors.
    // (tt.print doesn't accept vector types, so we have this intermediate op.)
    if (op.getNumOperands() == 0) {
      rewriter.create<triton::gen::PrintOp>(loc, op.getPrefix(), op.getHex(),
                                            ValueRange{},
                                            llvm::SmallVector<int, 0>{});
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = op.getOperands()[i];
      auto isSigned = {op.getIsSigned()[i]};
      if (!isa<RankedTensorType>(operand.getType())) {
        rewriter.create<triton::gen::PrintOp>(
            loc, op.getPrefix(), op.getHex(),
            rewriter.getRemappedValue(operand), isSigned);
        continue;
      }

      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto elemTy = tensorTy.getElementType();
      if (isa<triton::PointerType>(elemTy)) {
        elemTy = rewriter.getI64Type();
      }
      MemRefType memRefTy = MemRefType::get(tensorTy.getShape(), elemTy);

      Value allocVal = rewriter.create<memref::AllocOp>(
          loc, memRefTy, rewriter.getI64IntegerAttr(64));

      Value vec = rewriter.getRemappedValue(operand);
      VectorType vecTy = cast<VectorType>(vec.getType());

      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> indices(vecTy.getRank(), zeroIdx);

      rewriter.create<vector::TransferWriteOp>(loc, vec, allocVal, indices);

      Value allocUnrankedVal = rewriter.create<memref::CastOp>(
          loc, UnrankedMemRefType::get(elemTy, memRefTy.getMemorySpace()),
          allocVal);

      rewriter.create<triton::gen::PrintOp>(loc, op.getPrefix(), op.getHex(),
                                            allocUnrankedVal, isSigned);

      rewriter.create<memref::DeallocOp>(loc, allocVal);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AssertOpConversion : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value acc = rewriter.create<arith::ConstantOp>(loc, i1_ty,
                                                   rewriter.getOneAttr(i1_ty));
    Value condition = rewriter.getRemappedValue(op.getCondition());
    SmallVector<bool> dimsToReduce(
        cast<VectorType>(condition.getType()).getRank(), true);
    condition = rewriter.create<vector::MultiDimReductionOp>(
        loc, condition, acc, dimsToReduce, vector::CombiningKind::AND);
    rewriter.replaceOpWithNewOp<triton::gen::AssertOp>(op, condition,
                                                       op.getMessage());
    return success();
  }
};

struct ConvertDebugOps
    : public triton::impl::ConvertDebugOpsBase<ConvertDebugOps> {
  using ConvertDebugOpsBase::ConvertDebugOpsBase;

  ConvertDebugOps() : ConvertDebugOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonGENTypeConverter typeConverter;
    DebugOpsConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<PrintOpConversion>(typeConverter, context);
    patterns.add<AssertOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace gen {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDebugOps() {
  return std::make_unique<ConvertDebugOps>();
}

} // namespace gen
} // namespace triton
} // namespace mlir
