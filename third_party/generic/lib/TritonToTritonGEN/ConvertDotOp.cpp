#include "TypeConverter.h"

#include "generic/include/TritonToTritonGEN/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDOTOP
#include "generic/include/TritonToTritonGEN/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gen;

namespace {

class DotConversionTarget : public ConversionTarget {
public:
  explicit DotConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonGENDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::DotOp>();
  }
};

struct DotOpConversion : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Value a = rewriter.getRemappedValue(op.getA());
    Value b = rewriter.getRemappedValue(op.getB());
    Value c = rewriter.getRemappedValue(op.getC());

    auto aType = cast<ShapedType>(a.getType());
    auto bType = cast<ShapedType>(b.getType());
    auto cType = cast<ShapedType>(c.getType());
    assert(aType.getRank() == bType.getRank() &&
           bType.getRank() == cType.getRank() &&
           (aType.getRank() == 2 || aType.getRank() == 3) &&
           "Mixed ranks, not 2d or 3d matmul, unknown type of op");

    rewriter.replaceOpWithNewOp<gen::DotOp>(op, a, b, c, op.getInputPrecision(),
                                            op.getMaxNumImpreciseAcc());
    return success();
  }
};

struct ConvertDotOp : public triton::impl::ConvertDotOpBase<ConvertDotOp> {
  using ConvertDotOpBase::ConvertDotOpBase;

  ConvertDotOp() : ConvertDotOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonGENTypeConverter typeConverter;
    DotConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<DotOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace gen {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOp() {
  return std::make_unique<ConvertDotOp>();
}

} // namespace gen
} // namespace triton
} // namespace mlir
