#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREWRITETENSORDESCRIPTORTOPOINTER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}

struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  PaddingOption paddingOption;
  bool roundF32ToTF32;
};

Descriptor unpackDescriptor(TensorDescType type, Value desc) {
  int rank = type.getShape().size();
  // Find the make_tensor_descriptor op that created this descriptor
  auto makeTensorDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
  if (!makeTensorDescOp) {
    // desc.emitError("desc must be created by a make_tensor_descriptor op");
    return Descriptor();
  }
  auto base = makeTensorDescOp.getBase();
  auto shape = makeTensorDescOp.getShape();
  auto strides = makeTensorDescOp.getStrides();
  auto paddingOption = makeTensorDescOp.getPadding();
  auto roundF32ToTF32 = makeTensorDescOp.getRoundF32ToTF32();
  Descriptor res;
  res.base = base;
  res.shape = shape;
  res.strides = strides;
  res.paddingOption = paddingOption;
  res.roundF32ToTF32 = roundF32ToTF32;
  return res;
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        triton::ExpandDimsOp::create(builder, loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI64Type());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, indexRowType, offset);
  Value range = triton::MakeRangeOp::create(builder, loc, indexI32RowType, 0,
                                            blockShape[dim]);
  Value i64Range = arith::ExtSIOp::create(builder, loc, indexRowType, range);

  Value offsets = arith::AddIOp::create(builder, loc, splatOffset, i64Range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = triton::SplatOp::create(
        builder, loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, offsets[i], splatStride);
    Value broadcasted = triton::BroadcastOp::create(
        builder, loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        triton::AddPtrOp::create(builder, loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

Value generateMaskFromOffsetRanges(OpBuilder &builder, const Location &loc,
                                   ArrayRef<std::int64_t> blockShape,
                                   Descriptor &desc, ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange = offsetRanges[i];

    // Compare with lower bound
    Value lowerBound = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getI64Type(), 0);
    Value splatLowerBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              offsetWithRange, splatLowerBound);

    // Compare with upper bound (descriptor shape is i32; offsets are i64)
    Value upperBound = arith::ExtSIOp::create(
        builder, loc, builder.getI64Type(), desc.shape[i]);
    Value splatUpperBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), upperBound);
    Value cmpUpper =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                              offsetWithRange, splatUpperBound);

    // And and broadcast
    Value andResult = arith::AndIOp::create(builder, loc, cmpLower, cmpUpper);
    Value broadcasted =
        triton::BroadcastOp::create(builder, loc, maskTensorType, andResult);

    // And up all results
    if (!mask) {
      mask = broadcasted;
    } else {
      mask = arith::AndIOp::create(builder, loc, mask, broadcasted);
    }
  }

  return mask;
}

Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                   ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                      offsetRanges);
}

Value generateOther(OpBuilder &builder, Location loc, Type scalarTy,
                    ArrayRef<int64_t> blockShape,
                    triton::PaddingOption paddingOption) {
  auto blockTy = RankedTensorType::get(blockShape, scalarTy);
  if (mlir::isa<FloatType>(scalarTy)) {
    auto floatTy = mlir::cast<FloatType>(scalarTy);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getFloatAttr(floatTy, nan)));
    auto zeroValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getZeroAttr(floatTy)));
    if (paddingOption == triton::PaddingOption::PAD_NAN) {
      return nanValue;
    } else {
      return zeroValue;
    }
  } else {
    auto attr = builder.getZeroAttr(blockTy);
    return arith::ConstantOp::create(builder, loc, attr);
  }
}

Value generateOther(OpBuilder &builder, Location loc, TensorDescType descTy,
                    PaddingOption paddingOption) {
  auto blockTy = descTy.getSignlessBlockType();
  return generateOther(builder, loc, blockTy.getElementType(),
                       blockTy.getShape(), paddingOption);
}

Type getI32TypeLike(OpBuilder &builder, Type ty) {
  if (auto shapedTy = dyn_cast<ShapedType>(ty))
    return shapedTy.clone(builder.getI32Type());
  return builder.getI32Type();
}

Value getI32ConstLike(OpBuilder &builder, Location loc, Type likeType,
                      int32_t value) {
  auto i32Ty = getI32TypeLike(builder, likeType);
  if (auto shapedTy = dyn_cast<ShapedType>(i32Ty)) {
    auto attr =
        DenseElementsAttr::get(shapedTy, builder.getI32IntegerAttr(value));
    return arith::ConstantOp::create(builder, loc, shapedTy, attr);
  }
  return arith::ConstantOp::create(builder, loc, i32Ty,
                                   builder.getI32IntegerAttr(value));
}

Value roundF32ToTF32(OpBuilder &builder, Location loc, Value value) {
  auto valueTy = value.getType();
  auto i32Ty = getI32TypeLike(builder, valueTy);
  auto bits = triton::BitcastOp::create(builder, loc, i32Ty, value);

  auto expMask = getI32ConstLike(builder, loc, i32Ty, 0x7F800000);
  auto exp = arith::AndIOp::create(builder, loc, bits, expMask);
  auto isSpecial = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         exp, expMask);

  auto shift = getI32ConstLike(builder, loc, i32Ty, 13);
  auto lsb = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, bits, shift),
      getI32ConstLike(builder, loc, i32Ty, 1));
  auto roundBias = arith::AddIOp::create(
      builder, loc, lsb, getI32ConstLike(builder, loc, i32Ty, 0x00000FFF));
  auto rounded = arith::AndIOp::create(
      builder, loc, arith::AddIOp::create(builder, loc, bits, roundBias),
      getI32ConstLike(builder, loc, i32Ty, 0xFFFFE000));
  auto outBits =
      arith::SelectOp::create(builder, loc, isSpecial, bits, rounded);
  return triton::BitcastOp::create(builder, loc, valueTy, outBits);
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
}

struct RewriteMakeTensorDesc : OpRewritePattern<triton::MakeTensorDescOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::MakeTensorDescOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStridesPaddingOption;
    llvm::append_values(ptrShapeStridesPaddingOption, op.getBase());
    llvm::append_range(ptrShapeStridesPaddingOption,
                       castToI64(rewriter, op.getShape()));
    llvm::append_range(ptrShapeStridesPaddingOption, op.getStrides());
    rewriter.replaceOp(op, ptrShapeStridesPaddingOption);
    return mlir::success();
  }
};

struct RewriteRankPattern : OpRewritePattern<triton::DescriptorRankOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorRankOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto rank = desc.shape.size();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getI32Type(), rewriter.getI32IntegerAttr(rank));
    return success();
  }
};

struct RewriteShapePattern : OpRewritePattern<triton::DescriptorShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto dimOp = dyn_cast<arith::ConstantIntOp>(op.getDim().getDefiningOp());
    if (!dimOp) {
      // perhaps support dynamic dims later with select
      return op->emitError("dim must be a constant integer");
    }
    auto dim = dimOp.value();
    rewriter.replaceOp(op, desc.shape[dim]);
    return mlir::success();
  }
};

struct RewriteStridePattern : OpRewritePattern<triton::DescriptorStrideOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorStrideOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto dimOp = dyn_cast<arith::ConstantIntOp>(op.getDim().getDefiningOp());
    if (!dimOp) {
      // perhaps support dynamic dims later with select
      return op->emitError("dim must be a constant integer");
    }
    auto dim = dimOp.value();
    rewriter.replaceOp(op, desc.strides[dim]);
    return mlir::success();
  }
};

struct RewriteLoadPattern : OpRewritePattern<triton::DescriptorLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto other = generateOther(rewriter, loc, descTy, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, generatePtr(rewriter, loc, blockShape, desc, offsets),
        generateMask(rewriter, loc, blockShape, desc, offsets), other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getElementType().isF32() && desc.roundF32ToTF32) {
      result = roundF32ToTF32(rewriter, loc, result);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteStorePattern : OpRewritePattern<triton::DescriptorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getShape();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::pair<Value, Value>
generateGatherScatterPtrMask(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> blockShape, Descriptor &desc,
                             Value xOffsets, Value yOffset) {
  Value xOffsetRange =
      expandOffsets(builder, loc, blockShape, xOffsets, /*dim=*/0);
  yOffset = castToI64(builder, {yOffset})[0];
  auto xOffsetI64Ty = RankedTensorType::get(
      cast<RankedTensorType>(xOffsetRange.getType()).getShape(),
      yOffset.getType());
  xOffsetRange =
      arith::ExtSIOp::create(builder, loc, xOffsetI64Ty, xOffsetRange);
  auto yOffsetRange =
      getExpandedOffsetWithRange(builder, loc, blockShape, yOffset, /*dim=*/1);
  auto ptr = generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                         {xOffsetRange, yOffsetRange});
  auto mask = generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                           {xOffsetRange, yOffsetRange});
  return {ptr, mask};
}

struct RewriteGatherPattern : OpRewritePattern<triton::DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getResult().getType().getShape();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto other = generateOther(rewriter, loc,
                               descTy.getSignlessBlockType().getElementType(),
                               blockShape, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, ptr, mask, other, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getSignlessBlockType().getElementType().isF32() &&
        desc.roundF32ToTF32) {
      result = roundF32ToTF32(rewriter, loc, result);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteScatterPattern : OpRewritePattern<triton::DescriptorScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getSrc().getType().getShape();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, ptr, op.getSrc(), mask, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::optional<RMWOp> translateReduceKind(DescriptorReduceKind kind,
                                         TensorDescType ty) {
  auto scalarTy = ty.getElementType();
  switch (kind) {
  case DescriptorReduceKind::ADD:
    return scalarTy.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMIN;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MIN;
    }
    return {};
  case DescriptorReduceKind::MAX:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMAX;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MAX;
    }
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    break;
  }
  return {};
}

struct RewriteReducePattern : OpRewritePattern<triton::DescriptorReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getShape();
    auto desc = unpackDescriptor(descTy, op.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto rmwOp = translateReduceKind(op.getKind(), descTy);
    if (!rmwOp) {
      std::string msgstring;
      llvm::raw_string_ostream msg(msgstring);
      msg << "Cannot fallback on descriptor atomic op, unsupported for type "
          << descTy.getElementType();
      return op->emitError(msgstring);
    }

    triton::AtomicRMWOp::create(
        rewriter, loc, descTy.getSignlessBlockType(), *rmwOp,
        generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        MemSemantic::RELEASE, MemSyncScope::GPU);
    op.erase();
    return success();
  }
};

/**
 * @brief Lower tensor-descriptor IR to pointer-based `tt.load` / `tt.store` /
 * atomics using greedy pattern rewriting (`applyPatternsGreedily`).
 *
 * @details The pass does not run `DialectConversion`, does not use a
 * `TypeConverter`, and does not rewrite `tt.func` signatures or call/return
 * edges—those are handled by `triton-decompose-tensor-descriptor-parameters`.
 *
 * Patterns peel descriptor SSA values back to the defining
 * `tt.make_tensor_descriptor` (see `unpackDescriptor`) to recover base pointer,
 * per-dimension shapes, and strides, then replace descriptor consumers with
 * ordinary pointer arithmetic, masks, and `tt.load` / `tt.store` /
 * `tt.atomic_rmw` as appropriate (gather, scatter, reduce). Auxiliary ops
 * (`tt.descriptor_rank`, shape/stride queries) fold to constants or the
 * underlying SSA producers. `tt.make_tensor_descriptor` itself can be replaced
 * by a flat tuple of values (pointer, promoted shapes, strides, padding and
 * TF32-rounding flags) when full lowering is enabled.
 *
 * When `keepTensorDescOps` is set, descriptor-producing and descriptor-memory
 * patterns are skipped so descriptor ops remain in the IR for later passes.
 */
class TritonRewriteTensorDescriptorToPointerPass
    : public impl::TritonRewriteTensorDescriptorToPointerBase<
          TritonRewriteTensorDescriptorToPointerPass> {
public:
  using impl::TritonRewriteTensorDescriptorToPointerBase<
      TritonRewriteTensorDescriptorToPointerPass>::
      TritonRewriteTensorDescriptorToPointerBase;

  void runOnOperation() override {
    auto op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<RewriteRankPattern, RewriteShapePattern, RewriteStridePattern>(
        &getContext());

    if (!keepTensorDescOps) {
      patterns.add<RewriteMakeTensorDesc, RewriteLoadPattern,
                   RewriteStorePattern, RewriteGatherPattern,
                   RewriteScatterPattern, RewriteReducePattern>(&getContext());
    }

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton
