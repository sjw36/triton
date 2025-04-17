#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-loop-bisect"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexiblity to
// better interleave instructions for better instruction-level parallelism.
//
// This StreamPipeliner class creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
//    1.a. User may also specify `global_prefetch=<s>` to set the number of
//         stages between tt.load and ttg.local_store ops.
//    1.b. User may also specify `local_prefetch=<s>` to set the number of
//         stages between ttg.local_load and compute.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule method).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The ttg.local_store
//    will save the global load value to shared memory and the ttg.local_load
//    will load the relevant tiles for the tt.dot. These operations will be
//    scheduled according to various scheduling schemes outlined below in the
//    initSchedule method (see details there).
// 4. Finally the schedule will be passed to the PipelineExpander to rewrite
//    accordingly. The new implementation will consist of:
//    a. Prologue: containing the ramp-up of num_stages-1 stages for
//       iteratorions i=[0, num_stages-1).
//    b. New loop: ordered by cluster and iterated on each operation by
//       `i + (num_stages-op_stage)`.
//    c. Epilogue: ramp-down of the last `num_stages-1` iterations for the
//       ops in stages 1 to last_stage. This must consider that the loop
//       bounds may be shorter than num_stages. In this case, the epilogue
//       iterations must align with the prologue.
//
class LoopBisect {
public:
  LoopBisect(scf::ForOp _forOp)
      : forOp(_forOp) {
  }

  LogicalResult bisect();

private:

  LogicalResult getMidpoint(OpOperand &opr);
private:
  // Data members
  scf::ForOp forOp;

  DenseMap<Operation *, Value> op2MidPoint;

  // Mapping and indirection level for each `tt.load` to its use.
  SmallVector<std::tuple<Operation *, int, Operation *>> loadOpToIndLevelAndUse;

  // Capture list of new shared memory buffers.
  SmallVector<Value> sharedMemAllocs;
};


LogicalResult LoopBisect::getMidpoint(OpOperand &opr) {
  if (auto cmp = dyn_cast<arith::CmpIOp>(opr.getOwner())) {
    LDBG("CMP: " << cmp);
    auto pred = cmp.getPredicate();
    bool isEq = false;
    switch (pred) {
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::sle:
        isEq = true;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::slt:
        break;
      default:
        // Un-supported cmpi
        return failure();
    }

    // other most be loop invariant
    Value midp = cmp.getOperand(opr.getOperandNumber() ^ 1);
    if (auto *defOther = midp.getDefiningOp()) {
      if (forOp->isAncestor(defOther))
        return failure();
    } else
      assert(0);

    if (isEq) {
      bool isGt = pred == arith::CmpIPredicate::sge || pred == arith::CmpIPredicate::sgt;
      if (opr.getOperandNumber() == 1)
        isGt = !isGt;
      // return i >= c ? c - 1 : c + 1
      auto loc = cmp.getLoc();
      OpBuilder b(forOp);
      b.setInsertionPoint(forOp);
      auto incr = b.create<arith::ConstantIntOp>(loc, isGt ? -1 : 1, 32);
      midp = b.create<arith::AddIOp>(loc, midp, incr);
    }
    op2MidPoint[cmp] = midp;
    return success();
  }
  return failure();
}

LogicalResult LoopBisect::bisect() {
  auto lo = forOp.getLowerBound();
  auto hi = forOp.getUpperBound();
  auto step = forOp.getConstantStep();

  //LDBG("Loop: " << forOp);
  if (!step) {
    LDBG("Non-constant step");
    return failure();
  }

  // get midpoint
  auto iter = forOp.getInductionVar();
  for (OpOperand &use : iter.getUses()) {
    auto res = getMidpoint(use);
  }

  // TODO: for multiple points, determine if they can be sorted, or just pick one
  if (op2MidPoint.size() == 1) {
    auto [cmp, midp] = *op2MidPoint.begin();

    /// TODO(sjw): update upstream peelForLoop
    /// make midp floored with step
    /// bisect loop (lo .. midp)
    /// bisect loop (midp .. hi)
    IRMapping mapping;
    OpBuilder b(forOp);
    b.setInsertionPointAfter(forOp);
    scf::ForOp newForOp = cast<scf::ForOp>(b.clone(*forOp, mapping));
    newForOp.setLowerBound(midp);
    forOp.replaceAllUsesWith(newForOp.getResults());
    newForOp.getInitArgsMutable().assign(forOp->getResults());
    forOp.setUpperBound(midp);

    // replace cmp with constant True/False for each loop
    b.setInsertionPoint(forOp);
    auto loc = cmp->getLoc();
    cmp->replaceAllUsesWith(b.create<arith::ConstantIntOp>(loc, 0, 1));
    auto *newCmp = mapping.lookup(cmp);
    newCmp->replaceAllUsesWith(b.create<arith::ConstantIntOp>(loc, 1, 1));
  }

  return success();
}

struct LoopBisectPass : public TritonAMDGPULoopBisectBase<LoopBisectPass> {
  LoopBisectPass() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<scf::ForOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      LoopBisect sp(forOp);
      if (failed(sp.bisect()))
        continue;
    }
  }

private:

};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPULoopBisectPass() {
  return std::make_unique<LoopBisectPass>();
}
