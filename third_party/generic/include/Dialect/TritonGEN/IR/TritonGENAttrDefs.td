#ifndef TRITONGEN_ATTRDEFS
#define TRITONGEN_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "Dialect/TritonGEN/IR/TritonGENDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonGEN Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonGEN_AttrTrait : AttrInterface<"TritonGEN_AttrTrait"> {
  let cppNamespace = "::mlir::triton::gen";
}

class TritonGEN_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGEN_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGEN_AttrTrait], traits), baseCppClass> {

  let description = [{TritonGEN attr.}];
  let attrName = "triton.gen." # attrMnemonic;
}

#endif
