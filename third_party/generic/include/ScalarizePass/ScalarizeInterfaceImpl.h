#ifndef MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H
#define MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace triton {
namespace gen {

void registerTritonOpScalarizeExternalModels(DialectRegistry &registry);

} // namespace gen
} // namespace triton
} // namespace mlir

#endif // MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H
