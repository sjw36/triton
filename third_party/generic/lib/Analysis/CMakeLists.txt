add_triton_library(TritonGENAnalysis
  TensorPtrShapeInfo.cpp

  DEPENDS
  #TritonGENTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  TritonIR
  TritonGENIR
)
