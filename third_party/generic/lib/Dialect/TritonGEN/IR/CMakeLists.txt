add_triton_library(TritonGENIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
