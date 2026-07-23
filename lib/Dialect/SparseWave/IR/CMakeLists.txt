add_mlir_dialect_library(SparseWaveDialect
  SparseWaveDialect.cpp
  SparseWaveOps.cpp

  DEPENDS
  SparseWaveOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
