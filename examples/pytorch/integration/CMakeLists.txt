set(LIBS
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  SparseWaveInit
)

add_llvm_executable(sparsewave-pytorch-opt
  sparsewave-pytorch-opt.cpp
  SparseAttentionToSparseWave.cpp
)

llvm_update_compile_flags(sparsewave-pytorch-opt)
target_link_libraries(sparsewave-pytorch-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(sparsewave-pytorch-opt)
