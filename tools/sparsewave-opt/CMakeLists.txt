set(LIBS
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  SparseWaveInit
)

add_llvm_executable(sparsewave-opt sparsewave-opt.cpp)

llvm_update_compile_flags(sparsewave-opt)
target_link_libraries(sparsewave-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(sparsewave-opt)
