#include "SparseAttentionToSparseWave.h"
#include "sparsewave/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::sparsewave::registerAllPasses();
  mlir::sparsewave::example::registerSparseAttentionToSparseWavePass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::sparsewave::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "SparseWave PyTorch example optimizer\n", registry));
}
