// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=64 wave-size=16' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='position-block-size=64 wave-size=32' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE-BLOCK-SIZE

// INVALID-BLOCK-SIZE: position-parallel block size must be between 1 and 1024
// INVALID-WAVE-SIZE: wave position mapping requires a wave size of 32 or 64, but got 16
// INVALID-WAVE-BLOCK-SIZE: wave position mapping requires its block size to be a multiple of the wave size, but got 48 and 32

func.func @wave_mapping(%workerCount: index) {
  sparsewave.position_parallel %workerCount mapping = "wave" block_size = 48 {
  ^bb0(%worker: index, %lane: index, %waveSize: index):
    sparsewave.yield
  }
  return
}
