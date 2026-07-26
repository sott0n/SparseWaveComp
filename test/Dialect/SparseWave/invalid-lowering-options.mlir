// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=unknown' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-MAPPING
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=wave-per-row wave-size=64' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=wave-per-row block-size=48' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=block-per-row wave-size=64' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-WAVE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=block-per-row block-size=48' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=unknown' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-MAPPING
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-BLOCK-SIZE

// INVALID-MAPPING: unsupported SpMV mapping strategy 'unknown';
// INVALID-MAPPING-SAME: expected 'thread-per-row', 'wave-per-row', or 'block-per-row'
// INVALID-BLOCK-SIZE: SpMV block size must be between 1 and 1024
// INVALID-WAVE-SIZE: wave-per-row currently requires Wave32, but got wave size 64
// INVALID-WAVE-BLOCK-SIZE: wave-per-row requires the SpMV block size to be a multiple of 32, but got 48
// INVALID-BLOCK-WAVE-SIZE: block-per-row currently requires Wave32, but got wave size 64
// INVALID-BLOCK-BLOCK-SIZE: block-per-row requires the SpMV block size to be a multiple of 32, but got 48
// INVALID-SPMM-MAPPING: unsupported SpMM mapping strategy 'unknown'; expected 'thread-per-output'
// INVALID-SPMM-BLOCK-SIZE: SpMM block size must be between 1 and 1024

module {
}
