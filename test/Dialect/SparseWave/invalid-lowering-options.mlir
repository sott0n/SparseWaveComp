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
// RUN:   --schedule-sparsewave-position='mapping=block' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-POSITION-MAPPING
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-POSITION-SCHEDULE-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave wave-size=64' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-POSITION-WAVE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave block-size=48' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-POSITION-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='thread-chunk-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-POSITION-CHUNK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --schedule-sparsewave-position='mapping=wave thread-chunk-size=2' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-WAVE-CHUNK-SIZE
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
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-tile-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-TILE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-tile-size=33' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-TILE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=wave-per-row-tile wave-size=64' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-WAVE-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='spmm-mapping=wave-per-row-tile spmm-block-size=48' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SPMM-WAVE-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='sddmm-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SDDMM-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='sddmm-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-SDDMM-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='row-reduction-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ROW-REDUCTION-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='row-reduction-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ROW-REDUCTION-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='rowwise-map-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ROWWISE-MAP-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='rowwise-map-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ROWWISE-MAP-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='elementwise-block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ELEMENTWISE-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='elementwise-block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-ELEMENTWISE-BLOCK-SIZE

// INVALID-MAPPING: unsupported SpMV mapping strategy 'unknown';
// INVALID-MAPPING-SAME: expected 'thread-per-row', 'wave-per-row', or 'block-per-row'
// INVALID-BLOCK-SIZE: SpMV block size must be between 1 and 1024
// INVALID-WAVE-SIZE: wave-per-row currently requires Wave32, but got wave size 64
// INVALID-WAVE-BLOCK-SIZE: wave-per-row requires the SpMV block size to be a multiple of 32, but got 48
// INVALID-POSITION-MAPPING: unsupported position mapping 'block'; expected 'thread' or 'wave'
// INVALID-POSITION-SCHEDULE-BLOCK-SIZE: position block size must be between 1 and 1024
// INVALID-POSITION-WAVE-SIZE: wave position mapping currently requires Wave32, but got 64
// INVALID-POSITION-BLOCK-SIZE: wave position mapping requires the block size to be a multiple of 32, but got 48
// INVALID-POSITION-CHUNK-SIZE: thread position chunk size must be positive, but got 0
// INVALID-WAVE-CHUNK-SIZE: thread position chunk size applies only to thread mapping
// INVALID-BLOCK-WAVE-SIZE: block-per-row currently requires Wave32, but got wave size 64
// INVALID-BLOCK-BLOCK-SIZE: block-per-row requires the SpMV block size to be a multiple of 32, but got 48
// INVALID-SPMM-MAPPING: unsupported SpMM mapping strategy 'unknown';
// INVALID-SPMM-MAPPING-SAME: expected 'thread-per-output' or 'wave-per-row-tile'
// INVALID-SPMM-BLOCK-SIZE: SpMM block size must be between 1 and 1024
// INVALID-SPMM-TILE-SIZE: SpMM tile size must be between 1 and 32
// INVALID-SPMM-WAVE-SIZE: wave-per-row-tile currently requires Wave32, but got wave size 64
// INVALID-SPMM-WAVE-BLOCK-SIZE: wave-per-row-tile requires the SpMM block size to be a multiple of 32, but got 48
// INVALID-SDDMM-BLOCK-SIZE: SDDMM block size must be between 1 and 1024
// INVALID-ROW-REDUCTION-BLOCK-SIZE: CSR row-reduction block size must be between 1 and 1024
// INVALID-ROWWISE-MAP-BLOCK-SIZE: CSR row-wise map block size must be between 1 and 1024
// INVALID-ELEMENTWISE-BLOCK-SIZE: elementwise block size must be between 1 and 1024

module {
}
