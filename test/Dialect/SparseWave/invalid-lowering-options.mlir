// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='mapping=wave-per-row' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-MAPPING
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=0' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE
// RUN: not sparsewave-opt %s \
// RUN:   --convert-sparsewave-to-gpu='block-size=1025' \
// RUN:   2>&1 | FileCheck %s --check-prefix=INVALID-BLOCK-SIZE

// INVALID-MAPPING: unsupported SpMV mapping strategy 'wave-per-row';
// INVALID-MAPPING-SAME: expected 'thread-per-row'
// INVALID-BLOCK-SIZE: SpMV block size must be between 1 and 1024

module {
}
