set(SPARSEWAVE_BUNDLE "${LLVM_RUNTIME_OUTPUT_INTDIR}/sparsewave-bundle")

configure_file(sparsewave-bundle.py "${SPARSEWAVE_BUNDLE}" COPYONLY)

add_custom_target(sparsewave-bundle ALL
  DEPENDS "${SPARSEWAVE_BUNDLE}"
  COMMENT "Prepare the sparsewave-bundle tool"
)
