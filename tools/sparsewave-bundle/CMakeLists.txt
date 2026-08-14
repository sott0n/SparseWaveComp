set(SPARSEWAVE_BUNDLE "${LLVM_RUNTIME_OUTPUT_INTDIR}/sparsewave-bundle")
set(SPARSEWAVE_BUNDLE_MODULES
  sparsewave_bundle_hsaco.py
  sparsewave_bundle_ir.py
  sparsewave_bundle_manifest.py
  sparsewave_bundle_verify.py
)

configure_file(sparsewave-bundle.py "${SPARSEWAVE_BUNDLE}" COPYONLY)
set(SPARSEWAVE_BUNDLE_MODULE_OUTPUTS)
foreach(module IN LISTS SPARSEWAVE_BUNDLE_MODULES)
  set(MODULE_OUTPUT "${LLVM_RUNTIME_OUTPUT_INTDIR}/${module}")
  configure_file("${module}" "${MODULE_OUTPUT}" COPYONLY)
  list(APPEND SPARSEWAVE_BUNDLE_MODULE_OUTPUTS "${MODULE_OUTPUT}")
endforeach()

add_custom_target(sparsewave-bundle ALL
  DEPENDS
    "${SPARSEWAVE_BUNDLE}"
    ${SPARSEWAVE_BUNDLE_MODULE_OUTPUTS}
  COMMENT "Prepare the sparsewave-bundle tool"
)
