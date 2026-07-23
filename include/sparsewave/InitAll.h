#ifndef SPARSEWAVE_INITALL_H
#define SPARSEWAVE_INITALL_H

namespace mlir {
class DialectRegistry;

namespace sparsewave {

void registerAllDialects(DialectRegistry &registry);
void registerAllPasses();

} // namespace sparsewave
} // namespace mlir

#endif // SPARSEWAVE_INITALL_H
