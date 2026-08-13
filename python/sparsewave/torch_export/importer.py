import torch
from torch_mlir import fx


def import_torch_program(exported_program):
    """Import an ExportedProgram into the torch-mlir Torch dialect."""
    if not isinstance(exported_program, torch.export.ExportedProgram):
        raise TypeError("expected a torch.export.ExportedProgram")

    return fx.export_and_import(
        exported_program,
        output_type="raw",
        decomposition_table={},
    )
