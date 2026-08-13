import torch
from torch_mlir import fx


def import_torch_program(exported_program, *, function_name="main"):
    """Import an ExportedProgram into the torch-mlir Torch dialect."""
    if not isinstance(exported_program, torch.export.ExportedProgram):
        raise TypeError("expected a torch.export.ExportedProgram")

    return fx.export_and_import(
        exported_program,
        output_type="raw",
        decomposition_table={},
        func_name=function_name,
    )


def render_generic_torch_mlir(module):
    """Render Torch MLIR for a consumer that does not register Torch dialect."""
    return module.operation.get_asm(
        print_generic_op_form=True,
        use_local_scope=True,
    )
