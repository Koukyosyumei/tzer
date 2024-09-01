from typing import Dict
from tvm import tir
from .abstract import TIRAbstractTransformer

class TIRBufferVarRebinder(TIRAbstractTransformer[None]):
    def __init__(self) -> None:
        self.iter = 0
        self.buffer_map: Dict[tir.Var, tir.Buffer] = {}

    """Rebind buffer variables"""

    def visit_load(self, op: tir.BufferLoad, arg: None) -> tir.BufferLoad:
        new_buffer = tir.decl_buffer((1,), op.dtype)
        new_var = tir.Var(f'load_{self.iter}', 'handle')
        self.buffer_map[new_var] = new_buffer
        self.iter += 1
        return tir.BufferLoad(
            new_buffer.data,
            op.index,
            op.predicate,
        )

    def visit_store(self, op: tir.BufferStore, arg: None) -> tir.BufferStore:
        new_buffer = tir.decl_buffer((1,), 'float32')
        new_var = tir.Var(f'store_{self.iter}', 'handle')
        self.buffer_map[new_var] = new_buffer
        self.iter += 1
        return tir.BufferStore(
            new_buffer.data,
            op.value,
            op.index,
            op.predicate
        )


def rebind_buffer_var(func: tir.PrimFunc) -> tir.PrimFunc:
    rebinder = TIRBufferVarRebinder()
    func = rebinder(func, None)
    bf_map = {k: v for k, v in func.buffer_map.items()}
    for k, v in rebinder.buffer_map.items():
        bf_map[k] = v
    return tir.PrimFunc(
        list(func.params) + list(rebinder.buffer_map.keys()),
        func.body,
        buffer_map=bf_map,
    )
