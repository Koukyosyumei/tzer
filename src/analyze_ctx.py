from sys import meta_path
import tvm
from tvm import runtime
from tvm.ir import transform
from tvm.ir.module import IRModule
import tvm.relay as relay
from tvm import tir
from tvm.relay.backend import graph_executor_codegen

import time
import numpy as np
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name

from tzer import template, fuzz
from tzer.context import Context

from tzer.relay_seeds import MODEL_SEEDS

target_seeds = MODEL_SEEDS[4::]

seed = target_seeds[0]


if __name__ == "__main__":
    ctx = fuzz.make_context(seed)
    # ['tvm.relay.transform.AnnotateSpans', 'tvm.relay.transform.DynamicToStatic', 'tvm.relay.transform.SimplifyExpr', 'tvm.relay.transform.FoldScaleAxis', 'tvm.relay.transform.FoldExplicitPadding', 'tvm.relay.transform.CanonicalizeCast', 'tvm.relay.transform.FoldConstant']
    ctx.load("/home/koukyosyumei/Dev/tzer/fuzzing-report-b18f9d3c-f98c-40da-b24c-3ceeccf4b45c/InternalError__4fccc14d-0c35-4dbf-a902-3b9007cefbeb.ctx")
    # template.execute_both_mode(ctx)

    params = ctx.runtime.params
    module = ctx.runtime.module
    target = ctx.compile.target
    dev = tvm.device("cpu", 0) #tvm.cpu(0)

    if params is not None:
        module = IRModule.from_expr(bind_params_by_name(module["main"], params))

    with tvm.transform.PassContext(opt_level=0):
        libo0 = relay.build(module, target=target, params=params)
        graph_exe_o0 = graph_executor.GraphModule(libo0["default"](ctx.compile.get_device()))

    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

    with tvm.transform.PassContext(opt_level=4):
        with ctx.compile.target: # There can be target-aware passes...
            # compile the model
            module = tvm.transform.Sequential(passes=[f() for f in ctx.compile.relay_pass_types], opt_level=4)(module)
    
    # convert relay ir to tir
    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

        with ctx.compile.target:
            graph, lowered_func, params = graph_executor_codegen.GraphExecutorCodegen(None, target=target).codegen(module, module['main'])
            name_to_lowered_func = {str(k):v for k, v in lowered_func.items()}
            ir_m = name_to_lowered_func[str(target)]

    with tvm.transform.PassContext(opt_level=4):
        with ctx.compile.target:
            passes = [n.mutate() for n in ctx.compile.tir_pass_nodes]
            passes = [p for p in passes if str(p) != "PrimFuncPass(tir.LowerTVMBuiltin, opt_level=0)"]
            opt = tvm.transform.Sequential(
                    passes=passes,
                    opt_level=4
            )
            opt_ir_m = opt(ir_m)

        opt_execute = tvm.build(opt_ir_m, target=target)

        graph_exe_opt = tvm.contrib.graph_executor.create(graph, opt_execute, dev)
        graph_exe_opt.load_params(runtime.save_param_dict(params))