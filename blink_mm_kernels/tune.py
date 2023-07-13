import sys
import logging

import tvm
from tvm import autotvm
from tvm.contrib import utils


def tune(
    task_name, task_func, args,
    func_name="main", log_filepath=None, n_trial=1024,
    remote=False, target="llvm", key=None, host=None, port=None,
    code_path=None
):
    if log_filepath is None:
        temp = utils.tempdir()
        log_filepath = temp.relpath("log.json")

    task = autotvm.task.create(
        task_name,
        args=args,
        target=target
    )

    logger = logging.getLogger("autotvm")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(sys.stdout)]

    if remote:
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk"),
            runner=autotvm.RPCRunner(key, host, port)
        )
    else:
        measure_option = autotvm.measure_option(
            builder="local", runner="local")

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=n_trial,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_filepath)],
    )

    with autotvm.apply_history_best(log_filepath):
        with tvm.target.Target(task.target):
            s, arg_bufs = task_func(*args)

    ret = tvm.lower(s, arg_bufs, name=func_name)

    if code_path is not None:
        code = str(tvm.lower(s, arg_bufs, simple_mode=True))
        with open(code_path + ".tir", "w") as f:
            f.write(code)
        func = tvm.build(ret, target=target)
        func.save(code_path + ".s", "s")

    return ret
