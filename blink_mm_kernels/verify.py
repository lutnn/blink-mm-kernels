from tvm.contrib import utils, ndk
from tvm import runtime, rpc
import tvm.testing


def verify(
    func, args, rets,
    remote=False, key=None, host=None, port=None
):
    if remote:
        temp = utils.tempdir()
        filename = "libfunc.so"
        path = temp.relpath(filename)
        func.export_library(path, ndk.create_shared)
        remote = rpc.connect_tracker(host, port).request(key)
        remote.upload(path)
        func = remote.load_module(filename)
        dev = remote.cpu()
    else:
        dev = runtime.device("cpu")

    tvm_args = [tvm.nd.array(arg, dev) for arg in args]
    tvm_rets = [tvm.nd.empty(ret.shape, device=dev, dtype=ret.dtype)
                for ret in rets]

    func(*tvm_args, *tvm_rets)

    for i in range(len(rets)):
        tvm.testing.assert_allclose(rets[i], tvm_rets[i].numpy(), rtol=1e-4)

    time_f = func.time_evaluator(
        func.entry_name, dev, repeat=10, min_repeat_ms=10, number=10)
    results = time_f(*tvm_args, *tvm_rets).mean
    return results
