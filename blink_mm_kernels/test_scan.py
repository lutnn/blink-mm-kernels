import os

import numpy as np
from tvm import autotvm
import tvm

from .tune import tune
from .verify import verify
from .scan import *


def test_scan(target):
    @autotvm.template("test_scan")
    def func(ncodebooks, n, output_shape):
        n_padded = (n + 15) // 16 * 16
        if len(output_shape) == 4:
            # nchw
            out_features = output_shape[1]
        else:
            # batch, rows, features
            out_features = output_shape[2]

        index_data = te.placeholder(
            (ncodebooks, n_padded),
            name="index_data", dtype="int8"
        )
        lut_data = te.placeholder(
            (out_features, ncodebooks, 16),
            name="lut_data", dtype="int8"
        )
        scale = te.placeholder(
            (1,), name="scale",
            dtype="float32"
        )
        bias = te.placeholder(
            (out_features,), name="bias",
            dtype="float32"
        )

        cfg = autotvm.get_config()

        output_data = compute_scan(
            cfg, index_data, lut_data, output_shape, scale, bias)

        s = te.create_schedule([output_data.op])
        s, _ = schedule_scan(cfg, s, output_data, target)

        return s, [index_data, lut_data, scale, bias, output_data]

    return func


def gt(ncodebooks, n, output_shape):
    n_padded = (n + 15) // 16 * 16
    if len(output_shape) == 4:
        # nchw
        batch, out_features, height, width = output_shape
    else:
        # batch, rows, features
        batch, num_samples, out_features = output_shape

    index_data = np.random.randint(
        0, 16, size=(ncodebooks, n_padded)).astype(np.int8)
    lut_data = np.random.randint(
        0, 256, size=(out_features, ncodebooks, 16)).astype(np.int8)
    scale = np.random.randn(1).astype(np.float32)
    bias = np.random.randn(out_features).astype(np.float32)

    output_data = np.zeros((out_features, n_padded)).astype(np.int32)
    for i in range(out_features):
        for j in range(n_padded):
            for k in range(ncodebooks):
                output_data[i, j] += lut_data[i, k, index_data[k, j]]

    output_data = output_data * scale[0] + bias.reshape((-1, 1))
    output_data = output_data[:, :n].astype(np.float32)

    if len(output_shape) == 4:
        # nchw
        output_data = output_data.reshape((out_features, batch, height, width))
        output_data = np.transpose(output_data, (1, 0, 2, 3))
    else:
        # batch, rows, features
        output_data = output_data.reshape((out_features, batch, num_samples))
        output_data = np.transpose(output_data, (1, 2, 0))

    return (index_data, lut_data, scale, bias), (output_data,)


def verify_kernels(arch):
    if arch == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        remote_params = {
            "remote": True,
            "key": "pixel6",
            "host": "127.0.0.1",
            "port": 9190
        }
    elif arch == "x86":
        target = "llvm -mcpu=core-avx2"
        remote_params = {"remote": False}
    elif arch == "x86_avx512":
        target = "llvm -mcpu=cascadelake"
        remote_params = {"remote": False}

    test_scan_func = test_scan(arch)

    for args in [
        # bert
        (48, 64, (1, 64, 768)),
        (48, 64, (1, 64, 3072)),
        (192, 64, (1, 64, 768)),

        # resnet18 for imagenet
        (64, 3136, (1, 64, 56, 56)),
        (64, 784, (1, 128, 28, 28)),
        (16, 784, (1, 128, 28, 28)),
        (128, 784, (1, 128, 28, 28)),
        (128, 196, (1, 256, 14, 14)),
        (256, 196, (1, 256, 14, 14)),
        (32, 196, (1, 256, 14, 14)),
        (256, 49, (1, 512, 7, 7)),
        (64, 49, (1, 512, 7, 7)),
        (512, 49, (1, 512, 7, 7)),
    ]:
        func = tune(
            "test_scan",
            test_scan_func,
            args, func_name="main",
            target=target,
            code_path="code",
            **remote_params
        )
        verify(
            tvm.build(func, target=target),
            *gt(*args), **remote_params
        )


if __name__ == "__main__":
    os.environ["TVM_NUM_THREADS"] = "1"

    verify_kernels("x86")
