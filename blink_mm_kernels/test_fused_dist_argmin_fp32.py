import os

import numpy as np
from tvm import autotvm, te
import tvm

from .tune import tune
from .verify import verify
from .fused_dist_argmin_fp32 import *


@autotvm.template("test_fused_dist_argmin_fp32")
def test_fused_dist_argmin_fp32(input_shape, kernel, strides, padding, subvec_len):
    if kernel is not None:
        batch, channels, height, width = input_shape
        out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
        out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
        n = batch * out_height * out_width
        d = channels * kernel[0] * kernel[1]
        nc = d // subvec_len
        num_centroids = 16
    else:
        batch, num_samples, d = input_shape
        n = batch * num_samples
        nc = d // subvec_len
        assert d % subvec_len == 0
        num_centroids = 16

    input_data = te.placeholder(
        input_shape,
        name="input_data", dtype="float32"
    )
    centroids = te.placeholder(
        (nc, num_centroids // 8, subvec_len, 8),
        name="centroids", dtype="float32"
    )

    cfg = autotvm.get_config()

    output = compute_fused_dist_argmin_fp32(
        cfg, input_data, centroids, kernel, strides, padding)

    s = te.create_schedule([output.op])
    s, _ = schedule_fused_dist_argmin_fp32(cfg, s, output)

    return s, [input_data, centroids, output]


def gt(input_shape, kernel, strides, padding, subvec_len):
    from numpy_ml.neural_nets.utils import im2col
    from scipy.spatial import distance_matrix

    if kernel is not None:
        batch, channels, height, width = input_shape
        out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
        out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
        n = batch * out_height * out_width
        n_padded = (n + 15) // 16 * 16
        d = channels * kernel[0] * kernel[1]
        nc = d // subvec_len
        num_centroids = 16

        data = np.random.uniform(size=input_shape).astype(np.float32)
        assert strides[0] == strides[1]
        im2col_data, _ = im2col(
            np.transpose(data, (0, 2, 3, 1)),
            (*kernel, channels, channels),  # fake out channels
            padding, strides[0]
        )
        packed_x = np.pad(
            im2col_data.reshape((d, n)),
            [(0, 0), (0, n_padded - n)],
            mode="constant", constant_values=0
        )
        packed_x = np.transpose(
            packed_x.reshape((nc, subvec_len, n_padded // 8, 8)),
            (0, 2, 1, 3)
        )
    else:
        batch, num_samples, d = input_shape
        n = batch * num_samples
        nc = d // subvec_len
        assert d % subvec_len == 0 and n % 16 == 0
        n_padded = n
        data = np.random.uniform(size=input_shape).astype(np.float32)
        packed_x = np.transpose(
            data.reshape((n // 8, 8, nc, subvec_len)),
            (2, 0, 3, 1)
        )
        num_centroids = 16

    packed_y = np.random.uniform(size=(nc, num_centroids // 8, subvec_len, 8))\
        .astype(np.float32)

    x = np.transpose(packed_x, (0, 1, 3, 2)).reshape(
        (nc, n_padded, subvec_len))
    y = np.transpose(packed_y, (0, 1, 3, 2)).reshape(
        (nc, num_centroids, subvec_len))

    z = np.empty((nc, n_padded, num_centroids), np.float32)
    for i in range(nc):
        z[i] = distance_matrix(x[i], y[i])

    z = np.argmin(z, axis=-1).astype(np.int8)

    return (data, packed_y), (z, )


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

    for args in [
        # bert
        ((1, 64, 768), None, None, None, 16),
        ((1, 64, 3072), None, None, None, 16),

        # resnet20 for cifar10
        ((1, 16, 32, 32), (3, 3), (1, 1), (1, 1), 9),
        ((1, 32, 16, 16), (3, 3), (1, 1), (1, 1), 9),
        ((1, 64, 8, 8), (3, 3), (1, 1), (1, 1), 9),

        # resnet18 for imagenet
        ((1, 64, 56, 56), (3, 3), (1, 1), (1, 1), 9),
        ((1, 64, 56, 56), (3, 3), (2, 2), (1, 1), 9),
        ((1, 64, 56, 56), (1, 1), (2, 2), (0, 0), 4),
        ((1, 128, 28, 28), (3, 3), (1, 1), (1, 1), 9),
        ((1, 128, 28, 28), (3, 3), (2, 2), (1, 1), 9),
        ((1, 256, 14, 14), (3, 3), (1, 1), (1, 1), 9),
        ((1, 128, 28, 28), (1, 1), (2, 2), (0, 0), 4),
        ((1, 256, 14, 14), (3, 3), (2, 2), (1, 1), 9),
        ((1, 256, 14, 14), (1, 1), (2, 2), (0, 0), 4),
        ((1, 512, 7, 7), (3, 3), (1, 1), (1, 1), 9),
    ]:
        func = tune(
            "test_fused_dist_argmin_fp32",
            test_fused_dist_argmin_fp32,
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
