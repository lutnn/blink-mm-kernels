import os

from tvm import autotvm, te

from .tune import tune
from .im2col import (
    compute_im2col, compute_pack,
    schedule_im2col, schedule_pack
)


@autotvm.template("test_im2col")
def test_im2col(input_shape, kernel, strides, padding, pad_value, subvec_len, n_blk_size):
    input_data = te.placeholder(
        input_shape,
        name="input_data", dtype="float32"
    )

    cfg = autotvm.get_config()

    im2col_data = compute_im2col(
        cfg, input_data, kernel, strides, padding, pad_value)

    output = compute_pack(cfg, im2col_data, subvec_len, n_blk_size)

    s = te.create_schedule([output.op])

    s, _ = schedule_im2col(cfg, s, im2col_data)
    s, _ = schedule_pack(cfg, s, output)

    return s, [input_data, output]


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
        # resnet20 for cifar10
        ((1, 16, 32, 32), (3, 3), (1, 1), (1, 1), 0, 9, 8),
        ((1, 32, 16, 16), (3, 3), (1, 1), (1, 1), 0, 9, 8),
        ((1, 64, 8, 8), (3, 3), (1, 1), (1, 1), 0, 9, 8),

        # resnet18 for imagenet
        ((1, 64, 56, 56), (3, 3), (1, 1), (1, 1), 0, 9, 8),
        ((1, 64, 56, 56), (3, 3), (2, 2), (1, 1), 0, 9, 8),
        # ((1, 64, 56, 56), (1, 1), (2, 2), (0, 0), 0, 4, 8),
        ((1, 128, 28, 28), (3, 3), (1, 1), (1, 1), 0, 9, 8),
        ((1, 128, 28, 28), (3, 3), (2, 2), (1, 1), 0, 9, 8),
        ((1, 256, 14, 14), (3, 3), (1, 1), (1, 1), 0, 9, 8),
        ((1, 128, 28, 28), (1, 1), (2, 2), (0, 0), 0, 4, 8),
        ((1, 256, 14, 14), (3, 3), (2, 2), (1, 1), 0, 9, 8),
        ((1, 256, 14, 14), (1, 1), (2, 2), (0, 0), 0, 4, 8),
        ((1, 512, 7, 7), (3, 3), (1, 1), (1, 1), 0, 9, 8),
    ]:
        func = tune(
            "test_im2col",
            test_im2col,
            args, func_name="main",
            target=target,
            code_path="code",
            **remote_params
        )


if __name__ == "__main__":
    os.environ["TVM_NUM_THREADS"] = "1"

    verify_kernels("arm")
