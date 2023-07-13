import os

from tvm import autotvm, te

from .fused_dist_argmin_fp32 import compute_fused_dist_argmin_fp32, schedule_fused_dist_argmin_fp32
from .scan import compute_scan, schedule_scan
from .tune import tune


@autotvm.template("test_amm_conv2d")
def test_amm_conv2d(input_shape, kernel, strides, padding, subvec_len, output_channels, target):
    batch, channels, height, width = input_shape
    out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
    out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
    n = batch * out_height * out_width
    d = channels * kernel[0] * kernel[1]
    nc = d // subvec_len
    num_centroids = 16

    input_data = te.placeholder(
        input_shape,
        name="input_data", dtype="float32"
    )
    centroids = te.placeholder(
        (nc, num_centroids // 8, subvec_len, 8),
        name="centroids", dtype="float32"
    )
    lut_data = te.placeholder(
        (output_channels, nc, num_centroids),
        name="lut_data", dtype="int8"
    )
    scale = te.placeholder((1,), name="scale", dtype="float32")
    bias = te.placeholder((1,), name="bias", dtype="float32")

    cfg = autotvm.get_config()

    index_data = compute_fused_dist_argmin_fp32(
        cfg, input_data, centroids, kernel, strides, padding)
    output = compute_scan(
        cfg, index_data, lut_data,
        (batch, output_channels, out_height, out_width),
        scale, bias
    )

    s = te.create_schedule([output.op])
    s, _ = schedule_scan(cfg, s, output, target)
    s, _ = schedule_fused_dist_argmin_fp32(cfg, s, index_data)

    return s, [input_data, centroids, lut_data, scale, bias, output]


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
        # resnet18 for imagenet
        ((1, 64, 56, 56), (3, 3), (1, 1), (1, 1), 9, 64),
        ((1, 64, 56, 56), (3, 3), (2, 2), (1, 1), 9, 128),
        ((1, 64, 56, 56), (1, 1), (2, 2), (0, 0), 4, 128),
        ((1, 128, 28, 28), (3, 3), (1, 1), (1, 1), 9, 128),
        ((1, 128, 28, 28), (3, 3), (2, 2), (1, 1), 9, 256),
        ((1, 256, 14, 14), (3, 3), (1, 1), (1, 1), 9, 256),
        ((1, 128, 28, 28), (1, 1), (2, 2), (0, 0), 4, 256),
        ((1, 256, 14, 14), (3, 3), (2, 2), (1, 1), 9, 512),
        ((1, 256, 14, 14), (1, 1), (2, 2), (0, 0), 4, 512),
        ((1, 512, 7, 7), (3, 3), (1, 1), (1, 1), 9, 512),
    ]:
        func = tune(
            "test_amm_conv2d",
            test_amm_conv2d,
            args + (arch,),
            func_name="main",
            target=target,
            code_path="code",
            **remote_params
        )


if __name__ == "__main__":
    os.environ["TVM_NUM_THREADS"] = "1"

    verify_kernels("x86")
