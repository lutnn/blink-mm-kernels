from tvm import autotvm, te

from .argmin import compute_argmin, schedule_argmin
from .tune import tune


@autotvm.template("test_argmin")
def test_argmin(nc, n_padded, blk_size):
    num_centroids = 16

    dis = te.placeholder(
        (nc, n_padded // blk_size, num_centroids, blk_size),
        name="dis", dtype="float32"
    )

    cfg = autotvm.get_config()

    output = compute_argmin(cfg, dis)

    s = te.create_schedule([output.op])
    s, _ = schedule_argmin(cfg, s, output)

    return s, [dis, output]


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
        (64, 3136, 8),
    ]:
        func = tune(
            "test_argmin",
            test_argmin,
            args, func_name="main",
            target=target,
            code_path="code",
            **remote_params
        )


if __name__ == "__main__":
    verify_kernels("x86")
