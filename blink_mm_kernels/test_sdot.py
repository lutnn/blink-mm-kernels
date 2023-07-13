from tvm import autotvm, te, tir

from blink_mm_kernels.tune import tune
from .sdot import compute_sdot_gemm, schedule_sdot_gemm


@autotvm.template("test_sdot_gemm")
def test_sdot_gemm(batch, m, n, k):
    assert m % 4 == 0 and n % 4 == 0 and k % 4 == 0
    a = te.placeholder((batch, m // 4, k // 4, 4, 4), dtype="int8", name="a")
    b = te.placeholder((batch, n // 4, k // 4, 4, 4), dtype="int8", name="b")
    c = compute_sdot_gemm(a, b)
    cfg = autotvm.get_config()

    s = te.create_schedule([c.op])
    s, _ = schedule_sdot_gemm(s, c)

    return s, [a, b, c]


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
        (4, 512, 512, 512)
    ]:
        func = tune(
            "test_sdot_gemm",
            test_sdot_gemm,
            args, func_name="main",
            target=target,
            code_path="code",
            **remote_params
        )


if __name__ == "__main__":
    verify_kernels("arm")
