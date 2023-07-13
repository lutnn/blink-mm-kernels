import tvm
import tvm.testing
from tvm import te
from tvm.topi import nn

from tvm.topi.arm_cpu.tensor_intrin import gemm_acc_4x4_int8_int8_int32


def compute_sdot_gemm(a, b):
    # a: (batch, m_div_4, k_div_4, 4, 4)
    # b: (batch, n_div_4, k_div_4, 4, 4)

    batch, m_div_4, k_div_4, _, _ = a.shape
    _, n_div_4, _, _, _ = b.shape

    c_k = te.reduce_axis((0, k_div_4 * 4), "c_k")

    return te.compute(
        (batch, m_div_4, n_div_4, 4, 4),
        lambda i, j, k, l, m: te.sum(
            a[i, j, c_k // 4, l, c_k % 4].astype("int32") *
            b[i, k, c_k // 4, m, c_k % 4].astype("int32"),
            axis=c_k,
        ),
        name="c",
    )


def schedule_sdot_gemm(s, c):
    a = c.op.input_tensors[0]
    b = c.op.input_tensors[1]

    xi, yi = s[c].op.axis[-2:]
    ko, ki = s[c].split(s[c].op.reduce_axis[0], 4)

    s[c].reorder(
        ko, xi, yi, ki
    )

    gemm_acc = gemm_acc_4x4_int8_int8_int32(a.dtype)
    s[c].tensorize(xi, gemm_acc)

    return s, [a, b]
