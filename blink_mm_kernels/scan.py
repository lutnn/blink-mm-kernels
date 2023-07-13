import tvm
from tvm import te

from blink_mm_kernels.utils import factors


def pshufb_int8(dtype):
    index_data = te.placeholder((1, 16), dtype, name="A")
    lut_data = te.placeholder((1, 16), dtype, name="B")
    dtype_vec = dtype + "x16"

    result_data = te.compute(
        (1, 16),
        lambda i, j: tvm.tir.if_then_else(
            tvm.tir.all(
                tvm.tir.const(0, index_data.dtype) <= index_data[i, j],
                index_data[i, j] < tvm.tir.const(16, index_data.dtype)
            ),
            lut_data[i, index_data[i, j].astype("int32")],
            tvm.tir.const(0, lut_data.dtype)),
        name="result_data"
    )

    a_buffer = tvm.tir.decl_buffer(
        index_data.shape, dtype, name="a_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    b_buffer = tvm.tir.decl_buffer(
        lut_data.shape, dtype, name="b_buffer", offset_factor=1, strides=[te.var("sb"), 1]
    )
    c_buffer = tvm.tir.decl_buffer(
        result_data.shape, dtype, name="c_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    llvm_intrisics_pshuf_b = "llvm.x86.ssse3.pshuf.b.128"

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore([0, 0], tvm.tir.const(0, "int8x16")))
                return ib.get()
            vec_a = ins[0].vload([0, 0], dtype_vec)
            vec_b = ins[1].vload([0, 0], dtype_vec)

            lookup = tvm.tir.call_llvm_intrin(
                "int8x16",
                llvm_intrisics_pshuf_b,
                tvm.tir.const(2, "uint32"),
                vec_b,
                vec_a)

            ib.emit(outs[0].vstore([0, 0], lookup))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        result_data.op,
        _intrin_func,
        binds={
            index_data: a_buffer,
            lut_data: b_buffer,
            result_data: c_buffer
        },
        default_buffer_params=buffer_params,
    )


def tbl_int8(dtype):
    A = te.placeholder((1, 16), dtype, name="A")
    B = te.placeholder((1, 16), dtype, name="B")
    dtype_vec = dtype + "x16"

    C = te.compute(
        (1, 16),
        lambda i, j: tvm.tir.if_then_else(
            tvm.tir.all(
                tvm.tir.const(0, A.dtype) <= B[i, j],
                B[i, j] < tvm.tir.const(16, A.dtype)
            ),
            A[0, B[0, j].astype("int32")],
            tvm.tir.const(0, A.dtype)).astype("int8"),
        name="C"
    )

    a_buffer = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="a_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    b_buffer = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="b_buffer", offset_factor=1, strides=[te.var("sb"), 1]
    )
    c_buffer = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="c_buffer", offset_factor=1, strides=[te.var("sc"), 1],
    )

    llvm_intrisics_vqtbl1q = "llvm.aarch64.neon.tbl1"

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore([0, 0], tvm.tir.const(0, "int8")))
                return ib.get()
            vec_a = ins[0].vload([0, 0], dtype_vec)
            vec_b = ins[1].vload([0, 0], dtype_vec)

            lookup = tvm.tir.call_llvm_intrin(
                "int8x16",
                llvm_intrisics_vqtbl1q,
                tvm.tir.const(2, "uint32"),
                vec_b,
                vec_a
            )

            vec_int8 = lookup
            ib.emit(outs[0].vstore([0, 0], vec_int8))

            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: a_buffer, B: b_buffer, C: c_buffer},
        default_buffer_params=buffer_params,
    )


def compute_scan(cfg, index_data, lut_data, output_shape, scale, bias):
    nc, n_padded = index_data.shape
    if len(output_shape) == 4:
        # nchw
        batch_size, out_features, height, width = output_shape
    else:
        batch_size, num_samples, out_features = output_shape

    cfg.define_knob(
        "nc_blk_size",
        [i for i in factors(int(nc)) if i <= 128]
    )
    nc_blk_size = cfg["nc_blk_size"].val

    cfg.define_knob(
        "n_blk_size",
        [i for i in factors(int(n_padded) // 16) if i <= 8]
    )
    n_blk_size = cfg["n_blk_size"].val

    cfg.add_flop(nc * n_padded * out_features)

    index_data_permuted = te.compute(
        (n_padded // (n_blk_size * 16), nc, n_blk_size, 16),
        lambda i, j, k, l: index_data[j, i * (n_blk_size * 16) + k * 16 + l],
        name="index_data_permuted"
    )

    lookup_result = te.compute(
        (
            n_padded // (n_blk_size * 16),  # i
            out_features,                # j
            nc // nc_blk_size,              # k
            nc_blk_size,                    # l
            n_blk_size,                     # m
            16                              # n
        ),
        lambda i, j, k, l, m, n:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    tvm.tir.const(0, index_data.dtype)
                    <= index_data_permuted[i, k * nc_blk_size + l, m, n],
                    index_data_permuted[i, k * nc_blk_size + l, m, n]
                    < tvm.tir.const(16, index_data.dtype)
                ),
                lut_data[
                    j, k * nc_blk_size + l,
                    index_data_permuted[i, k * nc_blk_size + l, m, n]
                    .astype("int32")
                ],
                tvm.tir.const(0, lut_data.dtype)).astype("int8"),
        name="lookup_result"
    )

    l = te.reduce_axis((0, nc_blk_size), name="l")
    output_data_int16 = te.compute(
        (
            n_padded // (n_blk_size * 16),
            out_features,
            nc // nc_blk_size,
            n_blk_size,
            16
        ),
        lambda i, j, k, m, n:
            tvm.tir.sum(
                lookup_result[i, j, k, l, m, n].astype("int16"),
                axis=l,
        ),
        name="output_data_int16"
    )

    k = te.reduce_axis((0, nc // nc_blk_size), name="k")
    output_data_int32 = te.compute(
        (n_padded // (n_blk_size * 16), out_features, n_blk_size, 16),
        lambda i, j, l, m:
            tvm.tir.sum(
                output_data_int16[i, j, k, l, m].astype("int32"),
                axis=k,
        ),
        name="output_data_int32"
    )

    if len(output_shape) == 4:
        # nchw
        output_data_transposed = te.compute(
            (out_features, n_padded // (n_blk_size * 16), n_blk_size, 16),
            lambda i, j, k, l: output_data_int32[j, i, k, l]
            .astype("float") * scale[0].astype("float") + bias[i].astype("float"),
            name="output_data_transposed"
        )

        output_data = te.compute(
            output_shape,
            lambda i, j, k, l:
            output_data_transposed[
                j,
                (i * height * width + k * width + l) // (n_blk_size * 16),
                (i * height * width + k * width + l) // 16 % n_blk_size,
                (i * height * width + k * width + l) % 16
            ],
            name="output_data",
            tag="amm_conv2d"
        )
    else:
        # batch, rows, features
        output_data = te.compute(
            output_shape,
            lambda i, j, k:
            output_data_int32[
                (i * num_samples + j) // (n_blk_size * 16),
                k,
                (i * num_samples + j) // 16 % n_blk_size,
                (i * num_samples + j) % 16
            ].astype("float") * scale[0].astype("float") + bias[k].astype("float"),
            name="output_data",
            tag="amm_linear"
        )

    return output_data


def schedule_scan(cfg, s, output_data, target):
    assert target in ["arm", "x86", "x86_avx512"]

    if len(output_data.shape) == 4:
        output_data_transposed = output_data.op.input_tensors[0]
        output_data_int32 = output_data_transposed.op.input_tensors[0]
    else:
        output_data_int32 = output_data.op.input_tensors[0]
    output_data_int16 = output_data_int32.op.input_tensors[0]
    lookup_result = output_data_int16.op.input_tensors[0]
    index_data_permuted = lookup_result.op.input_tensors[0]
    lut_data = lookup_result.op.input_tensors[1]
    index_data = index_data_permuted.op.input_tensors[0]

    s[index_data_permuted].unroll(s[index_data_permuted].op.axis[2])
    s[index_data_permuted].vectorize(s[index_data_permuted].op.axis[3])

    s[lookup_result].unroll(s[lookup_result].op.axis[3])
    s[lookup_result].unroll(s[lookup_result].op.axis[4])
    if target == "arm":
        tbl = tbl_int8
    elif target in ["x86", "x86_avx512"]:
        tbl = pshufb_int8
    s[lookup_result].tensorize(
        s[lookup_result].op.axis[5],
        tbl(lookup_result.dtype)
    )

    _, li = s[output_data_int16].split(
        s[output_data_int16].op.reduce_axis[0], factor=4)
    output_data_int16_rf = s.rfactor(output_data_int16, li)
    s[output_data_int16_rf].reorder(
        s[output_data_int16_rf].op.axis[1],
        s[output_data_int16_rf].op.axis[2],
        s[output_data_int16_rf].op.axis[3],
        s[output_data_int16_rf].op.axis[4],
        s[output_data_int16_rf].op.axis[5],
        s[output_data_int16_rf].op.reduce_axis[0],
        s[output_data_int16_rf].op.axis[0],
    )
    s[output_data_int16_rf].unroll(s[output_data_int16_rf].op.axis[4])
    s[output_data_int16_rf].unroll(s[output_data_int16_rf].op.axis[0])
    s[output_data_int16_rf].unroll(s[output_data_int16_rf].op.reduce_axis[0])
    s[output_data_int16_rf].vectorize(s[output_data_int16_rf].op.axis[5])

    s[output_data_int16].unroll(s[output_data_int16].op.axis[3])
    s[output_data_int16].vectorize(s[output_data_int16].op.axis[4])

    s[output_data_int32].unroll(s[output_data_int32].op.reduce_axis[0])
    s[output_data_int32].unroll(s[output_data_int32].op.axis[2])
    s[output_data_int32].vectorize(s[output_data_int32].op.axis[3])

    if len(output_data.shape) == 4:
        s[output_data_transposed].vectorize(
            s[output_data_transposed].op.axis[3])
        s[output_data].vectorize(s[output_data].split(s[output_data].fuse(
            *s[output_data].op.axis
        ), factor=4)[1])
    else:
        s[output_data].fuse(*s[output_data].op.axis)

    s[index_data_permuted].compute_at(
        s[output_data_int32], s[output_data_int32].op.axis[0])
    s[lookup_result].compute_at(
        s[output_data_int16_rf], s[output_data_int16_rf].op.axis[3])
    s[output_data_int16_rf].compute_at(
        s[output_data_int16], s[output_data_int16].op.axis[2])
    s[output_data_int16].compute_at(
        s[output_data_int32], s[output_data_int32].op.axis[1])

    # parallel
    cfg.define_knob("parallel_scan_outer_axis", [False, True])
    if cfg["parallel_scan_outer_axis"].val:
        s[output_data_int32].parallel(s[output_data_int32].op.axis[0])
    else:
        s[index_data_permuted].parallel(s[index_data_permuted].op.axis[1])
        s[output_data_int32].parallel(s[output_data_int32].op.axis[1])
    if len(output_data.shape) == 4:
        s[output_data_transposed].parallel(
            s[output_data_transposed].op.axis[0])
        s[output_data].parallel(s[output_data].leaf_iter_vars[0])
    else:
        s[output_data].parallel(s[output_data].leaf_iter_vars[0])

    return s, index_data
