from tvm import te, tir

from .argmin import compute_argmin, schedule_argmin
from .sdot import compute_sdot_gemm, schedule_sdot_gemm


def compute_fused_dist_argmin_int8(
    cfg,
    input_data, input_data_scale, input_data_zero_point,
    centroids, centroids_scale, centroids_zero_point,
    kernel=None, strides=None, padding=None
):
    # centroids: (nc, num_centroids, subvec_len)
    subvec_len = centroids.shape[2]
    subvec_len_padded = (subvec_len + 3) // 4 * 4

    if kernel is not None:
        batch, channels, height, width = input_data.shape
        out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
        out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
        n = batch * out_height * out_width
        n_padded = (n + 15) // 16 * 16
        d = channels * kernel[0] * kernel[1]
        nc = d // subvec_len
        num_centroids = 16

        def packed_x_h(i, j, k, l, m):
            j, k = j * 4 + l, k * 4 + m
            return ((j // out_width) % out_height) * strides[0] + (((i * subvec_len + k) // kernel[1]) % kernel[0]) - padding[0]

        def packed_x_w(i, j, k, l, m):
            j, k = j * 4 + l, k * 4 + m
            return (j % out_width) * strides[1] + ((i * subvec_len + k) % kernel[1]) - padding[1]

        # im2col
        packed_x = te.compute(
            (nc, n_padded // 4, subvec_len_padded // 4, 4, 4),
            lambda i, j, k, l, m:
            tir.if_then_else(
                tir.Not(tir.any(
                    j * 4 + l >= n,
                    k * 4 + m >= subvec_len,
                    packed_x_h(i, j, k, l, m) < 0,
                    packed_x_h(i, j, k, l, m) >= height,
                    packed_x_w(i, j, k, l, m) < 0,
                    packed_x_w(i, j, k, l, m) >= width
                )),
                input_data[
                    # batch
                    (j * 4 + l) // (out_height * out_width),
                    # channels
                    (i * subvec_len + k * 4 + m) // (kernel[0] * kernel[1]),
                    # height
                    packed_x_h(i, j, k, l, m),
                    # width
                    packed_x_w(i, j, k, l, m),
                ],
                input_data_zero_point[0]
            ),
            name="packed_x"
        )
    else:
        assert subvec_len == subvec_len_padded
        batch, num_samples, d = input_data.shape
        n_padded = n = batch * num_samples
        assert n % 16 == 0
        nc = d // subvec_len
        num_centroids = 16

        packed_x = te.compute(
            (nc, n // 4, subvec_len // 4, 4, 4),
            lambda i, j, k, l, m: input_data[
                (j * 4 + l) // num_samples,
                (j * 4 + l) % num_samples,
                i * subvec_len + k * 4 + m
            ],
            name="packed_x"
        )

    assert d % subvec_len == 0

    cfg.add_flop(nc * n_padded * subvec_len * num_centroids * 2)

    packed_y = te.compute(
        (nc, num_centroids // 4, subvec_len_padded // 4, 4, 4),
        lambda i, j, k, l, m:
        tir.if_then_else(
            k * 4 + m < subvec_len,
            centroids[i, j * 4 + l, k * 4 + m],
            centroids_zero_point[i]
        ),
        name="packed_y"
    )

    # compute sigma y^2
    y2_k = te.reduce_axis((0, subvec_len), name="y2_k")
    y2 = te.compute(
        (nc, num_centroids),
        lambda i, j: te.sum(
            te.power(centroids[i, j, y2_k].astype("float32"), 2),
            axis=y2_k
        ),
        name="y2"
    )

    y1_k = te.reduce_axis((0, subvec_len), name="y1_k")
    y1 = te.compute(
        (nc, num_centroids),
        lambda i, j: te.sum(
            centroids[i, j, y1_k].astype("float32"),
            axis=y1_k
        ),
        name="y1"
    )

    # compute x*y
    xy = compute_sdot_gemm(packed_y, packed_x)

    cfg.define_knob("blk_size", [4, 8, 16])
    blk_size = cfg["blk_size"].val

    dis = te.compute(
        (nc, n_padded // blk_size, num_centroids, blk_size),
        lambda i, j, k, l:
        - 2 * xy[i, k // 4, (j * blk_size + l) // 4, k % 4, (j * blk_size + l) % 4].astype("float32") +
        centroids_scale[i] / input_data_scale[0] * y2[i, k] +
        2 * (input_data_zero_point[0].astype("float32") -
             centroids_scale[i] / input_data_scale[0] * centroids_zero_point[i]) * y1[i, k],
        name="dis"
    )

    min_dis_idx = compute_argmin(cfg, dis)

    output = te.compute(
        (nc, n_padded),
        lambda i, j: min_dis_idx[i, j], name="output"
    )

    return output


def schedule_fused_dist_argmin_int8(cfg, s, output):
    min_dis_idx = output.op.input_tensors[0]
    min_dis_idx_k = s[min_dis_idx].op.reduce_axis[0]

    idx = min_dis_idx.op.input_tensors[0]
    dis = min_dis_idx.op.input_tensors[1]
    xy = dis.op.input_tensors[0]
    xy_k = s[xy].op.reduce_axis[0]

    y2 = dis.op.input_tensors[3]
    y2_k = s[y2].op.reduce_axis[0]

    y1 = dis.op.input_tensors[6]
    y1_k = s[y1].op.reduce_axis[0]

    packed_y = xy.op.input_tensors[0]
    packed_x = xy.op.input_tensors[1]
    input_data = packed_x.op.input_tensors[0]

    use_im2col = len(input_data.shape) == 4

    # schedule for im2col
    s[packed_x].unroll(s[packed_x].op.axis[-1])
    s[packed_x].unroll(s[packed_x].op.axis[-3])
    if use_im2col:
        s[packed_x].unroll(s[packed_x].op.axis[-2])
    else:
        s[packed_x].vectorize(s[packed_x].op.axis[-2])

    s, _ = schedule_sdot_gemm(s, xy)

    s, _ = schedule_argmin(cfg, s, min_dis_idx)
    s[dis].vectorize(s[dis].op.axis[-1])
    blk_size = cfg["blk_size"].val
    output_xo, output_xi = s[output].split(
        s[output].op.axis[-1], blk_size)
    s[output].vectorize(output_xi)
    s[min_dis_idx].compute_at(s[output], output_xo)

    return s
