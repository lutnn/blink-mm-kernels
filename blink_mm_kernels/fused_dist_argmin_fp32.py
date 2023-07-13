from tvm import tir, te

from .argmin import compute_argmin, schedule_argmin


def compute_fused_dist_argmin_fp32(
    cfg, input_data, centroids,
    kernel=None, strides=None, padding=None
):
    cfg.define_knob("blk_size", [8, 16])
    blk_size = cfg["blk_size"].val

    if kernel is not None:
        subvec_len = centroids.shape[2]
        # centroids: (nc, num_centroids // 8, subvec_len, 8)
        batch, channels, height, width = input_data.shape
        out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
        out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
        n = batch * out_height * out_width
        n_padded = (n + 15) // 16 * 16
        d = channels * kernel[0] * kernel[1]
        nc = d // subvec_len
        num_centroids = 16

        def packed_x_h(i, j, k, l):
            return (((j * blk_size + l) // out_width) % out_height) * strides[0] + (((i * subvec_len + k) // kernel[1]) % kernel[0]) - padding[0]

        def packed_x_w(i, j, k, l):
            return ((j * blk_size + l) % out_width) * strides[1] + ((i * subvec_len + k) % kernel[1]) - padding[1]

        # im2col
        packed_x = te.compute(
            (nc, n_padded // blk_size, subvec_len, blk_size),
            lambda i, j, k, l:
            tir.if_then_else(
                tir.any(
                    j * blk_size + l >= n,
                    packed_x_h(i, j, k, l) < 0,
                    packed_x_h(i, j, k, l) >= height,
                    packed_x_w(i, j, k, l) < 0,
                    packed_x_w(i, j, k, l) >= width
                ),
                0,
                input_data[
                    # batch
                    (j * blk_size + l) // (out_height * out_width),
                    # channels
                    (i * subvec_len + k) // (kernel[0] * kernel[1]),
                    # height
                    packed_x_h(i, j, k, l),
                    # width
                    packed_x_w(i, j, k, l),
                ]
            ),
            name="packed_x"
        )
    else:
        subvec_len = centroids.shape[2]
        # centroids: (nc, num_centroids // 8, subvec_len, 8)
        batch, num_samples, d = input_data.shape
        n_padded = n = batch * num_samples
        assert n % 16 == 0
        nc = d // subvec_len
        num_centroids = 16

        packed_x = te.compute(
            (nc, n // blk_size, subvec_len, blk_size),
            lambda i, j, k, l: input_data[
                (j * blk_size + l) // num_samples,
                (j * blk_size + l) % num_samples,
                i * subvec_len + k
            ],
            name="packed_x"
        )

    assert d % subvec_len == 0

    cfg.add_flop(nc * n_padded * subvec_len * num_centroids * 2)

    # compute sigma y^2
    y2_k = te.reduce_axis((0, subvec_len), name="y2_k")
    y2 = te.compute(
        (nc, num_centroids),
        lambda i, j: te.sum(
            te.power(centroids[i, j // 8, y2_k, j % 8], 2),
            axis=y2_k
        ),
        name="y2"
    )

    # compute x*y
    xy_k = te.reduce_axis((0, subvec_len), name="xy_k")
    xy = te.compute(
        (nc, num_centroids, n_padded),
        lambda i, j, k: te.sum(
            packed_x[i, k // blk_size, xy_k, k % blk_size] *
            centroids[i, j // 8, xy_k, j % 8],
            axis=xy_k
        ),
        name="xy"
    )

    dis = te.compute(
        (nc, n_padded // blk_size, num_centroids, blk_size),
        lambda i, j, k, l: y2[i, k] - 2 * xy[i, k, j * blk_size + l],
        name="dis"
    )

    min_dis_idx = compute_argmin(cfg, dis)

    output = te.compute(
        (nc, n_padded),
        lambda i, j: min_dis_idx[i, j], name="output"
    )

    return output


def schedule_fused_dist_argmin_fp32(cfg, s, output):
    min_dis_idx = output.op.input_tensors[0]
    min_dis_idx_k = s[min_dis_idx].op.reduce_axis[0]

    idx = min_dis_idx.op.input_tensors[0]
    dis = min_dis_idx.op.input_tensors[1]
    y2 = dis.op.input_tensors[0]
    y2_k = s[y2].op.reduce_axis[0]

    xy = dis.op.input_tensors[1]
    xy_k = s[xy].op.reduce_axis[0]

    packed_x = xy.op.input_tensors[0]
    centroids = xy.op.input_tensors[1]
    input_data = packed_x.op.input_tensors[0]

    use_im2col = len(input_data.shape) == 4

    # schedule for im2col
    s[packed_x].unroll(s[packed_x].op.axis[-1])
    if use_im2col:
        s[packed_x].unroll(s[packed_x].op.axis[-2])
    else:
        s[packed_x].vectorize(s[packed_x].op.axis[-2])

    # schedule for y^2
    y2_xo, y2_xi = s[y2].split(s[y2].op.axis[1], 8)
    s[y2].reorder(y2_k, y2_xo)
    s[y2].unroll(y2_xo)
    s[y2].vectorize(y2_xi)
    s[y2].unroll(y2_k)

    # schedule for x*y
    _, _, xy_xi, xy_yi = s[xy].tile(
        s[xy].op.axis[1], s[xy].op.axis[2], 8, 8
    )
    s[xy].reorder(xy_k, xy_yi, xy_xi)
    s[xy].unroll(xy_k)
    s[xy].unroll(xy_xi)
    s[xy].vectorize(xy_yi)

    # schedule for distance
    dis_xo, _ = s[dis].split(s[dis].op.axis[2], 8)
    s[xy].compute_at(s[dis], dis_xo)
    s[dis].vectorize(s[dis].op.axis[-1])

    # schedule for min distance
    s, _ = schedule_argmin(cfg, s, min_dis_idx)
    s[packed_x].compute_at(s[min_dis_idx], s[min_dis_idx].op.axis[0])

    # schedule for output
    output_xo, output_xi = s[output].split(s[output].op.axis[-1], 8)
    s[min_dis_idx].compute_at(s[output], output_xo)
    s[y2].compute_at(s[output], s[output].op.axis[0])
    s[output].vectorize(output_xi)

    # parallel
    s[output].parallel(s[output].op.axis[0])

    return s, input_data
