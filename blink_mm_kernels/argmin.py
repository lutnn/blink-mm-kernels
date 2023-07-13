from tvm import tir, te


def get_argmin_reducer():
    def fcombine(x, y):
        lhs = tir.Select((x[1] < y[1]), x[0], y[0])
        rhs = tir.Select((x[1] < y[1]), x[1], y[1])
        return lhs, rhs

    def fidentity(t0, t1):
        return tir.const(-1, t0), tir.max_value(t1)

    return tir.comm_reducer(
        fcombine,
        fidentity,
        name='argmin'
    )


def compute_argmin(cfg, dis):
    nc, n_padded_div_blk_size, num_centroids, blk_size = dis.shape
    n_padded = n_padded_div_blk_size * blk_size

    cfg.add_flop(nc * n_padded * num_centroids)

    argmin = get_argmin_reducer()

    idx = te.compute(
        (num_centroids,),
        lambda i: tir.Cast("int8", i),
        name="idx"
    )

    min_dis_idx_k = te.reduce_axis((0, num_centroids), name="min_dis_idx_k")
    min_dis_idx, min_dis_val = te.compute(
        (nc, n_padded),
        lambda i, j: argmin(
            (idx[min_dis_idx_k], dis[i, j // blk_size, min_dis_idx_k, j % blk_size]),
            axis=min_dis_idx_k
        ),
        name="min_dis_idx"
    )

    return min_dis_idx


def schedule_argmin(cfg, s, min_dis_idx):
    min_dis_idx_k = s[min_dis_idx].op.reduce_axis[0]
    dis = s[min_dis_idx].op.input_tensors[1]

    _, min_dis_idx_ki = s[min_dis_idx].split(min_dis_idx_k, factor=4)
    min_dis_idx_rf, _ = s.rfactor(min_dis_idx, min_dis_idx_ki, 2)
    min_dis_idx_xo, min_dis_idx_xi = s[min_dis_idx].split(
        s[min_dis_idx].op.axis[1], 8)
    s[min_dis_idx].vectorize(min_dis_idx_xi)
    s[min_dis_idx_rf].compute_at(s[min_dis_idx], min_dis_idx_xi)
    s[dis].compute_at(s[min_dis_idx], min_dis_idx_xo)

    s[min_dis_idx_rf].unroll(s[min_dis_idx_rf].op.axis[-1])
    s[min_dis_idx_rf].reorder(
        s[min_dis_idx_rf].op.reduce_axis[0], s[min_dis_idx_rf].op.axis[-1])
    s[min_dis_idx].unroll(s[min_dis_idx].op.reduce_axis[0])

    return s, [dis]
