from tvm import tir, te
import tvm.topi


def compute_im2col(cfg, input_data, kernel, strides, padding, pad_value):
    batch, channels, height, width = input_data.shape
    out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
    out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
    n = batch * out_height * out_width
    d = channels * kernel[0] * kernel[1]

    input_data_padded = tvm.topi.nn.pad(
        input_data,
        (0, 0, padding[0], padding[1]),
        (0, 0, padding[0], padding[1]),
        pad_value=tir.Cast(input_data.dtype, pad_value),
        name="input_data_padded"
    )

    input_data_im2col = te.compute(
        (n, d),
        lambda i, j: input_data_padded[
            i // (out_height * out_width),
            j // (kernel[0] * kernel[1]),
            strides[0] * (i % (out_height * out_width) // out_width) +
            ((j % (kernel[0] * kernel[1])) // kernel[1]),
            strides[1] * (i % (out_height * out_width) % out_width) +
            ((j % (kernel[0] * kernel[1])) % kernel[1])
        ],
        name="input_data_im2col",
    )

    return input_data_im2col


def schedule_im2col(cfg, s, output):
    x, y = output.op.axis
    yo, yi = s[output].split(y, 16)

    s[output].unroll(yo)
    s[output].vectorize(yi)
    s[output].parallel(x)

    input_data = output.op.input_tensors[0].op.input_tensors[0]

    return s, [input_data]


def compute_pack(cfg, im2col_data, subvec_len, n_blk_size):
    n, d = im2col_data.shape
    assert d % subvec_len == 0
    n_padded = (n + 15) // 16 * 16
    im2col_data_padded = tvm.topi.nn.pad(
        im2col_data,
        (0, 0), (n_padded - n, 0),
        pad_value=tir.Cast(im2col_data.dtype, 0),
        name="im2col_data_padded"
    )

    data_packed = te.compute(
        (d // subvec_len, n_padded // n_blk_size, subvec_len, n_blk_size),
        lambda i, j, k, l:
        im2col_data_padded[j * n_blk_size + l, i * subvec_len + k],
        name="data_packed"
    )

    return data_packed


def schedule_pack(cfg, s, data_packed):
    s[data_packed].vectorize(s[data_packed].op.axis[3])
    s[data_packed].unroll(s[data_packed].op.axis[2])
    s[data_packed].parallel(s[data_packed].op.axis[0])

    im2col_data = data_packed.op.input_tensors[0].op.input_tensors[0]

    return s, [im2col_data]
