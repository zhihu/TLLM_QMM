# adapted from: https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_marlin.py#L194
# Copyright (c) 2023 潘其威(William)

import torch

def unpack_4bit_to_32bit_signed(qweight, qzeros):
    # Unpack 4-bit values and interpret them as signed integers
    unpacked_weights = torch.zeros(
        (qweight.shape[0] * 8, qweight.shape[1]),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for row in range(unpacked_weights.shape[0]):
        i = row % 8
        unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_weights, unpacked_zeros + 1


def unpack_qzeros(qzeros):
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_zeros + 1


def dequantize_weight(qweight, qzeros, scales):
    unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight
