from tllm_qmm import W4A8_FP8
from tllm_qmm import WeightOnlyGroupwiseQuantGEMM
from tllm_qmm.gptq_utils import dequantize_weight
import torch

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale_orig = (finfo.max - 0) / x.abs().max()
    scale = scale_orig.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()


input_rows = 8192
in_features = 49152
out_features = 8192
w_bit = 4
group_size = 128

MAX_INT32 = 0x7FFFFFFF
MIN_INT32 = -MAX_INT32 - 1

qweight = torch.randint(
    -(2**31),
    2**31,
    # MIN_INT32,
    # MAX_INT32,
    # (in_features, out_features // (32 // w_bit)),
    (in_features // (32 // w_bit), out_features),
    dtype=torch.int32,
    device="cuda",
)
inputs = torch.randn((input_rows, in_features), dtype=torch.half, device="cuda")
# inputs = torch.randn([input_rows], device="cuda").to(torch.float16).diag()
# inputs = torch.ones([input_rows], device="cuda").to(torch.float16).diag()
bias = torch.randn(1, out_features, dtype=torch.half, device="cuda")
has_bias = True

qzeros = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features // group_size, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cuda",
)

scales = (
    torch.randn(
        (in_features // group_size, out_features),
        dtype=torch.float16,
        device="cuda",
    )
    / 100
)

xinput, fp8_alpha = to_float8(inputs)
pre_quant_scale = fp8_alpha.reciprocal().half().repeat(1, in_features)
print(inputs.abs().max().item(), fp8_alpha.item())
print(scales.abs().max().item())

with torch.no_grad():

    tllm_matmul = WeightOnlyGroupwiseQuantGEMM(
        (W4A8_FP8 | 1) if has_bias else W4A8_FP8,
        1,
        input_rows,
        in_features,
        out_features,
        group_size,
        False,
    )
    qiweight, izeros = tllm_matmul.preprocess_weights_gptq(qweight, qzeros, scales)
    for i in range(10):
        cuda_out = tllm_matmul.forward(
            xinput,
            # pre_quant_scale.repeat(input_rows, 1),
            pre_quant_scale,
            qiweight.view(torch.half),
            scales,
            izeros,
            bias,
            fp8_alpha,
        )
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    cuda_out = tllm_matmul.forward(
        xinput,
        pre_quant_scale,
        qiweight.view(torch.half),
        scales,
        izeros,
        bias,
        fp8_alpha,
    )
    stop_event.record()
    torch.cuda.synchronize()
    print(f"cuda time: {start_event.elapsed_time(stop_event)}ms")
    """
    cuda_out = woq_groupwise_matmul(
        inputs,
        pre_quant_scale.repeat(input_rows, 1),
        qiweight.view(torch.half),
        scales,
        izeros,
        fp8_alpha,
        14,
    )
    """

    # awq dequantize
    torch_dq = dequantize_weight(qweight, qzeros, scales)
    tinputs = (
        torch.mul(inputs, pre_quant_scale.repeat(input_rows, 1)) * fp8_alpha.half()
    )
    torch_out = torch.matmul(
        inputs,
        torch_dq,
    )
    if has_bias:
        torch_out += bias

    out_cm = torch_dq.t().contiguous()

    # quant to fp8
    w_f8, w_inv_s = to_float8(out_cm.t())
    print("w_amax:", out_cm.abs().max(), "w_inv_s:", w_inv_s.item())
    # w_f8, w_inv_s = to_float8(out_cm.t())

    # perform the float8 matmul
    scaled_torch_out, _ = torch._scaled_mm(
        xinput, w_f8, out_dtype=torch.float16, scale_a=fp8_alpha, scale_b=w_inv_s
    )
    if has_bias:
        scaled_torch_out += bias
    # torch_out = torch.matmul(inputs, unpack(iweight, direction="column").to(torch.half))
    print(cuda_out.shape, torch_out.shape)
    print("torch_dq:", torch_dq)
    print("cuda_out:", cuda_out)
    print("scaled_torch_out:", scaled_torch_out)
    print("torch_out:", torch_out)
    print((cuda_out - torch_out).abs())
    print((scaled_torch_out - torch_out).abs())
    print((scaled_torch_out - cuda_out).abs())
    loss1 = ((cuda_out - torch_out) / 1).float().pow(2).mean().item()
    loss2 = ((scaled_torch_out - torch_out) / 1).float().pow(2).mean().item()
    print("mse1:", loss1)
    print("mse2:", loss2)
    # assert loss < 1e-1, loss
    woq_assert_near_eq(torch_out, cuda_out, 2)
    woq_assert_near_eq(torch_out, scaled_torch_out, 2)
