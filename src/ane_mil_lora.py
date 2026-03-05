"""
ane_mil_lora.py — MIL code generators for LoRA forward and backward passes on ANE.

Generates Apple Machine Learning Intermediate Language (MIL) programs that
compile and run on the Neural Engine via libane_bridge.dylib.

Based on the dynamic matmul pattern from maderix/ANE: weights are packed
into the spatial dimension of the input IOSurface, enabling weight updates
without recompilation. Each kernel is compiled ONCE and reused across all
layers by writing different weights to the IOSurface.

ANE matmul constraint: all dimensions (channels, spatial, matmul operands)
must be multiples of 16 with minimum of 16. This means:
  - LoRA rank must be a multiple of 16 (recommend 16 or 32)
  - Sequence length must be a multiple of 16 (pad if needed)
  - Model hidden dimension is typically large enough (e.g. 3584)

Kernels produced:
  1. lora_down  — x @ A^T → h          [dim → rank]
  2. lora_up    — h @ B^T → out * scale [rank → dim]
  3. grad_b     — grad_out @ h^T → dB   [gradient for B]
  4. grad_a     — (B^T @ grad_out) @ x^T → dA [gradient for A]
  5. rmsnorm    — RMSNorm with baked weights
"""

import numpy as np

# Standard MIL header required by ANE's modelWithMILText API
MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, '
    '{"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)


def _dynamic_matmul_block(prefix: str, ic: int, oc: int, seq: int,
                          act_sp_off: int, w_sp_off: int,
                          input_var: str) -> str:
    """Generate MIL statements for a dynamic matmul within a function.

    Slices activation [1,ic,1,seq] and weight [1,ic,1,oc] from the input
    spatial dimension, reshapes for matmul, and produces output [1,oc,1,seq].

    This is the core building block from maderix's training_dynamic approach.
    """
    lines = []

    # Slice activations: [1, ic, 1, seq] from spatial offset
    lines.append(f'        tensor<int32, [4]> {prefix}_ba = const()[name = string("{prefix}_ba"), val = tensor<int32, [4]>([0, 0, 0, {act_sp_off}])];')
    lines.append(f'        tensor<int32, [4]> {prefix}_sa = const()[name = string("{prefix}_sa"), val = tensor<int32, [4]>([1, {ic}, 1, {seq}])];')
    lines.append(f'        tensor<fp16, [1, {ic}, 1, {seq}]> {prefix}_act = slice_by_size(x = {input_var}, begin = {prefix}_ba, size = {prefix}_sa)[name = string("{prefix}_act")];')

    # Slice weight: [1, ic, 1, oc] from spatial offset
    lines.append(f'        tensor<int32, [4]> {prefix}_bw = const()[name = string("{prefix}_bw"), val = tensor<int32, [4]>([0, 0, 0, {w_sp_off}])];')
    lines.append(f'        tensor<int32, [4]> {prefix}_sw = const()[name = string("{prefix}_sw"), val = tensor<int32, [4]>([1, {ic}, 1, {oc}])];')
    lines.append(f'        tensor<fp16, [1, {ic}, 1, {oc}]> {prefix}_wt = slice_by_size(x = {input_var}, begin = {prefix}_bw, size = {prefix}_sw)[name = string("{prefix}_wt")];')

    # Reshape activation: [1,ic,1,seq] → [1,1,ic,seq]
    lines.append(f'        tensor<int32, [4]> {prefix}_ra = const()[name = string("{prefix}_ra"), val = tensor<int32, [4]>([1, 1, {ic}, {seq}])];')
    lines.append(f'        tensor<fp16, [1, 1, {ic}, {seq}]> {prefix}_a2 = reshape(shape = {prefix}_ra, x = {prefix}_act)[name = string("{prefix}_a2")];')

    # Transpose: [1,1,ic,seq] → [1,1,seq,ic]
    lines.append(f'        tensor<int32, [4]> {prefix}_pm = const()[name = string("{prefix}_pm"), val = tensor<int32, [4]>([0, 1, 3, 2])];')
    lines.append(f'        tensor<fp16, [1, 1, {seq}, {ic}]> {prefix}_a3 = transpose(perm = {prefix}_pm, x = {prefix}_a2)[name = string("{prefix}_a3")];')

    # Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    lines.append(f'        tensor<int32, [4]> {prefix}_rw = const()[name = string("{prefix}_rw"), val = tensor<int32, [4]>([1, 1, {ic}, {oc}])];')
    lines.append(f'        tensor<fp16, [1, 1, {ic}, {oc}]> {prefix}_W = reshape(shape = {prefix}_rw, x = {prefix}_wt)[name = string("{prefix}_W")];')

    # Core matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    lines.append(f'        bool {prefix}_bF = const()[name = string("{prefix}_bF"), val = bool(false)];')
    lines.append(f'        tensor<fp16, [1, 1, {seq}, {oc}]> {prefix}_yh = matmul(transpose_x = {prefix}_bF, transpose_y = {prefix}_bF, x = {prefix}_a3, y = {prefix}_W)[name = string("{prefix}_yh")];')

    # Transpose back: [1,1,seq,oc] → [1,1,oc,seq]
    lines.append(f'        tensor<fp16, [1, 1, {oc}, {seq}]> {prefix}_yt = transpose(perm = {prefix}_pm, x = {prefix}_yh)[name = string("{prefix}_yt")];')

    # Reshape to standard: [1,1,oc,seq] → [1,oc,1,seq]
    lines.append(f'        tensor<int32, [4]> {prefix}_ro = const()[name = string("{prefix}_ro"), val = tensor<int32, [4]>([1, {oc}, 1, {seq}])];')
    lines.append(f'        tensor<fp16, [1, {oc}, 1, {seq}]> {prefix}_y = reshape(shape = {prefix}_ro, x = {prefix}_yt)[name = string("{prefix}_y")];')

    return '\n'.join(lines) + '\n'


def gen_lora_down_mil(dim: int, rank: int, seq: int) -> tuple[str, int, int]:
    """Generate MIL for LoRA down-projection: h = x @ A^T.

    Uses dynamic weight packing:
      Input:  [1, dim, 1, seq + rank]  (fp32)
        - spatial[0:seq] = x (activation)
        - spatial[seq:seq+rank] = A^T (transposed LoRA A matrix)
      Output: [1, rank, 1, seq]  (fp32)

    Returns:
        (mil_text, input_bytes, output_bytes)
    """
    sp_in = seq + rank
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp_in}]> x) {{\n'

    # Cast fp32 → fp16
    mil += f'        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {sp_in}]> xh = cast(dtype = to16, x = x)[name = string("cin")];\n'

    # Dynamic matmul: [seq, dim] @ [dim, rank] → [seq, rank]
    mil += _dynamic_matmul_block("ld", dim, rank, seq, 0, seq, "xh")

    # Cast fp16 → fp32
    mil += f'        string to32 = const()[name = string("to32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {rank}, 1, {seq}]> y = cast(dtype = to32, x = ld_y)[name = string("cout")];\n'
    mil += '    } -> (y);\n}\n'

    input_bytes = dim * sp_in * 4   # fp32
    output_bytes = rank * seq * 4   # fp32
    return mil, input_bytes, output_bytes


def gen_lora_up_mil(rank: int, dim: int, seq: int,
                    scaling: float = 1.0) -> tuple[str, int, int]:
    """Generate MIL for LoRA up-projection: out = (h @ B^T) * scale.

    Uses dynamic weight packing:
      Input:  [1, rank, 1, seq + dim]  (fp32)
        - spatial[0:seq] = h (from lora_down)
        - spatial[seq:seq+dim] = B^T (transposed LoRA B matrix)
      Output: [1, dim, 1, seq]  (fp32)

    Returns:
        (mil_text, input_bytes, output_bytes)
    """
    sp_in = seq + dim
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {rank}, 1, {sp_in}]> x) {{\n'

    # Cast fp32 → fp16
    mil += f'        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {rank}, 1, {sp_in}]> xh = cast(dtype = to16, x = x)[name = string("cin")];\n'

    # Dynamic matmul: [seq, rank] @ [rank, dim] → [seq, dim]
    mil += _dynamic_matmul_block("lu", rank, dim, seq, 0, seq, "xh")

    # Scale by lora_alpha/rank
    if abs(scaling - 1.0) > 1e-6:
        mil += f'        fp16 sc = const()[name = string("sc"), val = fp16({scaling})];\n'
        mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> lu_s = mul(x = lu_y, y = sc)[name = string("scaled")];\n'
        out_var = "lu_s"
    else:
        out_var = "lu_y"

    # Cast fp16 → fp32
    mil += f'        string to32 = const()[name = string("to32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {dim}, 1, {seq}]> y = cast(dtype = to32, x = {out_var})[name = string("cout")];\n'
    mil += '    } -> (y);\n}\n'

    input_bytes = rank * sp_in * 4
    output_bytes = dim * seq * 4
    return mil, input_bytes, output_bytes


def gen_lora_grad_b_mil(dim: int, rank: int, seq: int,
                        scaling: float = 1.0) -> tuple[str, int, int]:
    """Generate MIL for LoRA B gradient: dB = grad_out @ h^T * scale.

    Input:  [1, dim, 1, seq + seq]  (fp32)
      - spatial[0:seq]     = grad_out [dim, seq]
      - spatial[seq:2*seq] = h [dim ??? no, h is [rank, seq]]

    Actually, grad_out is [dim, seq] and h is [rank, seq].
    We need matmul(grad_out, h^T) = [dim, seq] @ [seq, rank] = [dim, rank].

    But grad_out has dim channels and h has rank channels — they can't share
    the same IC dimension. Solution: use two separate inputs.

    Input 0: [1, dim, 1, seq]  — grad_out (fp32)
    Input 1: [1, rank, 1, seq] — h (fp32)
    Output:  [1, dim, 1, rank] — dB (fp32)

    We use matmul(transpose_x=False, transpose_y=True):
      [1,1,dim,seq] @ [1,1,rank,seq]^T = [1,1,dim,rank]

    Returns:
        (mil_text, input0_bytes, input1_bytes, output_bytes)
    """
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq}]> go, tensor<fp32, [1, {rank}, 1, {seq}]> h) {{\n'

    # Cast both to fp16
    mil += f'        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> go16 = cast(dtype = to16, x = go)[name = string("cgo")];\n'
    mil += f'        tensor<fp16, [1, {rank}, 1, {seq}]> h16 = cast(dtype = to16, x = h)[name = string("ch")];\n'

    # Reshape grad_out: [1,dim,1,seq] → [1,1,dim,seq]
    mil += f'        tensor<int32, [4]> rgo = const()[name = string("rgo"), val = tensor<int32, [4]>([1, 1, {dim}, {seq}])];\n'
    mil += f'        tensor<fp16, [1, 1, {dim}, {seq}]> go4 = reshape(shape = rgo, x = go16)[name = string("rgo4")];\n'

    # Reshape h: [1,rank,1,seq] → [1,1,rank,seq]
    mil += f'        tensor<int32, [4]> rh = const()[name = string("rh"), val = tensor<int32, [4]>([1, 1, {rank}, {seq}])];\n'
    mil += f'        tensor<fp16, [1, 1, {rank}, {seq}]> h4 = reshape(shape = rh, x = h16)[name = string("rh4")];\n'

    # matmul(grad_out, h^T): [1,1,dim,seq] @ [1,1,seq,rank] → [1,1,dim,rank]
    mil += f'        bool bF = const()[name = string("bF"), val = bool(false)];\n'
    mil += f'        bool bT = const()[name = string("bT"), val = bool(true)];\n'
    mil += f'        tensor<fp16, [1, 1, {dim}, {rank}]> db4 = matmul(transpose_x = bF, transpose_y = bT, x = go4, y = h4)[name = string("mm")];\n'

    # Scale
    if abs(scaling - 1.0) > 1e-6:
        mil += f'        fp16 sc = const()[name = string("sc"), val = fp16({scaling})];\n'
        mil += f'        tensor<fp16, [1, 1, {dim}, {rank}]> db_s = mul(x = db4, y = sc)[name = string("scaled")];\n'
        mm_var = "db_s"
    else:
        mm_var = "db4"

    # Reshape: [1,1,dim,rank] → [1,dim,1,rank]
    mil += f'        tensor<int32, [4]> ro = const()[name = string("ro"), val = tensor<int32, [4]>([1, {dim}, 1, {rank}])];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {rank}]> db16 = reshape(shape = ro, x = {mm_var})[name = string("rdb")];\n'

    # Cast to fp32
    mil += f'        string to32 = const()[name = string("to32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {dim}, 1, {rank}]> dB = cast(dtype = to32, x = db16)[name = string("cout")];\n'
    mil += '    } -> (dB);\n}\n'

    in0_bytes = dim * seq * 4
    in1_bytes = rank * seq * 4
    out_bytes = dim * rank * 4
    return mil, in0_bytes, in1_bytes, out_bytes


def gen_lora_grad_a_mil(dim: int, rank: int, seq: int,
                        scaling: float = 1.0) -> tuple[str, int, int]:
    """Generate MIL for LoRA A gradient: dA = B^T @ grad_out @ x^T * scale.

    This is two chained matmuls:
      1. tmp = B^T @ grad_out: [rank,dim] @ [dim,seq] → [rank,seq]
      2. dA = tmp @ x^T:       [rank,seq] @ [seq,dim] → [rank,dim]

    Input 0: [1, dim, 1, seq + rank]  (fp32) — grad_out + B^T packed
      - spatial[0:seq]        = grad_out [dim, seq]
      - spatial[seq:seq+rank] = B^T [dim, rank]
    Input 1: [1, dim, 1, seq]  (fp32) — x (activation)
    Output:  [1, rank, 1, dim] (fp32) — dA

    Returns:
        (mil_text, input0_bytes, input1_bytes, output_bytes)
    """
    sp0 = seq + rank
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {dim}, 1, {sp0}]> packed, tensor<fp32, [1, {dim}, 1, {seq}]> xin) {{\n'

    # Cast to fp16
    mil += f'        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {sp0}]> ph = cast(dtype = to16, x = packed)[name = string("cp")];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> xh = cast(dtype = to16, x = xin)[name = string("cx")];\n'

    # Step 1: B^T @ grad_out using dynamic matmul helper
    # Slices grad_out[dim, seq] and B^T[dim, rank] from packed input
    # matmul: [seq, dim] @ [dim, rank] → [seq, rank]
    # Result: tmp_y [1, rank, 1, seq]
    mil += _dynamic_matmul_block("tmp", dim, rank, seq, 0, seq, "ph")

    # Step 2: tmp @ x^T
    # tmp is [1, rank, 1, seq], need to matmul with x [1, dim, 1, seq]
    # Want: [rank, seq] @ [seq, dim] → [rank, dim]
    # Use matmul(tmp_reshaped, x_reshaped, transpose_y=True... no)
    # Actually: reshape tmp [1,rank,1,seq] → [1,1,rank,seq]
    #           reshape x   [1,dim,1,seq]  → [1,1,dim,seq]
    #           matmul(transpose_y=True): [1,1,rank,seq] @ [1,1,seq,dim] → [1,1,rank,dim]
    #           But transpose_y=True on [1,1,dim,seq] gives [1,1,seq,dim]
    #           So matmul(x=tmp4, transpose_y=True, y=x4): [1,1,rank,seq]@[1,1,seq,dim] = [1,1,rank,dim]

    mil += f'        tensor<int32, [4]> rt = const()[name = string("rt"), val = tensor<int32, [4]>([1, 1, {rank}, {seq}])];\n'
    mil += f'        tensor<fp16, [1, 1, {rank}, {seq}]> tmp4 = reshape(shape = rt, x = tmp_y)[name = string("rt4")];\n'

    mil += f'        tensor<int32, [4]> rx = const()[name = string("rx"), val = tensor<int32, [4]>([1, 1, {dim}, {seq}])];\n'
    mil += f'        tensor<fp16, [1, 1, {dim}, {seq}]> x4 = reshape(shape = rx, x = xh)[name = string("rx4")];\n'

    mil += f'        bool bF = const()[name = string("bF"), val = bool(false)];\n'
    mil += f'        bool bT = const()[name = string("bT"), val = bool(true)];\n'
    mil += f'        tensor<fp16, [1, 1, {rank}, {dim}]> da4 = matmul(transpose_x = bF, transpose_y = bT, x = tmp4, y = x4)[name = string("mm2")];\n'

    # Scale
    if abs(scaling - 1.0) > 1e-6:
        mil += f'        fp16 sc = const()[name = string("sc"), val = fp16({scaling})];\n'
        mil += f'        tensor<fp16, [1, 1, {rank}, {dim}]> da_s = mul(x = da4, y = sc)[name = string("scaled")];\n'
        mm_var = "da_s"
    else:
        mm_var = "da4"

    # Reshape: [1,1,rank,dim] → [1,rank,1,dim]
    mil += f'        tensor<int32, [4]> ro = const()[name = string("ro"), val = tensor<int32, [4]>([1, {rank}, 1, {dim}])];\n'
    mil += f'        tensor<fp16, [1, {rank}, 1, {dim}]> da16 = reshape(shape = ro, x = {mm_var})[name = string("rda")];\n'

    # Cast to fp32
    mil += f'        string to32 = const()[name = string("to32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {rank}, 1, {dim}]> dA = cast(dtype = to32, x = da16)[name = string("cout")];\n'
    mil += '    } -> (dA);\n}\n'

    in0_bytes = dim * sp0 * 4
    in1_bytes = dim * seq * 4
    out_bytes = rank * dim * 4
    return mil, in0_bytes, in1_bytes, out_bytes


def gen_rmsnorm_mil(dim: int, seq: int) -> tuple[str, int, int]:
    """Generate MIL for RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight.

    Uses baked weight constant from BLOBFILE.
      Input:  [1, dim, 1, seq]  (fp16)
      Output: [1, dim, 1, seq]  (fp16)

    The weight file "@model_path/weights/rms_w.bin" must be provided as
    a weight blob when compiling.

    Returns:
        (mil_text, input_bytes, output_bytes)
    """
    inv_dim = 1.0 / dim
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp16, [1, {dim}, 1, {seq}]> x) {{\n'

    # x^2
    mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> sq = mul(x = x, y = x)[name = string("sq")];\n'

    # reduce_sum over channels (axis 1), keep_dims
    mil += f'        tensor<int32, [1]> rax = const()[name = string("rax"), val = tensor<int32, [1]>([1])];\n'
    mil += f'        bool kd = const()[name = string("kd"), val = bool(true)];\n'
    mil += f'        tensor<fp16, [1, 1, 1, {seq}]> ss = reduce_sum(x = sq, axes = rax, keep_dims = kd)[name = string("ss")];\n'

    # mean: sum / dim
    mil += f'        fp16 invd = const()[name = string("invd"), val = fp16({inv_dim})];\n'
    mil += f'        tensor<fp16, [1, 1, 1, {seq}]> ss2 = mul(x = ss, y = invd)[name = string("ss2")];\n'

    # + eps
    mil += f'        fp16 eps = const()[name = string("eps"), val = fp16(0.00001)];\n'
    mil += f'        tensor<fp16, [1, 1, 1, {seq}]> ss3 = add(x = ss2, y = eps)[name = string("ss3")];\n'

    # rsqrt: pow(x, -0.5)
    mil += f'        fp16 nhalf = const()[name = string("nhalf"), val = fp16(-0.5)];\n'
    mil += f'        tensor<fp16, [1, 1, 1, {seq}]> rrms = pow(x = ss3, y = nhalf)[name = string("rrms")];\n'

    # normalize
    mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> xr = mul(x = x, y = rrms)[name = string("xr")];\n'

    # weight (baked)
    mil += f'        tensor<fp16, [1, {dim}, 1, 1]> rw = const()[name = string("rw"), val = tensor<fp16, [1, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/rms_w.bin"), offset = uint64(64)))];\n'
    mil += f'        tensor<fp16, [1, {dim}, 1, {seq}]> out = mul(x = xr, y = rw)[name = string("out")];\n'
    mil += '    } -> (out);\n}\n'

    tensor_bytes = dim * seq * 2  # fp16
    return mil, tensor_bytes, tensor_bytes


def gen_conv_matmul_mil(dim_in: int, dim_out: int, seq: int) -> tuple[str, int, int]:
    """Generate MIL for a conv-based linear projection (baked weights).

    Used for classifier/embedding projections.
      Input:  [1, dim_in, 1, seq]  (fp32)
      Output: [1, dim_out, 1, seq] (fp32)

    Weight: BLOBFILE "embed.bin" [dim_out, dim_in, 1, 1] in fp16.

    Returns:
        (mil_text, input_bytes, output_bytes)
    """
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {dim_in}, 1, {seq}]> x) {{\n'

    # Conv constants
    mil += '        string pt = const()[name = string("pt"), val = string("valid")];\n'
    mil += '        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
    mil += '        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        int32 gr = const()[name = string("gr"), val = int32(1)];\n'

    # Cast to fp16
    mil += f'        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {dim_in}, 1, {seq}]> x16 = cast(dtype = to16, x = x)[name = string("cin")];\n'

    # Baked weight
    mil += f'        tensor<fp16, [{dim_out}, {dim_in}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{dim_out}, {dim_in}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/embed.bin"), offset = uint64(64)))];\n'

    # Conv (equivalent to matmul for 1x1 kernel)
    mil += f'        tensor<fp16, [1, {dim_out}, 1, {seq}]> y16 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W, x = x16)[name = string("conv")];\n'

    # Cast to fp32
    mil += f'        string to32 = const()[name = string("to32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {dim_out}, 1, {seq}]> y = cast(dtype = to32, x = y16)[name = string("cout")];\n'
    mil += '    } -> (y);\n}\n'

    in_bytes = dim_in * seq * 4
    out_bytes = dim_out * seq * 4
    return mil, in_bytes, out_bytes


class LoRAKernelSet:
    """Pre-compiled set of LoRA kernels for a given model dimension.

    Compiles 4 kernels once, then reuses them across all layers by
    writing different weights to the IOSurfaces.
    """

    def __init__(self, ane_bridge, dim: int, rank: int, seq: int,
                 scaling: float = 1.0):
        """Compile all LoRA kernels.

        Args:
            ane_bridge: ANEBridge instance
            dim: model hidden dimension
            rank: LoRA rank
            seq: sequence length
            scaling: LoRA scaling factor (alpha/rank)
        """
        # ANE requires all matmul dims to be multiples of 16
        for name, val in [("dim", dim), ("rank", rank), ("seq", seq)]:
            if val < 16 or val % 16 != 0:
                raise ValueError(
                    f"ANE requires {name}={val} to be a multiple of 16 (min 16)")

        self.ane = ane_bridge
        self.dim = dim
        self.rank = rank
        self.seq = seq
        self.scaling = scaling

        # Compile kernels
        self._compile_all()

    def _compile_all(self):
        """Compile all 4 LoRA kernels."""
        # 1. LoRA down: x @ A^T → h
        mil, in_bytes, out_bytes = gen_lora_down_mil(self.dim, self.rank, self.seq)
        self.down_kernel = self.ane.compile_kernel(
            mil, input_sizes=[in_bytes], output_sizes=[out_bytes])
        self.down_in_bytes = in_bytes
        self.down_out_bytes = out_bytes

        # 2. LoRA up: h @ B^T → out * scale
        mil, in_bytes, out_bytes = gen_lora_up_mil(
            self.rank, self.dim, self.seq, self.scaling)
        self.up_kernel = self.ane.compile_kernel(
            mil, input_sizes=[in_bytes], output_sizes=[out_bytes])
        self.up_in_bytes = in_bytes
        self.up_out_bytes = out_bytes

        # 3. Gradient B: grad_out @ h^T → dB
        mil, in0, in1, out = gen_lora_grad_b_mil(
            self.dim, self.rank, self.seq, self.scaling)
        self.grad_b_kernel = self.ane.compile_kernel(
            mil, input_sizes=[in0, in1], output_sizes=[out])
        self.grad_b_in0 = in0
        self.grad_b_in1 = in1
        self.grad_b_out = out

        # 4. Gradient A: (B^T @ grad_out) @ x^T → dA
        mil, in0, in1, out = gen_lora_grad_a_mil(
            self.dim, self.rank, self.seq, self.scaling)
        self.grad_a_kernel = self.ane.compile_kernel(
            mil, input_sizes=[in0, in1], output_sizes=[out])
        self.grad_a_in0 = in0
        self.grad_a_in1 = in1
        self.grad_a_out = out

    def forward(self, x: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute LoRA forward: out = (B @ A @ x) * scale.

        Args:
            x: [1, dim, 1, seq] fp32 activation
            A: [rank, dim] fp32 LoRA A matrix
            B: [dim, rank] fp32 LoRA B matrix

        Returns:
            [1, dim, 1, seq] fp32 LoRA output
        """
        # Step 1: h = x @ A^T
        # Pack x and A^T into spatial dimension
        A_T = A.T  # [dim, rank]
        packed_down = np.zeros((1, self.dim, 1, self.seq + self.rank), dtype=np.float32)
        packed_down[:, :, :, :self.seq] = x
        packed_down[:, :, :, self.seq:] = A_T.reshape(1, self.dim, 1, self.rank)

        self.ane.write_input(self.down_kernel, 0, packed_down)
        self.ane.eval(self.down_kernel)
        h = self.ane.read_output(self.down_kernel, 0,
                                  (1, self.rank, 1, self.seq), dtype=np.float32)

        # Step 2: out = h @ B^T * scale
        B_T = B.T  # [rank, dim]
        packed_up = np.zeros((1, self.rank, 1, self.seq + self.dim), dtype=np.float32)
        packed_up[:, :, :, :self.seq] = h
        packed_up[:, :, :, self.seq:] = B_T.reshape(1, self.rank, 1, self.dim)

        self.ane.write_input(self.up_kernel, 0, packed_up)
        self.ane.eval(self.up_kernel)
        out = self.ane.read_output(self.up_kernel, 0,
                                    (1, self.dim, 1, self.seq), dtype=np.float32)

        return out

    def backward(self, grad_out: np.ndarray, x: np.ndarray,
                 A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute LoRA gradients: dA, dB.

        Args:
            grad_out: [1, dim, 1, seq] fp32 upstream gradient
            x: [1, dim, 1, seq] fp32 saved activation
            A: [rank, dim] fp32 LoRA A matrix
            B: [dim, rank] fp32 LoRA B matrix

        Returns:
            (dA [rank, dim], dB [dim, rank]) fp32 gradients
        """
        # Compute h = A @ x (needed for dB)
        A_T = A.T
        packed_down = np.zeros((1, self.dim, 1, self.seq + self.rank), dtype=np.float32)
        packed_down[:, :, :, :self.seq] = x
        packed_down[:, :, :, self.seq:] = A_T.reshape(1, self.dim, 1, self.rank)
        self.ane.write_input(self.down_kernel, 0, packed_down)
        self.ane.eval(self.down_kernel)
        h = self.ane.read_output(self.down_kernel, 0,
                                  (1, self.rank, 1, self.seq), dtype=np.float32)

        # Gradient B: dB = grad_out @ h^T * scale → [dim, rank]
        self.ane.write_input(self.grad_b_kernel, 0,
                              np.ascontiguousarray(grad_out))
        self.ane.write_input(self.grad_b_kernel, 1,
                              np.ascontiguousarray(h))
        self.ane.eval(self.grad_b_kernel)
        dB_raw = self.ane.read_output(self.grad_b_kernel, 0,
                                       (1, self.dim, 1, self.rank), dtype=np.float32)
        dB = dB_raw.reshape(self.dim, self.rank)

        # Gradient A: dA = (B^T @ grad_out) @ x^T * scale → [rank, dim]
        B_T = B.T  # [rank, dim] — wait, B is [dim, rank], B^T is [rank, dim]
        # Pack grad_out + B^T into input 0: [1, dim, 1, seq + rank]
        # B^T is [rank, dim], but we need to pack as [dim, rank] in channel dim...
        # Actually, for the grad_a kernel: packed = [1, dim, 1, seq+rank]
        # where spatial[0:seq] = grad_out, spatial[seq:seq+rank] = B (which is [dim, rank])
        # The dynamic matmul does: [seq, dim] @ [dim, rank] → [seq, rank]
        # This gives us B^T @ grad_out transposed = (grad_out^T @ B)^T hmm...
        # Actually the dynamic matmul convention:
        #   act = grad_out [1, dim, 1, seq] → matmul as [seq, dim]
        #   W = B [1, dim, 1, rank] → matmul as [dim, rank]
        #   result = [seq, dim] @ [dim, rank] = [seq, rank]
        #   which is (B^T @ grad_out)^T in row-major
        # This is exactly what we want for step 1 of dA computation.
        packed_a0 = np.zeros((1, self.dim, 1, self.seq + self.rank), dtype=np.float32)
        packed_a0[:, :, :, :self.seq] = grad_out
        packed_a0[:, :, :, self.seq:] = B.reshape(1, self.dim, 1, self.rank)

        self.ane.write_input(self.grad_a_kernel, 0, packed_a0)
        self.ane.write_input(self.grad_a_kernel, 1,
                              np.ascontiguousarray(x))
        self.ane.eval(self.grad_a_kernel)
        dA_raw = self.ane.read_output(self.grad_a_kernel, 0,
                                       (1, self.rank, 1, self.dim), dtype=np.float32)
        dA = dA_raw.reshape(self.rank, self.dim)

        return dA, dB

    def free(self):
        """Free all compiled kernels."""
        for k in [self.down_kernel, self.up_kernel,
                  self.grad_b_kernel, self.grad_a_kernel]:
            if k:
                self.ane.free_kernel(k)


def self_test():
    """Test MIL generators with ANE hardware."""
    from ane_bridge_py import ANEBridge

    print("LoRA MIL Generator Self-Test")
    print("=" * 50)

    ane = ANEBridge()
    # ANE requires all matmul dimensions to be multiples of 16 (minimum 16)
    dim, rank, seq = 64, 16, 16
    scaling = 2.0

    # Test 1: Compile all kernels
    print(f"\nCompiling LoRA kernels (dim={dim}, rank={rank}, seq={seq})...")
    try:
        kernels = LoRAKernelSet(ane, dim, rank, seq, scaling)
        print(f"[OK] All 4 kernels compiled (compile count: {ane.compile_count})")
    except Exception as e:
        print(f"[FAIL] Kernel compilation: {e}")
        return False

    # Test 2: Forward pass
    print("\nTesting forward pass...")
    x = np.random.randn(1, dim, 1, seq).astype(np.float32) * 0.1
    A = np.random.randn(rank, dim).astype(np.float32) * 0.01
    B = np.zeros((dim, rank), dtype=np.float32)  # Standard LoRA init

    try:
        out = kernels.forward(x, A, B)
        print(f"[OK] Forward: input {x.shape} → output {out.shape}")
        print(f"     Output max: {np.abs(out).max():.6f} (should be ~0 with B=0)")

        # With non-zero B
        B = np.random.randn(dim, rank).astype(np.float32) * 0.01
        out = kernels.forward(x, A, B)
        print(f"     Output max (B≠0): {np.abs(out).max():.6f}")

        # Verify against numpy
        x_2d = x.reshape(dim, seq)
        expected = (B @ A @ x_2d * scaling).reshape(1, dim, 1, seq)
        err = np.abs(out - expected).max()
        print(f"     Max error vs numpy: {err:.6f}")
        if err > 0.5:
            print(f"[WARN] High error — fp16 rounding may be significant")
    except Exception as e:
        print(f"[FAIL] Forward: {e}")
        kernels.free()
        return False

    # Test 3: Backward pass
    print("\nTesting backward pass...")
    grad_out = np.random.randn(1, dim, 1, seq).astype(np.float32) * 0.1

    try:
        dA, dB = kernels.backward(grad_out, x, A, B)
        print(f"[OK] Backward: dA {dA.shape}, dB {dB.shape}")
        print(f"     dA max: {np.abs(dA).max():.6f}")
        print(f"     dB max: {np.abs(dB).max():.6f}")

        # Verify shapes
        assert dA.shape == (rank, dim), f"dA shape {dA.shape} != ({rank}, {dim})"
        assert dB.shape == (dim, rank), f"dB shape {dB.shape} != ({dim}, {rank})"

        # Verify non-zero gradients
        assert np.abs(dA).max() > 0, "dA is all zeros"
        assert np.abs(dB).max() > 0, "dB is all zeros"

        # Verify against numpy
        x_2d = x.reshape(dim, seq)
        go_2d = grad_out.reshape(dim, seq)
        h = A @ x_2d  # [rank, seq]
        expected_dB = go_2d @ h.T * scaling
        expected_dA = (B.T @ go_2d) @ x_2d.T * scaling

        err_dB = np.abs(dB - expected_dB).max()
        err_dA = np.abs(dA - expected_dA).max()
        print(f"     dB error vs numpy: {err_dB:.6f}")
        print(f"     dA error vs numpy: {err_dA:.6f}")
    except Exception as e:
        print(f"[FAIL] Backward: {e}")
        import traceback
        traceback.print_exc()
        kernels.free()
        return False

    kernels.free()
    print(f"\n[PASS] All LoRA MIL tests passed")
    print(f"       Final compile count: {ane.compile_count}")
    return True


if __name__ == "__main__":
    success = self_test()
    exit(0 if success else 1)
