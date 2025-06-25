gpt2xl_params: dict[str, int | str] = {"name": "gpt2xl", "d": 1600, "l": 48}

gpt2l_params: dict[str, int | str] = {"name": "gpt2l", "d": 1280, "l": 36}

gpt2m_params: dict[str, int | str] = {"name": "gpt2m", "d": 1024, "l": 24}

gpt2s_params: dict[str, int | str] = {"name": "gpt2s", "d": 768, "l": 12}

v: int = 50257  # vocab size same for all models
s: int = 1024  # context length same for all models

for model in [gpt2xl_params, gpt2l_params, gpt2m_params, gpt2s_params]:
    d: int = model["d"]  # type: ignore
    num_layers: int = model["l"]  # type: ignore
    name: str = model["name"]  # type: ignore

    num_params: int = 2 * v * d + (1 + 2 * num_layers) * d + 192 * d * d + 576 * d * d

    mem_req: int = 4 * num_params

    print(f"Number of trainable parameters: {num_params}")
    print(f"Memory required: {mem_req} bytes")

    # Matrix multiplications in attention are the projections: 8sd^2 FLOPs, kq scores: 2sd^2 FLOPs, value multiplication: 2s^2d FLOPs, outputs: 2sd^2 so 48(12sd^2 + 2s^2d) for all attention FLOPs. Then we also have ffn which is 2(3(4sd^2)) FLOPs for two up_proj and one down_proj. We have 48x of these too. Finally, the lm_head projection is 2sdv FLOPs. Total FLOPs is
    s = 1024

    attn_flops: int = 48 * (12 * s * d**2 + 2 * s**2 * d)
    ffn_flops: int = 48 * 2 * 3 * 4 * s * d**2
    lm_head_flops: int = 2 * s * d * v

    total_flops: int = attn_flops + ffn_flops + lm_head_flops

    print(f"Attention FLOPs: {attn_flops}")
    print(f"FFN FLOPs: {ffn_flops}")
    print(f"LM Head FLOPs: {lm_head_flops}")
    print(f"Total FLOPs: {total_flops}")

    # If this is gpt2xl we want to also report stats with s = 16384
    if name == "gpt2xl":
        temp_s = 16384
        temp_attn_flops: int = 48 * (12 * temp_s * d**2 + 2 * temp_s**2 * d)
        temp_ffn_flops: int = 48 * 2 * 3 * 4 * temp_s * d**2
        temp_lm_head_flops: int = 2 * temp_s * d * v
        temp_total_flops: int = temp_attn_flops + temp_ffn_flops + temp_lm_head_flops
        print(f" Long Sequence Attention FLOPs: {temp_attn_flops}")
        print(f" Long Sequence FFN FLOPs: {temp_ffn_flops}")
        print(f" Long Sequence LM Head FLOPs: {temp_lm_head_flops}")
        print(f" Long Sequence Total FLOPs: {temp_total_flops}")

    print(f"Model: {name}")
    print("-" * 40)
