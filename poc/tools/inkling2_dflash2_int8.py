"""usage: inkling2_dflash2_int8.py DFLASH2_MODEL_SAFETENSORS TARGET_HF_DIR OUT_FILE
Converts incoai/Inkling-2-Mini-DFlash2 (+ the target's unembedding and embed norm) into the INT8 file read by InklingDflash2Drafter.
Linear weights: per-row symmetric INT8 (RTN) pre-shuffled for the tensor-core GEMV, fp32 scale. The convolution coefficient projection [576, 2304] is split into its pre-side
and post-side halves (288 rows each: [tap, group]); base kernels, codebooks and the mask embedding are kept in fp16 / fp32."""
import sys, torch, json
from safetensors import safe_open
from safetensors.torch import save_file
ck, hf, out = sys.argv[1], sys.argv[2], sys.argv[3]
V = 200058; VP = 200064
dev = 'cuda'
d = safe_open(ck, 'pt')
tgt = safe_open(f'{hf}/model-00001-of-00001.safetensors', 'pt')
T = {}
IDX = torch.tensor([[0, 1, 8, 9], [2, 3, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]])
def shuffle(q):
    N, K = q.shape
    assert N % 8 == 0 and K % 64 == 0, (N, K)
    sel = q.view(N // 8, 8, K // 64, 4, 16)[..., IDX.to(q.device)]
    return sel.permute(0, 2, 1, 4, 3, 5).contiguous().view(N, K)
def q8(name, w):
    w = w.float()
    s = w.abs().amax(1).clamp(min=1e-8) / 127.0
    q = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    T[name] = shuffle(q).cpu().contiguous(); T[name + '.s'] = s.float().cpu().contiguous()
def f32(name, w): T[name] = w.float().cpu().contiguous()
g = lambda k: d.get_tensor(k).to(dev)
q8('fc', g('fc.weight')); f32('hidden_norm', g('hidden_norm.weight')); f32('norm', g('norm.weight'))
for i in range(5):
    p = f'layers.{i}.'
    f32(f'l{i}.ln_in', g(p + 'input_layernorm.weight')); f32(f'l{i}.ln_post', g(p + 'post_attention_layernorm.weight'))
    f32(f'l{i}.q_norm', g(p + 'self_attn.q_norm.weight')); f32(f'l{i}.k_norm', g(p + 'self_attn.k_norm.weight'))
    for a, b in (('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj')): q8(f'l{i}.{a}', g(p + f'self_attn.{b}.weight'))
    for a, b in (('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')): q8(f'l{i}.{a}', g(p + f'mlp.{b}.weight'))
    for pre, src in (('ac', 'attention_conv'), ('mc', 'mlp_conv')):
        kp = g(p + f'{src}.kernel_projection.weight')          # [576, 2304]: rows = side (2) x tap (2) x group (144)
        q8(f'l{i}.{pre}_pre', kp[:288].contiguous()); q8(f'l{i}.{pre}_post', kp[288:].contiguous())
        base = g(p + f'{src}.base_kernel')                      # [side, tap, hidden]
        T[f'l{i}.{pre}_base0'] = base[0].half().cpu().contiguous(); T[f'l{i}.{pre}_base1'] = base[1].half().cpu().contiguous()
q8('sel_proj', g('candidate_selector.hidden_projection.weight'))
T['cb_pred'] = g('candidate_selector.predecessor_codebook')[:VP].half().cpu().contiguous()
T['cb_succ'] = g('candidate_selector.successor_codebook')[:VP].half().cpu().contiguous()
U = tgt.get_tensor('model.llm.unembed.weight')[:V].to(dev).float()
q8('U', torch.cat([U, torch.zeros(VP - V, U.shape[1], device=dev)], 0))
en = tgt.get_tensor('model.llm.embed_norm.weight').to(dev).float()
f32('embed_norm', en); f32('target_final_norm', tgt.get_tensor('model.llm.norm.weight'))
me = tgt.get_tensor('model.llm.embed.weight')[200056].to(dev).float()
f32('mask_embed', me * torch.rsqrt(me.pow(2).mean() + 1e-6) * en)
save_file(T, out)
json.dump({"type": "inkling2_dflash2_int8", "source": "incoai/Inkling-2-Mini-DFlash2"}, open(out.rsplit('.', 1)[0] + '.json', 'w'))
print('wrote', out, sum(v.numel() * v.element_size() for v in T.values()) / 1e6, 'MB')
