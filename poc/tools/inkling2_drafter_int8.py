"""Converts mtp.safetensors (+ the target's unembedding and embed norm) into the INT8 drafter file read by the Edge-LLM runtime (InklingDrafter).
Linear weights: per-row symmetric INT8 (RTN), scale fp32. M = hidden_out.W^T @ U^T folds the Markov correction into the vocabulary space (rank 256);
bU = hidden_out.bias @ U^T. Norm weights, mask embedding, mixing weights stay fp32; markov_w1 fp16."""
import sys, torch, json
from safetensors import safe_open
from safetensors.torch import save_file
hf, out = sys.argv[1], sys.argv[2]
V = 200058; VP = 200064      # tokenizer vocabulary and its padding to a multiple of 8 rows
dev = 'cuda'
mtp = safe_open(f'{hf}/mtp.safetensors', 'pt')
tgt = safe_open(f'{hf}/model-00001-of-00001.safetensors', 'pt')
T = {}
IDX = torch.tensor([[0, 1, 8, 9], [2, 3, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]])
def shuffle(q):           # [N, K] int8 -> mma-fragment order: per 8 rows x 64 k slab, lane (g, t) holds its B fragments of four k16 tiles in 16 contiguous bytes
    N, K = q.shape
    assert N % 8 == 0 and K % 64 == 0, (N, K)
    sel = q.view(N // 8, 8, K // 64, 4, 16)[..., IDX.to(q.device)]      # [G, g, kb, j, t, e]
    return sel.permute(0, 2, 1, 4, 3, 5).contiguous().view(N, K)       # [G, kb, g, t, j, e]
def q8(name, w):          # w [N, K] float -> int8 [N, K] (shuffled) + scale [N]
    w = w.float()
    s = w.abs().amax(1).clamp(min=1e-8) / 127.0
    q = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    T[name] = shuffle(q).cpu().contiguous(); T[name + '.s'] = s.float().cpu().contiguous()
def f32(name, w): T[name] = w.float().cpu().contiguous()
g = lambda k: mtp.get_tensor(k).to(dev)
for i in range(3):
    p = f'layers.{i}.'
    f32(f'l{i}.ln_in', g(p + 'input_layernorm.weight')); f32(f'l{i}.ln_post', g(p + 'post_attention_layernorm.weight')); f32(f'l{i}.dflare', g(p + 'dflare_norm.weight'))
    for a, b in (('q_norm', 'q_norm'), ('k_norm', 'k_norm'), ('k_norm_ctx', 'k_norm_ctx')): f32(f'l{i}.{a}', g(p + f'self_attn.{b}.weight'))
    for a, b in (('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj'), ('kctx', 'k_proj_ctx'), ('vctx', 'v_proj_ctx')): q8(f'l{i}.{a}', g(p + f'self_attn.{b}.weight'))
    q8(f'l{i}.gate', g(p + 'mlp.gate_proj.weight')); q8(f'l{i}.up', g(p + 'mlp.up_proj.weight')); q8(f'l{i}.down', g(p + 'mlp.down_proj.weight'))
f32('alpha', torch.softmax(g('fuse_weight').float(), -1))
f32('mask_embed', g('mask_embed')); f32('final_norm_drafter', g('norm.weight'))
f32('markov_dflare', g('markov_head.dflare_norm.weight'))
f32('embed_norm', tgt.get_tensor('model.llm.embed_norm.weight')); f32('target_final_norm', tgt.get_tensor('model.llm.norm.weight'))
wp = g('markov_head.rnn_proj.weight').float()          # [768, 5120]: columns 0..511 act on [state, w1], 512.. on [hid, feat]
f32('rnn_state_w', wp[:, :512].contiguous()); q8('rnn_inv', wp[:, 512:].contiguous()); f32('rnn_b', g('markov_head.rnn_proj.bias'))
T['markov_w1'] = mtp.get_tensor('markov_head.markov_w1.weight')[:VP].to(torch.float16).contiguous() if mtp.get_tensor('markov_head.markov_w1.weight').shape[0] >= VP else None
U = tgt.get_tensor('model.llm.unembed.weight')[:V].to(dev)   # [V, 2304] bf16
Uf = U.float()
ho_w = g('markov_head.hidden_out.weight').float()      # [2304, 256]
ho_b = g('markov_head.hidden_out.bias').float()
M = (ho_w.T @ Uf.T)                                    # [256, V]
bU = (ho_b @ Uf.T)                                     # [V]
pad = lambda x: torch.cat([x, torch.zeros(VP - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)], 0)
q8('U', pad(Uf)); q8('Mt', pad(M.T.contiguous()))
bUp = torch.full((VP,), -1e30, device=dev); bUp[:V] = bU   # padded rows can never win the argmax
f32('bU', bUp)
T['markov_w1'] = T['markov_w1']
# pad rows of the INT8 matrices must not win either: their scale is tiny, the bias row carries -1e30
save_file({k: v for k, v in T.items() if v is not None}, out)
print('wrote', out, sum(v.numel() * v.element_size() for v in T.values() if v is not None) / 1e6, 'MB')
