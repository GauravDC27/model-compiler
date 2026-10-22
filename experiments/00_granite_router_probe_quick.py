import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
M = "ibm-granite/granite-3.1-1b-a400m-instruct"
tok = AutoTokenizer.from_pretrained(M)
prompts = [
 "Should the loan application with credit score 640 and DTI 45% be approved? Answer yes or no.",
 "Call the weather tool for Paris tomorrow.",
 "Write a two-line poem about the sea.",
]
def run(dtype, batch):
    m = AutoModelForCausalLM.from_pretrained(M, dtype=dtype); m.eval()
    tok.padding_side = "left"
    enc = tok(batch, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = m(**enc, output_router_logits=True)
    return enc, out
t=time.time()
enc32, o32 = run(torch.float32, prompts[:1])
print("fp32 b1 forward", time.time()-t)
rl = torch.stack([r.float() for r in o32.router_logits])  # L x T x E
L,T,E = rl.shape
s = rl.sort(dim=-1, descending=True).values
margin = s[...,7]-s[...,8]
print("layers",L,"tokens",T,"experts",E)
for q in [0.01,0.05,0.1,0.5]:
    print("margin quantile",q, float(margin.flatten().quantile(q)))
print("frac margin<1e-2", float((margin<1e-2).float().mean()), "<1e-3", float((margin<1e-3).float().mean()))
enc16, o16 = run(torch.bfloat16, prompts[:1])
rl16 = torch.stack([r.float() for r in o16.router_logits])
top32 = rl.topk(8,dim=-1).indices.sort(-1).values; top16 = rl16.topk(8,dim=-1).indices.sort(-1).values
flip = (top32!=top16).any(-1).float()
print("fp32 vs bf16 per (layer,token) routing set differs:", float(flip.mean()), "tokens w/ any layer diff:", float(flip.any(0).float().mean()))
# batch composition: same prompt alone vs in a batch (left padded)
encb, ob = run(torch.float32, prompts)
rlb = torch.stack([r.float() for r in ob.router_logits]).view(L, len(prompts), -1, E)[:,0]
rlb = rlb[:, -T:]
topb = rlb.topk(8,dim=-1).indices.sort(-1).values
print("fp32 batch1 vs batch3 routing differs:", float((top32!=topb).any(-1).float().mean()), "max |dlogit|", float((rl-rlb).abs().max()))
print("next-token argmax b1 vs b3:", int(o32.logits[0,-1].argmax()), int(ob.logits[0,-1].argmax()), "max|dlogits|", float((o32.logits[0,-1]-ob.logits[0,-1]).abs().max()))
