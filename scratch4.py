# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *
import circuitsvis as cv

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("pythia-2.8b")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
vocab_df = pickle.load(open("vocab_df.pkl", "rb"))
short_chars_vocab_df = vocab_df.query("is_word & num_chars == 5")
# %%
alphabet = "abcdefghijklmnopqrstuvwxyz"
alphalist = [i for i in alphabet]
alpha_tokens = torch.tensor(
    [model.to_single_token(" " + c.upper()) for c in alphabet], device="cuda"
)
alpha_U = model.W_U[:, alpha_tokens]
alpha_U_cent = alpha_U - alpha_U.mean(-1, keepdim=True)

# %%
eff_embed = model.W_E + model.blocks[0].mlp(model.blocks[0].ln2(model.W_E[None]))
eff_embed = eff_embed.squeeze()
eff_embed.shape
# %%
tokens_5 = vocab_df.query("num_chars==5 & is_word").token.values
eff_embed_5 = eff_embed[tokens_5]
eff_embed_5.shape, tokens_5
# %%
def get_probe_metrics(probe, letter):
    log_probs = probe(eff_embed_5).log_softmax(dim=-1)
    labels = torch.tensor(short_chars_vocab_df[f"let{letter}"].values).cuda()
    clps = log_probs[np.arange(len(log_probs)), labels]
    loss = -clps.mean()
    acc = (log_probs.argmax(dim=-1) == labels).float().mean()
    rank = (log_probs > clps[:, None]).sum(-1)
    median_rank = rank.median()
    mean_recip_rank = (1 / (rank + 1)).mean()
    return {
        "loss": loss.item(),
        "acc": acc.item(),
        "median_rank": median_rank.item(),
        "mean_recip_rank": mean_recip_rank.item(),
    }
probes = []
for i in range(5):
    probe = nn.Linear(d_model, 26).cuda()
    probe.load_state_dict(torch.load(f"/workspace/GPT-Spelling-Bee/probe_let{i}.pt"))
    probes.append(probe)
probew = torch.stack([p.weight for p in probes], dim=0)
probeb = torch.stack([p.bias for p in probes], dim=0)
probew.shape, probeb.shape
# %%
def make_single_prompt():
    word = short_chars_vocab_df.string.sample().item().strip()
    return f" {word}:" + "".join([f" {c.upper()}" for c in word.strip()])


def make_kshot_prompt(k=3):
    return "\n".join([make_single_prompt() for _ in range(k)])


def make_kshot_prompts(n=10, k=3):
    return [make_kshot_prompt(k) for _ in range(n)]

def get_answer_index(prompts):
    batch_size = len(prompts)
    answer_index = torch.zeros((batch_size, 5), device="cuda", dtype=torch.int64) - 1
    for i in range(batch_size):
        for j in range(5):
            answer_index[i, j] = alphabet.index(prompts[i][2 * j - 9].lower())
    return answer_index










# %%
# prompts = make_kshot_prompts(batch_size, 2)
# tokens = model.to_tokens(prompts)
# answer_index = get_answer_index(prompts)

# batch_alpha_U = alpha_U_cent[:, answer_index]

# %%
batch_size = 128
clean_prompts = make_kshot_prompts(batch_size, 3)
corr_prompts = make_kshot_prompts(batch_size, 3)
corr_prompts = [i[:36]+j[36:] for i, j in zip(clean_prompts, corr_prompts)]
print(corr_prompts[0], clean_prompts[0])
clean_tokens = model.to_tokens(clean_prompts)
clean_answer_index = get_answer_index(clean_prompts)

corr_tokens = model.to_tokens(corr_prompts)
corr_answer_index = get_answer_index(corr_prompts)
# %%
def metric(logits, answer_index, normalize=True):
    if len(logits.shape)==3:
        logits = logits[:, -4, :]
    log_probs = logits.log_softmax(dim=-1)
    alpha_log_probs = log_probs[:, alpha_tokens]
    # for i in range(5):
    #     print(alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, i]].mean())
    clps = alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, 2]] - alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, 1]]
    loss = clps.mean()
    if normalize:
        return (loss - CORR_BASELINE) / (CLEAN_BASELINE - CORR_BASELINE)
    else:
        return loss
clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)
CLEAN_BASELINE = (metric(clean_logits, clean_answer_index, False))
CORR_BASELINE = (metric(corr_logits, clean_answer_index, False))
# %%
def patching_metrics(logits, answer_index):
    if logits.shape[1] != 5:
        logits = logits[:, -6:-1, :]
    log_probs = logits.log_softmax(dim=-1)
    alpha_log_probs = log_probs[:, :, alpha_tokens]
    metrics = {}
    for i in range(5):
        metrics[f"prob_letter{i}"] = alpha_log_probs[:, i, :].exp().sum(-1).mean(0).item()
    answer_log_probs = alpha_log_probs[np.arange(len(alpha_log_probs))[:, None], :, answer_index]
    for i in range(5):
        metrics[f"Let{i}"] = answer_log_probs[:, i, i].mean().item()
        metrics[f"Let{i}Acc"] = (alpha_log_probs[:, i, :].max(dim=-1).values==answer_log_probs[:, i, i]).float().mean().item()
    for i in range(1, 5):
        metrics[f"Let{i}Prev"] = answer_log_probs[:, i-1, i].mean().item()
        metrics[f"Let{i}PrevDiff"] = metrics[f"Let{i}"] - metrics[f"Let{i}Prev"]
    for i in range(4):
        metrics[f"Let{i}Next"] = answer_log_probs[:, i+1, i].mean().item()
        metrics[f"Let{i}NextDiff"] = metrics[f"Let{i}"] - metrics[f"Let{i}Next"]
    return metrics
patching_metrics(clean_logits, clean_answer_index)
resid_patch_df = pd.read_csv("resid_patch_df.csv", index_col=0)
mlp_patch_df = pd.read_csv("mlp_patch_df.csv", index_col=0)
attn_patch_df = pd.read_csv("attn_patch_df.csv", index_col=0)
mlp_30_path_patch_df = pd.read_csv("mlp_30_path_patch_df.csv", index_col=0)
# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: not "hook_pre" in n and not "resid_mid" in n)
corr_logits, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: not "hook_pre" in n and not "resid_mid" in n)
# %%
next2diff_U = alpha_U_cent[:, clean_answer_index[:, 2]] - alpha_U_cent[:, clean_answer_index[:, 3]]
clean_resid_stack, resid_labels = clean_cache.decompose_resid(apply_ln=True, pos_slice=-4, return_labels=True)
clean_nextdla = (clean_resid_stack * next2diff_U.T).sum(-1).mean(-1)
corr_resid_stack, resid_labels = corr_cache.decompose_resid(apply_ln=True, pos_slice=-4, return_labels=True)
corr_nextdla = (corr_resid_stack * next2diff_U.T).sum(-1).mean(-1)
line([clean_nextdla, corr_nextdla, clean_nextdla - corr_nextdla], x=resid_labels, title="Next DLA", line_labels=["clean", "corr", "diff"])
# %%
line((clean_resid_stack * next2diff_U.T).sum(-1).std(-1))
# %%
imshow((clean_resid_stack * next2diff_U.T).sum(-1) / (clean_resid_stack * next2diff_U.T).sum(-1).sum(dim=0, keepdim=True), y=resid_labels, x=nutils.process_tokens_index(clean_tokens[:, -7]), zmin=-0.5, zmax=0.5)
# %%
mlp30_nextdla = (clean_resid_stack * next2diff_U.T).sum(-1)[-3]
words = model.to_str_tokens((clean_tokens[:, -7]))
display(pd.Series(index=words, data=mlp30_nextdla.cpu().numpy()).sort_values().head(20))
display(pd.Series(index=words, data=mlp30_nextdla.cpu().numpy()).sort_values().tail(20))
# %%
next2diff_U = alpha_U_cent[:, clean_answer_index[:, 2]] - alpha_U_cent[:, clean_answer_index[:, 3]]
next2diff_out = model.W_out[30] @ next2diff_U
clean_resid_stack, resid_labels = clean_cache.decompose_resid(apply_ln=True, pos_slice=-4, return_labels=True)
clean_nextdla = (clean_resid_stack * next2diff_U.T).sum(-1).mean(-1)
corr_resid_stack, resid_labels = corr_cache.decompose_resid(apply_ln=True, pos_slice=-4, return_labels=True)
corr_nextdla = (corr_resid_stack * next2diff_U.T).sum(-1).mean(-1)
line([clean_nextdla, corr_nextdla, clean_nextdla - corr_nextdla], x=resid_labels, title="Next DLA", line_labels=["clean", "corr", "diff"])
# %%
W_in30 = model.W_in[30]
W_out30 = model.W_out[30]
line(alpha_U.T @ W_in30, line_labels=alphalist)
# %%
nutils.show_df(nutils.create_vocab_df(W_out30[5458] @ model.W_U).tail(30))
# %%
model.cfg.use_hook_mlp_in = True
# %%
filter_not_qkv_input = lambda name: "_input" not in name and "_result" not in name and "_attn_in" not in name and "attn_scores" not in name and "pattern" not in name and "hook_pre" not in name
filter_is_pattern = lambda name: "pattern" in name or "attn_scores" in name
def get_cache_fwd_and_bwd(model, tokens, metric, pos_slice=slice(-7, -3), to_cpu=True):
    for name, param in model.named_parameters():
        if "W_E" not in name:
            param.requires_grad = False
    device = "cpu" if to_cpu else "cuda"
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act[:, pos_slice].detach().to(device)
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    def forward_cache_hook_pattern(act, hook):
        cache[hook.name] = act[:, :, pos_slice].detach().to(device)
    model.add_hook(filter_is_pattern, forward_cache_hook_pattern, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act[:, pos_slice].detach().to(device)
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")
    def backward_cache_hook_pattern(act, hook):
        grad_cache[hook.name] = act[:, :, pos_slice].detach().to(device)
    model.add_hook(filter_is_pattern, backward_cache_hook_pattern, "bwd")

    torch.set_grad_enabled(True)
    value = metric(model(tokens))
    # Scale the loss by the batch size, since the metric averages over the batch
    # This is useful because it makes the gradient independent of batch size
    value *= len(tokens)
    value.backward()
    model.reset_hooks()
    torch.set_grad_enabled(False)
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)
toy_loss, toy_cache, toy_grad_cache = get_cache_fwd_and_bwd(model, "The cat sat on the mat", lambda x: x.mean(), pos_slice=slice(3, 5))
# %%
def metric_let2_lp(logits, answer_index, normalize=True):
    if len(logits.shape)==3:
        logits = logits[:, -4, :]
    log_probs = logits.log_softmax(dim=-1)
    alpha_log_probs = log_probs[:, alpha_tokens]
    clps = alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, 2]]
    loss = clps.mean()
    if normalize:
        return (loss - CORR_BASELINE) / (CLEAN_BASELINE - CORR_BASELINE)
    else:
        return loss
clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)
CLEAN_BASELINE = (metric_let2_lp(clean_logits, clean_answer_index, False))
CORR_BASELINE = (metric_let2_lp(corr_logits, clean_answer_index, False))
# %%
def cache_to_device(cache, device="cpu"):
    return ActivationCache({k:v.to(device) for k, v in cache.cache_dict.items()}, cache.model)
    
# %%
clean_baseline_2, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, clean_tokens, lambda x: metric_let2_lp(x, clean_answer_index, True))
corr_baseline_2, corr_cache, corr_grad_cache = get_cache_fwd_and_bwd(model, corr_tokens, lambda x: metric_let2_lp(x, clean_answer_index, True))
# %%
noising_cache_cpu = ActivationCache({
    k: clean_grad_cache[k] * (clean_cache[k] - corr_cache[k]) for k in tqdm.tqdm(clean_cache.cache_dict.keys())
}, model)
noising_cache = cache_to_device(noising_cache_cpu, "cuda")
del noising_cache_cpu
denoising_cache_cpu = ActivationCache({
    k: corr_grad_cache[k] * (clean_cache[k] - corr_cache[k]) for k in tqdm.tqdm(clean_cache.cache_dict.keys())
}, model)
denoising_cache = cache_to_device(denoising_cache_cpu, "cuda")
del denoising_cache_cpu

# %%
head_attr = einops.rearrange(denoising_cache.stack_activation("z").mean(1).sum(-1), "layer pos head -> pos layer head")
imshow(head_attr, facet_col=0, facet_labels=["word", ":", "Let0", "Let1"], yaxis="Layer", xaxis="Head", title="Denoising Head Attr")
nois_head_attr = einops.rearrange(noising_cache.stack_activation("z").mean(1).sum(-1), "layer pos head -> pos layer head")
imshow(nois_head_attr, facet_col=0, facet_labels=["word", ":", "Let0", "Let1"], yaxis="Layer", xaxis="Head", title="Noise head attr")
# %%
pos_labels = ["word", ":", "Let0", "Let1"]
clean_attrib_pattern = (clean_cache.stack_activation("pattern") * clean_grad_cache.stack_activation("pattern")).cuda()
clean_attrib_pattern_l28h17 = clean_attrib_pattern[28, :, 17]
imshow(clean_attrib_pattern_l28h17, facet_col=1, facet_labels=pos_labels, yaxis="Batch", y=nutils.process_tokens_index(clean_tokens[:, -7]), xaxis="Pos", x=nutils.process_tokens_index(clean_tokens[0]))

clean_pattern = (clean_cache.stack_activation("pattern")).cuda()
clean_pattern_l28h17 = clean_pattern[28, :, 17]
imshow(clean_pattern_l28h17, facet_col=1, facet_labels=pos_labels, yaxis="Batch", y=nutils.process_tokens_index(clean_tokens[:, -7]), xaxis="Pos", x=nutils.process_tokens_index(clean_tokens[0]))
# %%
# attn_diff = clean_pattern_l28h17[:, -1, -4] - clean_pattern_l28h17[:, -1, -5]
# words = [s[1:] for s in model.to_str_tokens(clean_tokens[:, -7])]
# is_vowl_
histogram(nutils.cos(corr_grad_cache["z", 28][:, -1, 17, :], clean_grad_cache["z", 28][:, -1, 17, :]), marginal="box")
# %%
l28h17_clean_z = clean_cache["z", 28][:, -1, 17, :]
l28h17_corr_z = corr_cache["z", 28][:, -1, 17, :]
l28h17_diff_z = l28h17_clean_z - l28h17_corr_z
W_O = model.W_O[28, 17].clone()
clean_answer_U = alpha_U_cent[:, clean_answer_index[:, 2]]
clean_answer_O = W_O @ clean_answer_U
clean_answer_O = clean_answer_O.T.cpu()
clean_answer_O.shape, l28h17_clean_z.shape
histogram((clean_answer_O * l28h17_clean_z).sum(-1), marginal="box")
histogram((clean_answer_O * l28h17_corr_z).sum(-1), marginal="box")
histogram((clean_answer_O * l28h17_diff_z).sum(-1), marginal="box")
# %%

# %%
histogram(nutils.cos(corr_grad_cache["z", 28][:, -1, 17, :], clean_grad_cache["z", 28][:, -1, 17, :]), marginal="box", title="Cos corr & clean")
histogram(nutils.cos(clean_answer_O, clean_grad_cache["z", 28][:, -1, 17, :]), marginal="box", title="Cos wdla & clean")
histogram(nutils.cos(corr_grad_cache["z", 28][:, -1, 17, :], clean_answer_O), marginal="box", title="Cos corr & wdla")
# %%
histogram((clean_grad_cache["z", 28][:, -1, 17, :] * (clean_cache["z", 28][:, -1, 17, :] - corr_cache["z", 28][:, -1, 17, :])).sum(-1), marginal="box", title="Noising")
# %%
clean_ln_scale = clean_cache.stack_activation("scale", sublayer_type="ln2")
mlp_grad_in = (corr_grad_cache.stack_activation("normalized", sublayer_type="ln2") / clean_ln_scale).cuda()
query_grad_in = einops.einsum(corr_grad_cache.stack_activation("q").cuda()[:, :, -1], model.W_Q, 1/clean_ln_scale[:, :, -1].squeeze().cuda(), "layer batch head d_head, layer head d_model d_head, layer batch -> layer batch head d_model")

key_grad_in = einops.einsum(corr_grad_cache.stack_activation("k").cuda()[:, :, -1], model.W_Q, 1/clean_ln_scale[:, :, -1].squeeze().cuda(), "layer batch head d_head, layer head d_model d_head, layer batch -> layer batch head d_model")
value_grad_in = einops.einsum(corr_grad_cache.stack_activation("v").cuda()[:, :, -1], model.W_Q, 1/clean_ln_scale[:, :, -1].squeeze().cuda(), "layer batch head d_head, layer head d_model d_head, layer batch -> layer batch head d_model")

final_grad_in = (corr_grad_cache["normalized"][:, -1, :] / corr_cache["scale"][:, -1, :])[None].cuda()

labels = (
    ["Logits"]+
    [f"MLP{i}" for i in range(n_layers)] +
    [f"{t}L{l}H{h}" for t in "QKV" for l in range(n_layers) for h in range(n_heads)]
    )
layers = (
    [n_layers]+
    [i for i in range(n_layers)] +
    [l for t in "QKV" for l in range(n_layers) for h in range(n_heads)]
)
typ = (
    ["Logits"]+
    ["MLP" for i in range(n_layers)] +
    [t for t in "QKV" for l in range(n_layers) for h in range(n_heads)]
)
path_end_df = pd.DataFrame({
    "label": labels,
    "layer": layers,
    "type": typ,
})
path_end_tensor = torch.cat([
    final_grad_in,
    mlp_grad_in[:, :, -1],
    einops.rearrange(query_grad_in, "layer batch head d_model -> (layer head) batch d_model"),
    einops.rearrange(key_grad_in, "layer batch head d_model -> (layer head) batch d_model"),
    einops.rearrange(value_grad_in, "layer batch head d_model -> (layer head) batch d_model"),
], dim=0)
print(path_end_tensor.shape)
# %%

l28h17_diff_result = l28h17_diff_z.cuda() @ model.W_O[28, 17]
l28h17_path_scores = (path_end_tensor[path_end_df.query("layer>28").index.values] * l28h17_diff_result).sum(-1)
line(l28h17_path_scores.mean(-1), x=path_end_df.query("layer>28").label, title="Path scores")
# %%
line(l28h17_path_scores[:, :8].T, x=path_end_df.query("layer>28").label, title="Path scores")
# %%
def plot_line_and_df(y, x, ascending=False, return_df=False, sort_col=None, line_labels = None, refactor_decomp=False, **kwargs):
    y = to_numpy(y)
    if refactor_decomp:
        prefix = len([i for i in x if "out" not in i])
        y = np.concatenate([y[..., :prefix], y[..., prefix::2], y[..., prefix+1::2]], axis=-1)
        x = x[:prefix]+x[prefix::2]+x[prefix+1::2]

    if line_labels is None:
        line_labels = [str(i) for i in range(y.shape[0])]
    if sort_col is None:
        sort_col = line_labels[0]
    line(y=y, x=x, line_labels=line_labels, **kwargs)
    if len(y.shape)==1:
        df = pd.DataFrame({"0": to_numpy(y)}, index=x).sort_values("0", ascending=ascending)
    else:
        df = pd.DataFrame({line_labels[i]: to_numpy(y[i]) for i in range(y.shape[0])}, index=x).sort_values(sort_col, ascending=ascending)

    nutils.show_df(df.head(20))
    if return_df:
        return df
# plot_line_and_df(y=l28h17_path_scores[:, :5].T, x=path_end_df.query("layer>28").label, title="Path scores", line_labels=["A", "BC", "DFE", "GHI", "KLJ"])

# %%
layer, head = 20, 29
label = f"L{layer}H{head}"
clean_pattern = clean_cache["pattern", layer][:, head, :, :]
clean_grad_pattern = clean_grad_cache["pattern", layer][:, head, :, :]
corr_pattern = corr_cache["pattern", layer][:, head, :, :]
imshow([clean_pattern.mean(0), corr_pattern.mean(0), clean_pattern.mean(0)-corr_pattern.mean(0)], x=nutils.process_tokens_index(clean_tokens[0]), y=["word", ":", "Let0", "Let1"], title=f"Pattern {label}", facet_col=0, facet_labels=["clean", "corr", "diff"])
imshow(clean_pattern.std(0), x=nutils.process_tokens_index(clean_tokens[0]), y=["word", ":", "Let0", "Let1"], title=f"Pattern std {label}")
imshow((clean_pattern * clean_grad_pattern).mean(0), x=nutils.process_tokens_index(clean_tokens[0]), y=["word", ":", "Let0", "Let1"], title=f"Attribution pattern {label}")
head_diff_z = clean_cache["z", layer][:, -1, head, :] - corr_cache["z", layer][:, -1, head, :]
head_diff_result = head_diff_z.cuda() @ model.W_O[layer, head]
head_path_scores = (path_end_tensor[path_end_df.query("layer>"+str(layer)).index.values] * head_diff_result).sum(-1)
plot_line_and_df(head_path_scores.mean(-1), x=path_end_df.query("layer>"+str(layer)).label, title=f"L{layer}H{head} Path scores")
# %%
CCD_LABELS = ["clean", "corr", "diff"]
clean_resid_stack, resid_labels = clean_cache.decompose_resid(30, pos_slice=-1, return_labels=True)
corr_resid_stack, resid_labels = corr_cache.decompose_resid(30, pos_slice=-1, return_labels=True)
mlp30_in = corr_grad_cache["normalized", 30, "ln2"][:, -1, :] / clean_ln_scale[30, :, -1, :]
clean_mlp_path = (clean_resid_stack * mlp30_in).sum(-1).mean(-1)
corr_mlp_path = (corr_resid_stack * mlp30_in).sum(-1).mean(-1)
diff_mlp_path = clean_mlp_path - corr_mlp_path
plot_line_and_df([clean_mlp_path, corr_mlp_path, diff_mlp_path], resid_labels, line_labels=CCD_LABELS, refactor_decomp=True, sort_col="diff")
# %%
imshow(clean_cache["pattern", 28][:, 17, -1, -5:-3])
attn_diff = clean_cache["pattern", 28][:, 17, -1, -5] - clean_cache["pattern", 28][:, 17, -1, -4]
words = [s[1:] for s in model.to_str_tokens(clean_tokens[:, -7])]
l28h17attn_df = pd.DataFrame({
    "attn_diff": to_numpy(attn_diff),
    "word": words,
    "Let0": [w[0] for w in words],
    "Let0_vowel": [w[0] in "aeiou" for w in words],
    "Let1": [w[1] for w in words],
    "Let1_vowel": [w[1] in "aeiou" for w in words],
    "Let2": [w[2] for w in words],
    "Let2_vowel": [w[2] in "aeiou" for w in words],
})
print(l28h17attn_df.groupby("Let0_vowel")["attn_diff"].mean())
print(l28h17attn_df.groupby("Let1_vowel")["attn_diff"].mean())
print(l28h17attn_df.groupby("Let2_vowel")["attn_diff"].mean())

# %%
l28h17attn_df["mode"] = [("V" if a else "C")+("V" if b else "C")+("V" if c else "C") for a, b, c in zip(l28h17attn_df.Let0_vowel, l28h17attn_df.Let1_vowel, l28h17attn_df.Let2_vowel)]
l28h17attn_df.groupby("mode")["attn_diff"].mean()
l28h17attn_df["mode"] = [("V" if a else "C")+("V" if b else "C") for a, b, c in zip(l28h17attn_df.Let0_vowel, l28h17attn_df.Let1_vowel, l28h17attn_df.Let2_vowel)]
print(l28h17attn_df.groupby("mode")["attn_diff"].mean())
print(l28h17attn_df.groupby("mode")["attn_diff"].count())
# %%
nutils.show_df(l28h17attn_df.sort_values("attn_diff")[["attn_diff", "mode", "word", "Let1_vowel", "Let2_vowel"]])
# %%
l28h17attn_df.query("attn_diff>0.5 & Let1_vowel")["word"].to_list()
# %%
_, clean_cache_2 = model.run_with_cache(clean_tokens, names_filter=[utils.get_act_name("pattern", 28), utils.get_act_name("z", 28), utils.get_act_name("v", 28)])
pattern = clean_cache_2["pattern", 28][:, 17, :, :]
prompt_token_labels = nutils.process_tokens_index(clean_tokens[0])
prompt_token_labels = prompt_token_labels[:-7]+["word", ":", "Let0", "Let1", "Let2", "Let3", "Let4"]
imshow(pattern[:10, -7:, -7:], x=prompt_token_labels[-7:], y=prompt_token_labels[-7:], title="Pattern", facet_col=0, facet_labels=words[:10])
# %%
v = clean_cache_2["v", 28][:, -5:-3, 17, :]
v_dla = v @ model.W_O[28, 17] @ alpha_U
print(v_dla.shape)
temp_df = plot_line_and_df(v_dla[np.arange(128), :, clean_answer_index[:, 2]].T, x=nutils.process_tokens_index(words), return_df = True)
temp_df["is_vowel"] = [w[1] in "aeiou" for w in words]
temp_df.groupby("is_vowel").mean()
# imshow(v @ model.W_O[28, 17] @ alpha_U, facet_col=1, y=words, x=alphalist)
# %%

temp_prompts = make_kshot_prompts(512, 3)
temp_tokens = model.to_tokens(temp_prompts)
temp_words = model.to_str_tokens(temp_tokens[:, -7])
temp_answer_index = get_answer_index(temp_prompts)
_, temp_cache = model.run_with_cache(temp_tokens, names_filter=[utils.get_act_name("pattern", 28), utils.get_act_name("z", 28), utils.get_act_name("v", 28)])
v = temp_cache["v", 28][:, -5:-3, 17, :]
v_dla = v @ model.W_O[28, 17] @ alpha_U
print(v_dla.shape)
temp_df = plot_line_and_df(v_dla[np.arange(len(v_dla)), :, temp_answer_index[:, 2]].T, x=nutils.process_tokens_index(temp_words), return_df = True)
temp_df["is_vowel"] = [w[1] in "aeiou" for w in temp_words]
temp_df.groupby("is_vowel").mean()

# %%
v_real_dla = v_dla[np.arange(len(v_dla)), :, temp_answer_index[:, 2]]
v_real_dla_diff = v_real_dla[:, 1] - v_real_dla[:, 0]
attn_diff = temp_cache["pattern", 28][:, 17, -4, -4] - temp_cache["pattern", 28][:, 17, -4, -5]
scatter(x=v_real_dla_diff, y=attn_diff, xaxis="Logit Diff from value", yaxis="Attn Diff")
# %%
px.scatter(x=to_numpy(v_real_dla_diff), y=to_numpy(attn_diff), trendline="ols", labels={"x": "Logit Diff from value", "y": "Attn Diff"})
# %%
