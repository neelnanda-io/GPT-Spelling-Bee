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
# %%

records = []
blank_probe = nn.Linear(d_model, 26).cuda()
blank_probe.bias[:] = 0.
for layer in tqdm.trange(n_layers):
    for head in range(n_heads):
        blank_probe.weight[:] = (model.W_V[layer, head] @ model.W_O[layer, head] @ alpha_U_cent).T
        for letter in range(5):
            record = {
                "layer": layer,
                "head": head,
                "letter": letter,
                "label": f"L{layer}H{head}"
            }

            record.update(get_probe_metrics(blank_probe, letter))
            records.append(record)
head_probe_df = pd.DataFrame(records)
for metric in ["acc", "loss", "median_rank"]:
    px.line(head_probe_df, x="label", color="letter", y=metric, title=metric).show()
# %%
nutils.show_df(head_probe_df.query("letter!=0").sort_values("median_rank", ascending=True).head(30))
# %%
nutils.show_df(head_probe_df.pivot_table(columns="letter", index="label", values="median_rank").astype(int).sort_values(1).head(10))
nutils.show_df(head_probe_df.pivot_table(columns="letter", index="label", values="median_rank").astype(int).sort_values(2).head(10))
nutils.show_df(head_probe_df.pivot_table(columns="letter", index="label", values="median_rank").astype(int).sort_values(0).head(10))
# %%
px.scatter(head_probe_df.pivot_table(columns="letter", index="label", values="loss"), x=1, y=2, color=0, hover_name=head_probe_df.query("letter==0").label.values)
# %%
head_probe_df.query("label=='L19H2'")
# %%
head_probe_loss_df = head_probe_df.pivot_table(columns="letter", index="label", values="loss")
head_probe_loss_df["min"] = head_probe_loss_df.min(axis=1)
nutils.show_df(head_probe_loss_df.sort_values("min").head(20))
# %%

nutils.show_df(head_probe_df.query("label=='L16H5'"))
nutils.show_df(head_probe_df.query("label=='L13H14'"))
nutils.show_df(head_probe_df.query("label=='L13H8'"))
# %%
nutils.show_df(head_probe_df.query("label=='L14H10'"))
nutils.show_df(head_probe_df.query("label=='L20H6'"))
nutils.show_df(head_probe_df.query("label=='L17H9'"))
nutils.show_df(head_probe_df.query("label=='L19H11'"))

# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: not "hook_pre" in n and not "hook_post" in n)
corr_logits, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: not "hook_pre" in n and not "hook_post" in n)

# %%
patching_metrics(clean_logits, clean_answer_index)
# %%
records = []
for letter in range(5):
    clean_resid_pre = clean_cache["resid_pre", 30][:, -6+letter, :]
    corr_resid_pre = corr_cache["resid_pre", 30][:, -6+letter, :]
    answer_index_letter = clean_answer_index[:, letter]
    W_U_letter = alpha_U_cent[:, answer_index_letter]
    W_U_letter = W_U_letter / W_U_letter.norm(dim=0, keepdim=True)
    W_U_letter = W_U_letter.T
    print(W_U_letter.shape)
    clean_U_proj = (clean_resid_pre * W_U_letter).sum(-1, keepdim=True) * W_U_letter
    corr_U_proj = (corr_resid_pre * W_U_letter).sum(-1, keepdim=True) * W_U_letter

    denoised_resid_pre = corr_resid_pre + clean_U_proj - corr_U_proj
    clean_ln_scale = clean_cache["scale", 30, "ln2"][:, -6+letter, :]
    corr_ln_scale = corr_cache["scale", 30, "ln2"][:, -6+letter, :]
    denoised_dla = ((model.blocks[30].mlp(denoised_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    clean_dla = ((model.blocks[30].mlp(clean_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    corr_dla = ((model.blocks[30].mlp(corr_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    corr_dla_2 = ((model.blocks[30].mlp(corr_resid_pre[:, None, :] / corr_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    noised_resid_pre = clean_resid_pre - clean_U_proj + corr_U_proj
    noised_dla = ((model.blocks[30].mlp(noised_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    print(f"{noised_dla.mean().item()=}")
    zero_abl_resid_pre = clean_resid_pre - clean_U_proj
    zero_abl_dla = ((model.blocks[30].mlp(zero_abl_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    print(f"{zero_abl_dla.mean().item()=}")
    big_neg_resid_pre = clean_resid_pre - W_U_letter * 20
    big_neg_dla = ((model.blocks[30].mlp(big_neg_resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
    record = {
        "letter": letter,
        "noised_dla": noised_dla.mean().item(),
        "big_neg_dla": big_neg_dla.mean().item(),
        "denoised_dla": denoised_dla.mean().item(),
        "clean_dla": clean_dla.mean().item(),
        "corr_dla": corr_dla.mean().item(),
        "corr_dla_2": corr_dla_2.mean().item(),
    }
    records.append(record)
nutils.show_df(pd.DataFrame(records).set_index("letter").T)
# %%
resid_pre = clean_resid_pre - W_U_letter * 100
dla = ((model.blocks[30].mlp(resid_pre[:, None, :] / clean_ln_scale[:, None, :]).squeeze() * W_U_letter).sum(-1))
print(dla.mean().item())
# %%
patching_metrics(clean_cache["resid_pre", 30] @ model.W_U)
# %%
clean_alpha_U_cent = alpha_U_cent[:, clean_answer_index]
clean_mlp_out = clean_cache["mlp_out", 30][:, -6:-1, :]
corr_mlp_out = corr_cache["mlp_out", 30][:, -6:-1, :]
imshow(einops.einsum(clean_alpha_U_cent, clean_mlp_out, "d_model batch out_letter, batch pos d_model -> batch pos out_letter").mean(0), yaxis="Position", xaxis="Letter", title="Clean MLP Out", y=[":"]+[f"Let{i}" for i in range(4)])
imshow(einops.einsum(clean_alpha_U_cent, clean_mlp_out - clean_mlp_out.mean(0, keepdim=True), "d_model batch out_letter, batch pos d_model -> batch pos out_letter").mean(0), yaxis="Position", xaxis="Letter", title="Clean MLP Out Mean Centered", y=[":"]+[f"Let{i}" for i in range(4)])
imshow(einops.einsum(clean_alpha_U_cent, corr_mlp_out, "d_model batch out_letter, batch pos d_model -> batch pos out_letter").mean(0), yaxis="Position", xaxis="Letter", title="Corr MLP Out", y=[":"]+[f"Let{i}" for i in range(4)])
imshow(einops.einsum(clean_alpha_U_cent, corr_mlp_out - corr_mlp_out.mean(0, keepdim=True), "d_model batch out_letter, batch pos d_model -> batch pos out_letter").mean(0), yaxis="Position", xaxis="Letter", title="corr MLP Out Mean Centered", y=[":"]+[f"Let{i}" for i in range(4)])
# %%
num_batches = 50
act_names = [
    utils.get_act_name("mlp_out", 30),
    utils.get_act_name("mlp_out", 31),
    utils.get_act_name("post", 30),
    utils.get_act_name("resid_pre", 30),
    utils.get_act_name("resid_pre", 31),
]
# def get_many_acts(act, layer, num_batches=50):
many_prompts = make_kshot_prompts(batch_size*num_batches, 3)
many_tokens = model.to_tokens(many_prompts)
many_answer_index = get_answer_index(many_prompts)

temp_cache_list = []
for i in tqdm.trange(num_batches):
    temp_tokens = many_tokens[i*batch_size:(i+1)*batch_size]
    _, temp_cache = model.run_with_cache(temp_tokens, names_filter=lambda name: name in act_names, return_type=None)
    temp_cache_list.append(temp_cache)
many_cache = ActivationCache({
    name: torch.cat([temp_cache[name] for temp_cache in temp_cache_list], dim=0)
    for name in act_names
}, model)
many_cache["mlp_out", 30].shape




# %%
many_cache_cpu = ActivationCache({
    name: torch.cat([temp_cache[name] for temp_cache in temp_cache_list], dim=0).detach().cpu()
    for name in act_names
}, model)
del temp_cache_list
# %%
many_cache = ActivationCache({
    name: value[:, -6:-1, :].cuda()
    for name, value in many_cache_cpu.cache_dict.items()
}, model)
# %%
neuron_acts = many_cache["post", 30]
W_out_alpha_U = model.W_out[30] @ alpha_U_cent
neuron_wdla = einops.rearrange(W_out_alpha_U[:, many_answer_index], "d_mlp batch letter -> batch letter d_mlp")
neuron_dla = neuron_wdla * neuron_acts
neuron_dla.shape
# %%
records = []
ks = [1, 3, 5, 10, 50, 100]
for letter in range(5):
    record = {
        "letter": letter,
        "k": d_mlp,
    }
    neuron_dla_letter = neuron_dla[:, letter, :]
    record["dla"] = neuron_dla_letter.sum(-1).mean(0).item()
    record["median_dla"] = neuron_dla_letter.sum(-1).median(0).values.item()
    records.append(record)
    for k in ks:
        record = {
            "letter": letter,
            "k": k,
        }
        record["dla"] = neuron_dla_letter.topk(k, dim=-1).values.sum(-1).mean(0).item()
        record["median_dla"] = neuron_dla_letter.topk(k, dim=-1).values.sum(-1).median(0).values.item()
        records.append(record)
dla_sparsity_df = pd.DataFrame(records)
dla_sparsity_df.to_csv("dla_sparsity_df.csv")
px.line(dla_sparsity_df, y="dla", color="k", x="letter", title="DLA Sparsity").show()
px.line(dla_sparsity_df, y="median_dla", color="k", x="letter", title="Median DLA Sparsity").show()
        
        
# %%
nutils.show_df(nutils.create_vocab_df(model.W_out[30, 1499] @ model.W_U).head(50))
# %%
def f(x):
    print(((many_answer_index==17)[:, :] * x).sum(0) / (many_answer_index==17)[:, :].sum(0))
    print(((many_answer_index!=17)[:, :] * x).sum(0) / (many_answer_index!=17)[:, :].sum(0))
f(neuron_dla[..., 1499])
# %%
print(((many_answer_index==17)[:, :] * neuron_acts[:, :, 1499]).sum(0) / (many_answer_index==17)[:, :].sum(0))
print(((many_answer_index!=17)[:, :] * neuron_acts[:, :, 1499]).sum(0) / (many_answer_index!=17)[:, :].sum(0))
# %%
r_unembed = alpha_U[:, 17]
r_unembed = r_unembed / r_unembed.norm()
win = model.W_in[30, :, 1499]
win = win / win.norm()

f(many_cache["resid_pre", 30] @ r_unembed)
f(many_cache["resid_pre", 30] @ win)

new_resid_pre = many_cache["resid_pre", 30]
new_resid_pre = new_resid_pre - (new_resid_pre @ r_unembed)[:, :, None] * r_unembed[None, None, :]
f(new_resid_pre @ win)
new_resid_pre = many_cache["resid_pre", 30]
new_resid_pre = new_resid_pre - 20 * r_unembed
f(new_resid_pre @ win)

# %%
r_prompts = []
not_r_prompts = []
for prompt in many_prompts:
    if prompt[-1]=="R":
        r_prompts.append(prompt)
    else:
        not_r_prompts.append(prompt)
r_prompts = r_prompts[:64]
r_tokens = model.to_tokens(r_prompts)
r_answer_index = get_answer_index(r_prompts)
not_r_prompts = not_r_prompts[:64]
not_r_tokens = model.to_tokens(not_r_prompts)
not_r_answer_index = get_answer_index(not_r_prompts)

r_logits, r_cache = model.run_with_cache(r_tokens, names_filter=lambda n: not "hook_pre" in n and not "hook_post" in n)
not_r_logits, not_r_cache = model.run_with_cache(not_r_tokens, names_filter=lambda n: not "hook_pre" in n and not "hook_post" in n)
# %%
r_resid_stack, resid_labels = r_cache.decompose_resid(30, True, apply_ln=True, pos_slice=-2, return_labels=True)
not_r_resid_stack, resid_labels = not_r_cache.decompose_resid(30, True, apply_ln=True, pos_slice=-2, return_labels=True)
r_nla = r_resid_stack @ win
not_r_dla = not_r_resid_stack @ win
line([r_nla.mean(1), not_r_dla.mean(1), r_nla.mean(1) - not_r_dla.mean(1)], x=resid_labels, line_labels=["R", "Not R", "Diff"])
diff_dna = r_nla.mean(1) - not_r_dla.mean(1)
# %%
r_unembed = alpha_U_cent[:, 17]
r_unembed = r_unembed / r_unembed.norm()
win_excl = win - (win @ r_unembed) * r_unembed
r_nla = r_resid_stack @ win_excl
not_r_dla = not_r_resid_stack @ win_excl
line([r_nla.mean(1), not_r_dla.mean(1), r_nla.mean(1) - not_r_dla.mean(1)], x=resid_labels, line_labels=["R", "Not R", "Diff"], title="L30N1499 DLA Excl R Unembed")
diff_dna_excl = r_nla.mean(1) - not_r_dla.mean(1)
# %%
r_unembed = alpha_U_cent[:, 17]
r_unembed = r_unembed / r_unembed.norm()
win_excl = win - (win @ r_unembed) * r_unembed
alpha_ave = alpha_U.mean(-1)
alpha_ave = alpha_ave / alpha_ave.norm()
win_excl = win_excl - (win_excl @ alpha_ave) * alpha_ave
r_nla = r_resid_stack @ win_excl
not_r_dla = not_r_resid_stack @ win_excl
line([r_nla.mean(1), not_r_dla.mean(1), r_nla.mean(1) - not_r_dla.mean(1)], x=resid_labels, line_labels=["R", "Not R", "Diff"], title="L30N1499 DLA Excl R Unembed")
diff_dna_excl_excl = r_nla.mean(1) - not_r_dla.mean(1)

# %%
scatter(x=diff_dna_excl_excl, y=diff_dna, color=diff_dna_excl, hover=resid_labels, xaxis="Excluding R Unembed", yaxis="Vanilla DNA", include_diag=True)
# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n.endswith("out"))
corr_logits, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: n.endswith("out"))

def replace_mlp_in_hook(mlp_input, hook, from_cache, to_cache, act_name, src_layer):
    mlp_input = mlp_input.clone()
    mlp_input += from_cache[act_name, src_layer][:, :, :] - to_cache[act_name, src_layer][:, :, :]
    return mlp_input

records = []
pos_labels = {
    -7: "word",
    -6: ":",
    -5: "Let0",
    -4: "Let1",
    -3: "Let2",
    -2: "Let3",
}
# for pos in tqdm.trange(-7, -3):
for layer in tqdm.trange(30):
    for layer_type in ["attn", "mlp"]:
        noised_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("mlp_in", 30),
                    partial(replace_mlp_in_hook, from_cache=corr_cache, to_cache=clean_cache, act_name=layer_type+"_out", src_layer=layer),
                )
            ]
        )
        record = {
            "layer": layer,
            "layer_type": layer_type,
            "mode": "noising",
        }
        record.update(patching_metrics(noised_logits, clean_answer_index))
        records.append(record)
        denoised_logits = model.run_with_hooks(
            corr_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("mlp_in", 30),
                    partial(replace_mlp_in_hook, from_cache=clean_cache, to_cache=corr_cache, act_name=layer_type+"_out", src_layer=layer)
                )
            ]
        )
        record = {
            "layer": layer,
            "layer_type": layer_type,
            "mode": "denoising",
        }
        record.update(patching_metrics(denoised_logits, clean_answer_index))
        records.append(record)
mlp_30_path_patch_df = pd.DataFrame(records)
mlp_30_path_patch_df.to_csv("mlp_30_path_patch_df.csv")
px.line(mlp_30_path_patch_df, x="layer", facet_col="mode", color="layer_type", y="Let2PrevDiff", title="mlp_30_path patching")
# %%
for i in range(5):
    px.line(mlp_30_path_patch_df, x="layer", facet_col="mode", color="layer_type", y="Let"+str(i), title="mlp_30_path patching Let"+str(i)).show()
# %%
