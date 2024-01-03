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
    
# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n.endswith("resid_pre"))
corr_logits, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: n.endswith("resid_pre"))

def replace_resid_hook(resid_pre, hook, new_cache, pos):
    resid_pre[:, pos, :] = new_cache[hook.name][:, pos, :]
    return resid_pre

records = []
pos_labels = {
    -7: "word",
    -6: ":",
    -5: "Let0",
    -4: "Let1",
    -3: "Let2",
    -2: "Let3",
}
for pos in tqdm.trange(-7, -3):
    for layer in tqdm.trange(n_layers):
        noised_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("resid_pre", layer),
                    partial(replace_resid_hook, pos=pos, new_cache=corr_cache),
                )
            ]
        )
        record = {
            "pos": pos,
            "pos_label": pos_labels[pos],
            "layer": layer,
            "mode": "noising",
        }
        record.update(patching_metrics(noised_logits, clean_answer_index))
        records.append(record)
        denoised_logits = model.run_with_hooks(
            corr_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("resid_pre", layer),
                    partial(replace_resid_hook, pos=pos, new_cache=clean_cache),
                )
            ]
        )
        record = {
            "pos": pos,
            "pos_label": pos_labels[pos],
            "layer": layer,
            "mode": "denoising",
        }
        record.update(patching_metrics(denoised_logits, clean_answer_index))
        records.append(record)
resid_patch_df = pd.DataFrame(records)
resid_patch_df.to_csv("resid_patch_df.csv")
px.line(resid_patch_df, x="layer", facet_col="mode", color="pos_label", y="Let2PrevDiff", title="Residual stream patching on Let2P2 - Let1P2")

# %%
franken_tokens = clean_tokens.clone()
franken_tokens[:, 9] = corr_tokens[:, 9]
franken_logits = model(franken_tokens)
metric(franken_logits, clean_answer_index), metric(franken_logits, corr_answer_index)
# %%
clean_batch_alpha_U = alpha_U_cent[:, clean_answer_index]
clean_batch_alpha_U
# %%
clean_baseline = patching_metrics(clean_logits, clean_answer_index)
corr_baseline = patching_metrics(corr_logits, clean_answer_index)

metric = "Let0NextDiff"
print(f"{clean_baseline[metric]=}")
print(f"{corr_baseline[metric]=}")
px.line(resid_patch_df, x="layer", facet_col="mode", color="pos_label", y=metric, title="Residual stream patching on "+metric)
# %%
for metric in clean_baseline.keys():
    print(f"{clean_baseline[metric]=}")
    print(f"{corr_baseline[metric]=}")
    px.line(resid_patch_df, x="layer", facet_col="mode", color="pos_label", y=metric, title="Residual stream patching on "+metric).show()
# %%
pos = -7
layer = 0
noised_logits = model.run_with_hooks(
    clean_tokens,
    fwd_hooks = [
        (
            utils.get_act_name("resid_pre", layer),
            partial(replace_resid_hook, pos=pos, new_cache=corr_cache),
        )
    ]
)
record = {
    "pos": pos,
    "pos_label": pos_labels[pos],
    "layer": layer,
    "mode": "noising",
}
record.update(patching_metrics(noised_logits, clean_answer_index))
x = (noised_logits.log_softmax(dim=-1)[:, -4, alpha_tokens][np.arange(batch_size), clean_answer_index[:, 2]])
pos = -4
layer = 0
noised_logits = model.run_with_hooks(
    clean_tokens,
    fwd_hooks = [
        (
            utils.get_act_name("resid_pre", layer),
            partial(replace_resid_hook, pos=pos, new_cache=corr_cache),
        )
    ]
)
record = {
    "pos": pos,
    "pos_label": pos_labels[pos],
    "layer": layer,
    "mode": "noising",
}
record.update(patching_metrics(noised_logits, clean_answer_index))
y = (noised_logits.log_softmax(dim=-1)[:, -4, alpha_tokens][np.arange(batch_size), clean_answer_index[:, 2]])
line([x, y])
scatter(x=x, y=y)
# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda n: n.endswith("mlp_out"))
corr_logits, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: n.endswith("mlp_out"))

def replace_mlp_hook(mlp_out, hook, new_cache, pos):
    mlp_out[:, pos, :] = new_cache[hook.name][:, pos, :]
    return mlp_out

records = []
pos_labels = {
    -7: "word",
    -6: ":",
    -5: "Let0",
    -4: "Let1",
    -3: "Let2",
    -2: "Let3",
}
for pos in tqdm.trange(-7, -1):
    for layer in tqdm.trange(n_layers):
        noised_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("mlp_out", layer),
                    partial(replace_mlp_hook, pos=pos, new_cache=corr_cache),
                )
            ]
        )
        record = {
            "pos": pos,
            "pos_label": pos_labels[pos],
            "layer": layer,
            "mode": "noising",
        }
        record.update(patching_metrics(noised_logits, clean_answer_index))
        records.append(record)
        denoised_logits = model.run_with_hooks(
            corr_tokens,
            fwd_hooks = [
                (
                    utils.get_act_name("mlp_out", layer),
                    partial(replace_mlp_hook, pos=pos, new_cache=clean_cache),
                )
            ]
        )
        record = {
            "pos": pos,
            "pos_label": pos_labels[pos],
            "layer": layer,
            "mode": "denoising",
        }
        record.update(patching_metrics(denoised_logits, clean_answer_index))
        records.append(record)
mlp_patch_df = pd.DataFrame(records)
mlp_patch_df.to_csv("mlp_patch_df.csv")
px.line(mlp_patch_df, x="layer", facet_col="mode", color="pos_label", y="Let2PrevDiff", title="MLP stream patching on Let2P2 - Let1P2")
# %%
