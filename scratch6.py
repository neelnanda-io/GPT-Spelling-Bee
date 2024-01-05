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
corr_prompts = [i[:36] + j[36:] for i, j in zip(clean_prompts, corr_prompts)]
print(corr_prompts[0], clean_prompts[0])
clean_tokens = model.to_tokens(clean_prompts)
clean_answer_index = get_answer_index(clean_prompts)

corr_tokens = model.to_tokens(corr_prompts)
corr_answer_index = get_answer_index(corr_prompts)


# %%
def metric(logits, answer_index, normalize=True):
    if len(logits.shape) == 3:
        logits = logits[:, -4, :]
    log_probs = logits.log_softmax(dim=-1)
    alpha_log_probs = log_probs[:, alpha_tokens]
    # for i in range(5):
    #     print(alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, i]].mean())
    clps = (
        alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, 2]]
        - alpha_log_probs[np.arange(len(alpha_log_probs)), answer_index[:, 1]]
    )
    loss = clps.mean()
    if normalize:
        return (loss - CORR_BASELINE) / (CLEAN_BASELINE - CORR_BASELINE)
    else:
        return loss


clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)
CLEAN_BASELINE = metric(clean_logits, clean_answer_index, False)
CORR_BASELINE = metric(corr_logits, clean_answer_index, False)


# %%
def patching_metrics(logits, answer_index):
    if logits.shape[1] != 5:
        logits = logits[:, -6:-1, :]
    log_probs = logits.log_softmax(dim=-1)
    alpha_log_probs = log_probs[:, :, alpha_tokens]
    metrics = {}
    for i in range(5):
        metrics[f"prob_letter{i}"] = (
            alpha_log_probs[:, i, :].exp().sum(-1).mean(0).item()
        )
    answer_log_probs = alpha_log_probs[
        np.arange(len(alpha_log_probs))[:, None], :, answer_index
    ]
    for i in range(5):
        metrics[f"Let{i}"] = answer_log_probs[:, i, i].mean().item()
        metrics[f"Let{i}Acc"] = (
            (alpha_log_probs[:, i, :].max(dim=-1).values == answer_log_probs[:, i, i])
            .float()
            .mean()
            .item()
        )
    for i in range(1, 5):
        metrics[f"Let{i}Prev"] = answer_log_probs[:, i - 1, i].mean().item()
        metrics[f"Let{i}PrevDiff"] = metrics[f"Let{i}"] - metrics[f"Let{i}Prev"]
    for i in range(4):
        metrics[f"Let{i}Next"] = answer_log_probs[:, i + 1, i].mean().item()
        metrics[f"Let{i}NextDiff"] = metrics[f"Let{i}"] - metrics[f"Let{i}Next"]
    return metrics


patching_metrics(clean_logits, clean_answer_index)
resid_patch_df = pd.read_csv("resid_patch_df.csv", index_col=0)
mlp_patch_df = pd.read_csv("mlp_patch_df.csv", index_col=0)
attn_patch_df = pd.read_csv("attn_patch_df.csv", index_col=0)
mlp_30_path_patch_df = pd.read_csv("mlp_30_path_patch_df.csv", index_col=0)
# %%
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=lambda n: not "hook_pre" in n and not "resid_mid" in n
)
corr_logits, corr_cache = model.run_with_cache(
    corr_tokens, names_filter=lambda n: not "hook_pre" in n and not "resid_mid" in n
)
# %%
x = []
for letter in range(5):
    print(letter)
    probew_norm = nutils.normalise(probew[letter])
    print(f"{probew_norm.shape=}")

    clean_word_token = clean_tokens[:, -7]
    clean_eff_embed = eff_embed[clean_word_token]
    corr_word_token = corr_tokens[:, -7]
    corr_eff_embed = eff_embed[corr_word_token]

    clean_probew = probew_norm[clean_answer_index[:, letter]]
    corr_probew = probew_norm[corr_answer_index[:, letter]]

    clean_proj = (clean_eff_embed * clean_probew).sum(-1, keepdim=True) * clean_probew
    corr_proj = (corr_eff_embed * corr_probew).sum(-1, keepdim=True) * corr_probew

    diff_proj = corr_proj - clean_proj

    def apply_diff_hook(resid_post_0, hook):
        resid_post_0[:, -7, :] += diff_proj
        return resid_post_0

    def corr_attn_0(attn_out_0, hook):
        attn_out_0[:, -7, :] = corr_cache["attn_out", 0][:, -7, :]
        return attn_out_0

    clean_logits = model(clean_tokens)
    corr_logits = model(corr_tokens)
    patched_logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[
            (utils.get_act_name("resid_post", 0), apply_diff_hook),
            (utils.get_act_name("attn_out", 0), corr_attn_0),
        ],
    )

    clean_clean_lp = clean_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    clean_corr_lp = clean_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]
    corr_clean_lp = corr_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    corr_corr_lp = corr_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]
    patched_clean_lp = patched_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    patched_corr_lp = patched_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]

    print(
        (patched_corr_lp.mean() - clean_corr_lp.mean())
        / (corr_corr_lp.mean() - clean_corr_lp.mean())
    )
    print(
        (patched_clean_lp.mean() - clean_clean_lp.mean())
        / (corr_clean_lp.mean() - clean_clean_lp.mean())
    )
    x.append(
        (
            (
                (patched_corr_lp.mean() - clean_corr_lp.mean())
                / (corr_corr_lp.mean() - clean_corr_lp.mean())
            ).item(),
            (
                (patched_clean_lp.mean() - clean_clean_lp.mean())
                / (corr_clean_lp.mean() - clean_clean_lp.mean())
            ).item(),
        )
    )

    print(f"{corr_corr_lp.mean().item()=}")
    print(f"{patched_corr_lp.mean().item()=}")
    print(f"{clean_clean_lp.mean().item()=}")
    print(f"{patched_clean_lp.mean().item()=}")
    print(f"{corr_clean_lp.mean().item()=}")
    print(f"{clean_corr_lp.mean().item()=}")

    line(
        [
            clean_clean_lp,
            clean_corr_lp,
            corr_clean_lp,
            corr_corr_lp,
            patched_clean_lp,
            patched_corr_lp,
        ],
        x=nutils.process_tokens_index(clean_tokens[:, -7]),
        line_labels=[
            "clean_clean_lp",
            "clean_corr_lp",
            "corr_clean_lp",
            "corr_corr_lp",
            "patched_clean_lp",
            "patched_corr_lp",
        ],
    )
line(to_numpy(x).T)
# %%
histogram(patched_corr_lp - corr_corr_lp)
# %%
x = []
for letter in range(5):
    print(letter)
    probew_norm = nutils.normalise(probew[letter])
    print(f"{probew_norm.shape=}")

    clean_word_token = clean_tokens[:, -7]
    clean_eff_embed = eff_embed[clean_word_token]
    corr_word_token = corr_tokens[:, -7]
    corr_eff_embed = eff_embed[corr_word_token]

    clean_probew = probew_norm[clean_answer_index[:, letter]]
    corr_probew = probew_norm[corr_answer_index[:, letter]]

    clean_proj = (clean_eff_embed * clean_probew).sum(-1, keepdim=True) * clean_probew
    corr_proj = (corr_eff_embed * corr_probew).sum(-1, keepdim=True) * corr_probew

    diff_proj = corr_proj - clean_proj

    def replace_resid_post(resid_post_0, hook):
        resid_post_0[:, -7, :] = corr_cache["resid_post", 0][:, -7, :]
        return resid_post_0

    clean_logits = model(clean_tokens)
    corr_logits = model(corr_tokens)
    patched_logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[
            (utils.get_act_name("resid_post", 0), replace_resid_post),
            # (utils.get_act_name("attn_out", 0), corr_attn_0),
        ],
    )

    clean_clean_lp = clean_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    clean_corr_lp = clean_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]
    corr_clean_lp = corr_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    corr_corr_lp = corr_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]
    patched_clean_lp = patched_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, clean_tokens[:, -5 + letter]
    ]
    patched_corr_lp = patched_logits.log_softmax(dim=-1)[
        np.arange(batch_size), -6 + letter, corr_tokens[:, -5 + letter]
    ]

    print(
        (patched_corr_lp.mean() - clean_corr_lp.mean())
        / (corr_corr_lp.mean() - clean_corr_lp.mean())
    )
    print(
        (patched_clean_lp.mean() - clean_clean_lp.mean())
        / (corr_clean_lp.mean() - clean_clean_lp.mean())
    )
    x.append(
        [
            (
                (patched_corr_lp.mean() - clean_corr_lp.mean())
                / (corr_corr_lp.mean() - clean_corr_lp.mean())
            ).item(),
            (
                (patched_clean_lp.mean() - clean_clean_lp.mean())
                / (corr_clean_lp.mean() - clean_clean_lp.mean())
            ).item(),
        ]
    )

    print(f"{corr_corr_lp.mean().item()=}")
    print(f"{patched_corr_lp.mean().item()=}")
    print(f"{clean_clean_lp.mean().item()=}")
    print(f"{patched_clean_lp.mean().item()=}")
    print(f"{corr_clean_lp.mean().item()=}")
    print(f"{clean_corr_lp.mean().item()=}")

    line(
        [
            clean_clean_lp,
            clean_corr_lp,
            corr_clean_lp,
            corr_corr_lp,
            patched_clean_lp,
            patched_corr_lp,
        ],
        x=nutils.process_tokens_index(clean_tokens[:, -7]),
        line_labels=[
            "clean_clean_lp",
            "clean_corr_lp",
            "corr_clean_lp",
            "corr_corr_lp",
            "patched_clean_lp",
            "patched_corr_lp",
        ],
    )
line(to_numpy(x).T)
# %%
