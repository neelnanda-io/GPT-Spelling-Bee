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
layer = 14
head = 10
W_V = model.W_V[layer, head]
data = eff_embed_5 @ W_V
train_indices = np.random.uniform(size=(len(eff_embed_5),)) < 0.8
train_data = data[train_indices].data.clone()
test_data = data[~train_indices].data.clone()
vocab_df_5 = copy.deepcopy(vocab_df.query("is_word & num_chars==5"))
vocab_df_5["is_train"] = train_indices
# frequent_let0 = np.arange(26)[
#     vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index() > 50
# ]
# frequent_let0_letters = [alphalist[i] for i in frequent_let0]
# frequent_let1 = np.arange(26)[
#     vocab_df.query("num_chars==5 & is_word").let1.value_counts().sort_index() > 50
# ]
# frequent_let1_letters = [alphalist[i] for i in frequent_let1]
x = vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index()
frequent_let0 = x.index[x > 50]
frequent_let0_letters = [alphalist[i] for i in frequent_let0]
x = vocab_df.query("num_chars==5 & is_word").let1.value_counts().sort_index()
frequent_let1 = x.index[x > 50]
frequent_let1_letters = [alphalist[i] for i in frequent_let1]
x = vocab_df.query("num_chars==5 & is_word").let2.value_counts().sort_index()
frequent_let2 = x.index[x > 50]
frequent_let2_letters = [alphalist[i] for i in frequent_let2]
x = vocab_df.query("num_chars==5 & is_word").let3.value_counts().sort_index()
frequent_let3 = x.index[x > 50]
frequent_let3_letters = [alphalist[i] for i in frequent_let3]
x = vocab_df.query("num_chars==5 & is_word").let4.value_counts().sort_index()
frequent_let4 = x.index[x > 50]
frequent_let4_letters = [alphalist[i] for i in frequent_let4]


# %%
# first_letters = vocab_df.query("num_chars==5 & is_word").let0.values
# train_first = first_letters[train_indices]
# test_first = first_letters[~train_indices]

# second_letters = vocab_df.query("num_chars==5 & is_word").let1.values
# train_second = second_letters[train_indices]
# test_second = second_letters[~train_indices]
# third_letters = vocab_df.query("num_chars==5 & is_word").let2.values
# train_third = third_letters[train_indices]
# test_third = third_letters[~train_indices]

# %%
from sae_utils import AutoEncoder

cfg = {
    "seed": 49,
    # "batch_size": 4096,
    # "buffer_mult": 384,
    "lr": 1e-5,
    # "num_tokens": int(2e9),
    "l1_coeff": 1e-3,
    "beta1": 0.9,
    "beta2": 0.99,
    "act_size": 80,
    "dict_size": 64,
    "enc_dtype": "fp32",
    "device": "cuda:0",
}
temp_encoder = AutoEncoder(cfg)
loss, reconstr, acts, l2, l1 = temp_encoder(test_data)
print(loss, l2, l1)
# loss.backward()
# %%
torch.set_grad_enabled(True)
encoder = AutoEncoder(cfg)
encoder_optim = torch.optim.Adam(
    encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
)
l1_losses = []
l2_losses = []
l0_losses = []

test_l1_losses = []
test_l2_losses = []
test_l0_losses = []
# %%

for i in tqdm.trange(20000):
    loss, reconstr, acts, l2, l1 = encoder(train_data)
    loss.backward()
    encoder.remove_parallel_decoder_grad()
    encoder_optim.step()
    encoder.set_decoder_norm_to_unit_norm()
    encoder_optim.zero_grad()
    l1_losses.append(l1.item())
    l2_losses.append(l2.item())
    l0_losses.append((acts != 0).float().sum(-1).mean().item())
    with torch.no_grad():
        loss, reconstr, acts, l2, l1 = encoder(test_data)
        test_l1_losses.append(l1.item())
        test_l2_losses.append(l2.item())
        test_l0_losses.append((acts != 0).float().sum(-1).mean().item())
    if (i + 1) % 10000 == 0:
        line(
            [
                l0_losses,
                l1_losses,
                l2_losses,
                test_l0_losses,
                test_l1_losses,
                test_l2_losses,
            ],
            line_labels=["L0", "L1", "L2", "test L0", "test L1", "test L2"],
        )
# line([l0_losses, l1_losses, l2_losses, test_l0_losses, test_l1_losses, test_l2_losses], line_labels=["L0", "L1", "L2", "test L0", "test L1", "test L2"])
# %%
train_loss, train_reconstr, train_acts, train_l2, train_l1 = encoder(train_data)
test_loss, test_reconstr, test_acts, test_l2, test_l1 = encoder(test_data)
merged_acts = torch.zeros((len(vocab_df_5), train_acts.shape[1])).cuda()
merged_acts[train_indices] = train_acts
merged_acts[~train_indices] = test_acts
histogram((train_acts != 0).float().mean(0))
histogram((test_acts != 0).float().mean(0))
# %%
train_one_hot_first = torch.tensor(
    (train_first[:, None] == np.arange(26)[None, :])
).cuda()
train_ave_act_by_letter = (
    (train_one_hot_first[:, None, :] * train_acts[:, :, None]).sum(0)
    / train_one_hot_first.sum(0)
).T
line(
    train_ave_act_by_letter[frequent_let0],
    line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0],
    title="Train Ave Act by Letter",
)

train_act_firings = train_acts > 0
train_ave_act_firing_by_letter = (
    (train_one_hot_first[:, None, :] * train_act_firings[:, :, None]).sum(0)
    / train_one_hot_first.sum(0)
).T
line(
    train_ave_act_firing_by_letter[frequent_let0],
    line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0],
    title="Train Frac is Firing by Letter",
)

test_one_hot_first = torch.tensor(
    (test_first[:, None] == np.arange(26)[None, :])
).cuda()
test_ave_act_by_letter = (
    (test_one_hot_first[:, None, :] * test_acts[:, :, None]).sum(0)
    / (test_one_hot_first.sum(0) + 1e-5)
).T
line(
    test_ave_act_by_letter[frequent_let0],
    line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0],
    title="test Ave Act by Letter",
)

test_act_firings = test_acts > 0
test_ave_act_firing_by_letter = (
    (test_one_hot_first[:, None, :] * test_act_firings[:, :, None]).sum(0)
    / (test_one_hot_first.sum(0) + 1e-5)
).T
line(
    test_ave_act_firing_by_letter[frequent_let0],
    line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0],
    title="test Frac is Firing by Letter",
)
# %%
test_one_hot_first = torch.tensor(
    (test_first[:, None] == np.arange(26)[None, :])
).cuda()
test_ave_act_by_letter = (
    (test_one_hot_first[:, None, :] * test_acts[:, :, None]).sum(0)
    / (test_one_hot_first.sum(0) + 1e-5)
).T
line(test_ave_act_by_letter, line_labels=alphalist, title="test Ave Act by Letter")
# %%
imshow(train_ave_act_by_letter, y=alphalist)
# %%
px.scatter(
    x=train_ave_act_by_letter.max(-1).values.cpu().detach().numpy(),
    y=test_ave_act_by_letter.max(-1).values.cpu().detach().numpy(),
).show()
px.scatter(
    x=vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index().values,
    y=test_ave_act_by_letter.max(-1).values.cpu().detach().numpy(),
    hover_name=alphalist,
).show()
# %%
px.scatter(
    x=vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index().values,
    y=test_ave_act_firing_by_letter.max(-1).values.cpu().detach().numpy(),
    hover_name=alphalist,
).show()

# %%
x = test_ave_act_by_letter.argmax(dim=0).sort().indices
imshow(test_ave_act_by_letter[:, x])
# %%
import umap
import hdbscan

reducer = umap.UMAP(
    n_neighbors=3,
    min_dist=0.06,
    n_components=2,
    metric="euclidean",
    random_state=42,
)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)


ummap_result = reducer.fit_transform(encoder.W_dec.detach().cpu())
clusterer.fit(ummap_result)
# print(clusterer.labels_)


fig = px.scatter(
    ummap_result,
    x=0,
    y=1,
    color=[str(i) for i in clusterer.labels_],
    hover_data=[[f"feature_{i}" for i in range(len(encoder.W_dec))]],
    opacity=0.5,
    template="plotly",
)

# make points larger
fig.update_traces(marker=dict(size=12))

# make it wide and tall
fig.update_layout(height=800, width=1200)

fig.show()

# %%
cluster_id = 0
cluster_features = np.arange(512)[clusterer.labels_ == cluster_id]
line(
    train_ave_act_by_letter[:, cluster_features],
    line_labels=alphalist,
    x=[str(i) for i in cluster_features],
)
line(
    test_ave_act_by_letter[:, cluster_features],
    line_labels=alphalist,
    x=[str(i) for i in cluster_features],
)
line(
    train_ave_act_firing_by_letter[:, cluster_features],
    line_labels=alphalist,
    x=[str(i) for i in cluster_features],
)
line(
    test_ave_act_firing_by_letter[:, cluster_features],
    line_labels=alphalist,
    x=[str(i) for i in cluster_features],
)

# %%
imshow(
    nutils.cos(
        encoder.W_dec[cluster_features, None, :],
        encoder.W_dec[None, cluster_features, :],
    )
)
imshow(
    nutils.cos(
        encoder.W_enc.T[cluster_features, None, :],
        encoder.W_enc.T[None, cluster_features, :],
    )
)
# %%

vocab_df_5
train_strings = vocab_df_5.query("is_train").string.values
train_strings
# %%
temp_df = copy.deepcopy(
    vocab_df_5.query("is_train")[["token", "string", "let0", "let1", "let2"]]
)
for c, i in enumerate(cluster_features):
    temp_df[f"f{c}"] = to_numpy(train_acts[:, i])
nutils.show_df(temp_df.sort_values("f1", ascending=False).head(25))
# %%
nutils.show_df(temp_df.sort_values("f1", ascending=False).head(25))
nutils.show_df(temp_df.sort_values("f5", ascending=False).head(25))
nutils.show_df(temp_df.sort_values("f7", ascending=False).head(25))
nutils.show_df(temp_df.sort_values("f9", ascending=False).head(25))
# %%
cluster_train_acts = train_acts[:, cluster_features]
cluster_train_firing = cluster_train_acts > 0
imshow(
    (
        (cluster_train_firing[:, :, None] == cluster_train_firing[:, None, :])
        * (cluster_train_firing[:, :, None] | cluster_train_firing[:, None, :])
    )
    .float()
    .sum(0)
    / (cluster_train_firing[:, :, None] | cluster_train_firing[:, None, :]).sum(0)
)


# %%
def f(s):
    s = s.strip()
    if s[0] != "f":
        return "not f"
    else:
        if s[1] in "aeiou":
            return "f vowel"
        else:
            return "f consonant"


temp_df["mode"] = temp_df.string.apply(f)
nutils.show_df(
    temp_df.groupby("mode")[[f"f{i}" for i in range(len(cluster_features))]].mean()
)


# %%
def f(s):
    s = s.strip()
    if s[0] != "f":
        return "not f"
    else:
        return f"f{s[1]}"
        # if s[1] in "aeiou":
        #     return "f vowel"
        # else:
        #     return "f consonant"


temp_df["mode"] = temp_df.string.apply(f)
nutils.show_df(
    temp_df.groupby("mode")[[f"f{i}" for i in range(len(cluster_features))]].mean()
)
# %%
nutils.show_df(
    temp_df.groupby("mode")[[f"f{i}" for i in range(len(cluster_features))]].mean().T
)

# %%
train_df = vocab_df_5.query("is_train")
temp_df = pd.DataFrame(to_numpy(train_acts), index=train_df.string.values)
temp_df["let1"] = train_df.let1.values
nutils.show_df(temp_df.groupby("let1")[[i for i in range(512)]].mean())
temp_df2 = temp_df.groupby("let1")[[i for i in range(512)]].mean().T
let1_frequent = np.arange(26)[train_df.let1.value_counts().sort_index() > 40]
temp_df2 = temp_df2[let1_frequent]
temp_df2["max"] = temp_df2.max(axis=1)
nutils.show_df(temp_df2.sort_values("max", ascending=False).head(20))

# %%
temp_df = pd.DataFrame(to_numpy(train_acts > 0), index=train_df.string.values)
temp_df["let1"] = train_df.let1.values
nutils.show_df(temp_df.groupby("let1")[[i for i in range(512)]].mean())
temp_df2 = temp_df.groupby("let1")[[i for i in range(512)]].mean().T
let1_frequent = np.arange(26)[train_df.let1.value_counts().sort_index() > 40]
temp_df2 = temp_df2[let1_frequent]
temp_df2["max"] = temp_df2.max(axis=1)
nutils.show_df(temp_df2.sort_values("max", ascending=False).head(20))


# %%
def show_max_act(index, return_df=False):
    max_act_df = copy.deepcopy(vocab_df_5).drop(
        ["is_alpha", "is_word", "is_fragment", "has_space", "num_chars", "let5"], axis=1
    )
    acts = np.zeros(len(vocab_df_5))
    acts[train_indices] = to_numpy(train_acts[:, index])
    acts[~train_indices] = to_numpy(test_acts[:, index])
    max_act_df["act"] = acts
    max_act_df = max_act_df.sort_values("act", ascending=False)
    nutils.show_df(max_act_df.head(25))
    if return_df:
        return max_act_df


# # %%
# def f(s):
#     s = s.strip()
#     if s[0] == "a" and s[1] == "n":
#         return "an"
#     elif s[0] == "i" and s[1] == "n":
#         return "in"
#     elif s[0] == "a":
#         return "a"
#     elif s[0] == "i":
#         return "i"
#     elif s[1] == "n":
#         return "_n"
#     else:
#         return "-"


# temp_df = copy.deepcopy(vocab_df_5)
# temp_df["mode"] = temp_df.string.apply(f)
# flabels = []
# for i in range(merged_acts.shape[1]):
#     temp_df[f"f{i}"] = to_numpy(merged_acts[:, i])
#     flabels.append(f"f{i}")
# x = temp_df.groupby("mode")[flabels].mean()
# line(x.values, line_labels=x.index)
# %%
vocab_df_5["let0_freq"] = vocab_df_5.let0.apply(lambda n: n in frequent_let0)
vocab_df_5["let1_freq"] = vocab_df_5.let1.apply(lambda n: n in frequent_let1)
# %%
merged_acts = torch.zeros((len(vocab_df_5), train_acts.shape[1])).cuda()
merged_acts[train_indices] = train_acts
merged_acts[~train_indices] = test_acts

acts_df = pd.DataFrame(to_numpy(merged_acts))
temp_df = copy.deepcopy(vocab_df_5).reset_index(drop=True)
acts_df = pd.concat([temp_df, acts_df], axis=1)

firing_df = pd.DataFrame(to_numpy(merged_acts > 0))
temp_df = copy.deepcopy(vocab_df_5).reset_index(drop=True)
firing_df = pd.concat([temp_df, firing_df], axis=1)
acts_df
# %%
frequent_let0_letters = [alphalist[i] for i in frequent_let0]
frequent_let1_letters = [alphalist[i] for i in frequent_let1]
# %%
d_sae = train_acts.shape[1]
line(
    acts_df.query("let0_freq").groupby("let0")[list(range(d_sae))].mean().values,
    title="Ave act let0",
    line_labels=frequent_let0_letters,
)
line(
    firing_df.query("let0_freq").groupby("let0")[list(range(d_sae))].mean().values,
    title="Ave firing let0",
    line_labels=frequent_let0_letters,
)

line(
    acts_df.query("let1_freq").groupby("let1")[list(range(d_sae))].mean().values,
    title="Ave act let1",
    line_labels=frequent_let1_letters,
)
line(
    firing_df.query("let1_freq").groupby("let1")[list(range(d_sae))].mean().values,
    title="Ave firing let1",
    line_labels=frequent_let1_letters,
)

# %%
d_sae = encoder.W_enc.shape[1]
any_pos_diff = np.zeros((26, d_sae))

for i in range(26):
    filt = acts_df.string.str.contains(alphalist[i])
    ave_on = acts_df[filt][list(range(64))].mean(axis=0).values
    ave_off = acts_df[~filt][list(range(64))].mean(axis=0).values
    ave_diff = ave_on - ave_off
    any_pos_diff[i] = ave_diff
line(any_pos_diff, line_labels=alphalist)
# %%
each_pos_diff = np.zeros((5, 26, d_sae))

frequent_lets = [
    frequent_let0,
    frequent_let1,
    frequent_let2,
    frequent_let3,
    frequent_let4,
]
for pos in range(5):
    for i in range(26):
        # if i not in frequent_lets[pos]:
        #     continue
        filt = acts_df[f"let{pos}"] == i
        ave_on = acts_df[filt][list(range(64))].mean(axis=0).values
        ave_on = np.nan_to_num(ave_on)
        ave_off = acts_df[~filt][list(range(64))].mean(axis=0).values
        ave_diff = ave_on - ave_off
        each_pos_diff[pos, i] = ave_diff
# %%
feature_df = pd.DataFrame(
    {
        "0_max": each_pos_diff[0].max(axis=0),
        "0_max_letter": [alphalist[i] for i in each_pos_diff[0].argmax(axis=0)],
        "1_max": each_pos_diff[1].max(axis=0),
        "1_max_letter": [alphalist[i] for i in each_pos_diff[1].argmax(axis=0)],
        "2_max": each_pos_diff[2].max(axis=0),
        "2_max_letter": [alphalist[i] for i in each_pos_diff[2].argmax(axis=0)],
        "3_max": each_pos_diff[3].max(axis=0),
        "3_max_letter": [alphalist[i] for i in each_pos_diff[3].argmax(axis=0)],
        "4_max": each_pos_diff[4].max(axis=0),
        "4_max_letter": [alphalist[i] for i in each_pos_diff[4].argmax(axis=0)],
        "any_max": any_pos_diff.max(axis=0),
        "any_max_letter": [alphalist[i] for i in any_pos_diff.argmax(axis=0)],
    }
)
nutils.show_df(feature_df)
# %%
