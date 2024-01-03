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

# example_prompt = """The string " heaven" begins with the letter " H"
# The string " same" begins with the letter " S"
# The string " bath" begins with the letter \""""
# example_answer = "B"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# def create_full_spelling_prompt(
#     word: str, word_list: List[Tuple[str, str]], num_shots: int
# ) -> Tuple[str, str]:
#     """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
#     Tuples are of the form (word, spelling).
#     Creates a prompt that asks for the full spelling of a word.

#     Returns the prompt and the answer to the prompt."""
#     assert (
#         0 <= num_shots < len(word_list)
#     ), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
#     _, answer = [w for w in word_list if w[0] == word][
#         0
#     ]  # Assumes unique words in word_list. TODO: Improve?
#     word_list = [
#         item for item in word_list if item[0] != word
#     ]  # Remove any words that are the same as the word we want to spell.
#     prompt = ""
#     if num_shots > 0:
#         samples = random.sample(word_list, num_shots)
#         for sample in samples:
#             prompt += f"Q: How do you spell '{sample[0]}'? A: {sample[1]}\n\n"
#     prompt += f"Q: How do you spell '{word}'? A:"
#     return prompt, answer


# word = "cat"
# word_list = [
#     ("cat", "C A T"),
#     ("dog", "D O G"),
#     ("bird", "B I R D"),
#     ("mouse", "M O U S E"),
# ]
# num_shots = 3

# prompt, answer = create_full_spelling_prompt(word, word_list, num_shots)
# print(prompt)
# print(answer)
# %%

# example_prompt = """ string: S T R I N G
#  heaven: H E A V E N
#  xenograft: X E N O G R A F T"""
# logits, cache = model.run_with_cache(example_prompt)
# cv.logits.token_log_probs(model.to_tokens(example_prompt), model(example_prompt)[0].log_softmax(dim=-1), model.to_string)
# # %%
# for i, s in enumerate(model.to_str_tokens(example_prompt)):
#     print(i, s)
# %%
# resid_stack, labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=34, return_labels=True)
# T_dir = model.W_U[:, model.to_single_token(" T")]
# E_dir = model.W_U[:, model.to_single_token(" E")]
# diff_dir = T_dir - E_dir
# resid_stack = resid_stack.squeeze()
# line([resid_stack @ T_dir, resid_stack @ E_dir, resid_stack @ diff_dir], line_labels=["T", "E", "T-E"], x=labels, title="DLA on S in ESTABL")
# %%
# dla_df = pd.DataFrame({
#     "label": labels,
#     "Tdla": to_numpy(resid_stack @ T_dir),
#     "Edla": to_numpy(resid_stack @ E_dir),
#     "diff_dla": to_numpy(resid_stack @ diff_dir),
#     "is_head": [l.startswith("L") for l in labels],
# })
# dla_df
# # %%
# nutils.show_df(dla_df.sort_values("diff_dla", ascending=False).head(30))
# # %%
# key_heads = dla_df.query("is_head").sort_values("diff_dla", ascending=False).head(10)
# def unpack_label(label):
#     x = re.match(r'L(\d+)H(\d+)', label)
#     return int(x.group(1)), int(x.group(2))
# attns = []
# for i in key_heads.label.values:
#     L, H = unpack_label(i)
#     attn = cache["pattern", L][0, H, 34]
#     attns.append(attn)
# line(attns, x=nutils.process_tokens_index(example_prompt), line_labels=key_heads.label.values, title="Attention on S in ESTABL")
# %%
vocab_df = pd.DataFrame(
    {
        "token": np.arange(d_vocab),
        "string": model.to_str_tokens(np.arange(d_vocab)),
    }
)
vocab_df["is_alpha"] = vocab_df.string.str.match(r"^( ?)[a-z]+$")
vocab_df["is_word"] = vocab_df.string.str.match(r"^ [a-z]+$")
vocab_df["is_fragment"] = vocab_df.string.str.match(r"^[a-z]+$")
vocab_df["has_space"] = vocab_df.string.str.match(r"^ [A-Za-z]+$")
vocab_df["num_chars"] = vocab_df.string.apply(lambda n: len(n.strip()))
vocab_df
# %%
letters = [[] for _ in range(20)]
alphabet = "abcdefghijklmnopqrstuvwxyz"
for row in tqdm.tqdm(vocab_df.iterrows()):
    row = row[1]
    string = row.string.strip()
    for i in range(20):
        if not row.is_alpha or i >= len(string):
            letters[i].append(-1)
        else:
            letters[i].append(alphabet.index(string[i]))
# %%
letters_array = np.array(letters, dtype=np.int32)
(letters_array != -1).sum(-1)

# %%
vocab_df["let0"] = letters_array[0]
vocab_df["let1"] = letters_array[1]
vocab_df["let2"] = letters_array[2]
vocab_df["let3"] = letters_array[3]
vocab_df["let4"] = letters_array[4]
vocab_df["let5"] = letters_array[5]
vocab_df
# %%
sub_vocab_df = vocab_df.query("is_alpha & num_chars>=4")
# %%
eff_embed = model.W_E + model.blocks[0].mlp(model.blocks[0].ln2(model.W_E[None]))
eff_embed = eff_embed.squeeze()
eff_embed.shape
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

char_index = 1
col_label = f"let{char_index}"
X = to_numpy(eff_embed[sub_vocab_df.token.values])
y = sub_vocab_df[col_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
probe = LogisticRegression(max_iter=10)
probe.fit(X_train, y_train)
probe.score(X_test, y_test)
# %%
lp_test = probe.predict_log_proba(X_test)
clp_test = lp_test[np.arange(5242), y_test]
# clp_test.sort()
line(clp_test)
# %%
temp_df = pd.DataFrame(
    {
        "letter": y_test,
        "clp": to_numpy(clp_test),
        "max_lp": to_numpy(lp_test.max(-1)),
        "rank": (lp_test > clp_test[:, None]).sum(-1),
    }
)
temp_df
# %%
# alpha_U = model.W_U[:, [model.to_single_token(" " + c.upper()) for c in alphabet]]
# print(alpha_U.shape)
# head_probe_outs = eff_embed[sub_vocab_df.token.values] @ model.OV @ alpha_U
# head_probe_outs = head_probe_outs.AB
# # head_probe_outs = head_probe_outs[None, None]
# # %%
# imshow(
#     (
#         head_probe_outs.argmax(dim=-1)
#         == torch.tensor(sub_vocab_df.let0.values, device="cuda")
#     )
#     .float()
#     .mean(-1),
#     yaxis="Layer",
#     xaxis="Head",
#     title="Accuracy of Head Probes on First Letter",
# )
# imshow(
#     (
#         head_probe_outs.argmax(dim=-1)
#         == torch.tensor(sub_vocab_df.let1.values, device="cuda")
#     )
#     .float()
#     .mean(-1),
#     yaxis="Layer",
#     xaxis="Head",
#     title="Accuracy of Head Probes on Second Letter",
# )
# imshow(
#     (
#         head_probe_outs.argmax(dim=-1)
#         == torch.tensor(sub_vocab_df.let2.values, device="cuda")
#     )
#     .float()
#     .mean(-1),
#     yaxis="Layer",
#     xaxis="Head",
#     title="Accuracy of Head Probes on Third Letter",
# )
# imshow(
#     (
#         head_probe_outs.argmax(dim=-1)
#         == torch.tensor(sub_vocab_df.let3.values, device="cuda")
#     )
#     .float()
#     .mean(-1),
#     yaxis="Layer",
#     xaxis="Head",
#     title="Accuracy of Head Probes on Fourth Letter",
# )
# imshow(
#     (
#         head_probe_outs.argmax(dim=-1)
#         == torch.tensor(sub_vocab_df.let4.values, device="cuda")
#     )
#     .float()
#     .mean(-1),
#     yaxis="Layer",
#     xaxis="Head",
#     title="Accuracy of Head Probes on Fifth Letter",
# )
# %%
# letter = 0
# sub_vocab_df_array = torch.tensor(
#     np.stack(
#         [
#             sub_vocab_df.let0.values,
#             sub_vocab_df.let1.values,
#             sub_vocab_df.let2.values,
#             sub_vocab_df.let3.values,
#             sub_vocab_df.let4.values,
#         ],
#         axis=0,
#     ),
#     device="cuda",
# )
# x = (
#     (
#         (head_probe_outs.argmax(dim=-1) == letter)
#         == (sub_vocab_df_array[:, None, None, :] == letter)
#     )
#     .float()
#     .mean(-1)
# )
# line(x.reshape(5, -1))
# %%
# l = []
# for i in range(26):
#     l.append(X_train[:5000][y_train[:5000] == i].mean(0))
# per_token_ave = np.stack(l, axis=0)
# per_token_X_test = X_test @ per_token_ave.T
# temp_probe = LogisticRegression(max_iter=500)
# temp_probe.fit(X_train[5000:] @ per_token_ave.T, y_train[5000:])

# temp_probe.score(X_test @ per_token_ave.T, y_test)
# %%
short_chars_vocab_df = vocab_df.query("is_word & num_chars == 5")
short_chars_vocab_df


# %%
def make_single_prompt():
    word = short_chars_vocab_df.string.sample().item().strip()
    return f" {word}:" + "".join([f" {c.upper()}" for c in word.strip()])


def make_kshot_prompt(k=3):
    return "\n".join([make_single_prompt() for _ in range(k)])


def make_kshot_prompts(n=10, k=3):
    return [make_kshot_prompt(k) for _ in range(n)]


batch_size = 256
prompts = make_kshot_prompts(batch_size, 2)
tokens = model.to_tokens(prompts)
try:
    del logits, cache
except:
    pass
logits, cache = model.run_with_cache(tokens)

# %%
answer_tokens = torch.zeros((batch_size, 5), device="cuda", dtype=torch.int64) - 1
for i in range(batch_size):
    for j in range(5):
        answer_tokens[i, j] = alphabet.index(prompts[i][2 * j - 9].lower())
alpha_tokens = torch.tensor(
    [model.to_single_token(" " + c.upper()) for c in alphabet], device="cuda"
)
alpha_log_probs = logits.log_softmax(dim=-1)[:, -6:-1, alpha_tokens]
alpha_log_probs.gather(-1, answer_tokens[:, :, None]).squeeze().mean(0)
# %%
for i, s in enumerate(model.to_str_tokens(prompts[0])):
    print(i, s)
# %%
head_df = pd.DataFrame(
    {
        "head": [h for l in range(n_layers) for h in range(n_heads)],
        "layer": [l for l in range(n_layers) for h in range(n_heads)],
        "label": [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)],
    }
)
WORD = 9
for i in range(5):
    head_df[f"attn{i}"] = to_numpy(
        cache.stack_activation("pattern")[:, :, :, WORD + 1 + i, WORD].mean(1).flatten()
    )
# %%
nutils.show_df(head_df.sort_values("attn4", ascending=False).head(20))
# %%
alpha_U = model.W_U[:, alpha_tokens]
alpha_U_cent = alpha_U - alpha_U.mean(-1, keepdim=True)
resid_stack, labels = cache.decompose_resid(
    pos_slice=(-6, -1), apply_ln=True, return_labels=True
)
resid_stack_dla = resid_stack @ alpha_U_cent
# %%
line(
    resid_stack_dla.gather(
        -1,
        einops.repeat(answer_tokens, "x y -> component x y 1", component=len(labels)),
    )
    .squeeze()
    .mean(1)
    .T,
    x=labels,
)
# %%
attn_mask = np.array(["attn" in i for i in labels])
mlp_mask = np.array(["mlp" in i for i in labels])
line(
    resid_stack_dla[attn_mask]
    .gather(
        -1, einops.repeat(answer_tokens, "x y -> component x y 1", component=n_layers)
    )
    .squeeze()
    .mean(1)
    .T,
    # x=[i for i in labels],
    title="Attn layer DLA",
)
line(
    resid_stack_dla[mlp_mask]
    .gather(
        -1, einops.repeat(answer_tokens, "x y -> component x y 1", component=n_layers)
    )
    .squeeze()
    .mean(1)
    .T,
    # x=[i for i in labels],
    title="mlp layer DLA",
)

# %%
# Studying MLP per neuron DLA
layer = 30
neuron_acts = cache["post", layer][:, -6:-1, :]
W_out_alpha = model.blocks[layer].mlp.W_out @ alpha_U_cent
W_out_answer = W_out_alpha[:, answer_tokens]
per_neuron_dla = einops.einsum(
    W_out_answer, neuron_acts, "mlp batch letter, batch letter mlp -> letter batch mlp"
)
line(per_neuron_dla[2, :5])
# %%
line(per_neuron_dla[2, answer_tokens[:, 2] == 14])

# %%
win = model.W_in[30, :, 8924]
wout = model.W_out[30, 8924]
o_U = alpha_U_cent[:, 14]
print(f"{nutils.cos(win, o_U)=}")
print(f"{nutils.cos(win, wout)=}")
print(f"{nutils.cos(o_U, wout)=}")
line(
    wout @ alpha_U_cent,
    x=[i for i in alphabet],
    title="DLA of L30N8924 the Vowels and X neuron",
)
# %%
nutils.show_df(nutils.create_vocab_df(model.W_out[30, 8924] @ model.W_U).head(100))
# %%
neuron_dla_df = nutils.create_vocab_df(model.W_out[30, 8924] @ model.W_U)
neuron_dla_df["first_letter"] = neuron_dla_df.token.apply(
    lambda s: "-" if len(s) < 2 or s[0] != nutils.SPACE else s[1]
)
px.histogram(neuron_dla_df, x="logit", color="first_letter", barmode="overlay")

# %%
head_df = pd.DataFrame(
    {
        "head": [h for l in range(n_layers) for h in range(n_heads)],
        "layer": [l for l in range(n_layers) for h in range(n_heads)],
        "label": [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)],
    }
)
WORD = 9
for i in range(5):
    head_df[f"attn{i}"] = to_numpy(
        cache.stack_activation("pattern")[:, :, :, WORD + 1 + i, WORD].mean(1).flatten()
    )
head_df
# %%
z_stack = cache.stack_activation("z")[:, :, -6:-1, :, :]
print(z_stack.shape)
W_O_alpha = model.W_O @ alpha_U_cent
print(W_O_alpha.shape)
W_O_answer = W_O_alpha[..., answer_tokens]
print(W_O_answer.shape)
# %%
head_dla = (
    einops.einsum(
        z_stack,
        W_O_answer,
        1 / cache["scale"][:, -6:-1, 0],
        "layer batch letter head d_head, layer head d_head batch letter, batch letter -> letter layer head",
    )
    / batch_size
)
head_dla.shape
imshow(
    head_dla,
    facet_col=0,
    xaxis="Head",
    yaxis="Layer",
    title="Head DLA",
    facet_name="Layer",
)

# %%
for i in range(5):
    head_df[f"dla{i}"] = to_numpy(head_dla[i].flatten())
head_df["dla_max_abs"] = to_numpy(head_dla.max(dim=0).values.flatten().abs())
nutils.show_df(head_df.sort_values("dla_max_abs", ascending=False).head(20))
# %%
layer = 16
head = 5
imshow(
    cache["pattern", layer][0, head],
    x=nutils.process_tokens_index(tokens[0]),
    y=nutils.process_tokens_index(tokens[0]),
)
imshow(
    cache["pattern", layer][:, head].mean(0),
    x=nutils.process_tokens_index(tokens[0]),
    y=nutils.process_tokens_index(tokens[0]),
)
# %%
specific_head_dla = (
    z_stack[layer, :, 0, head, :] @ model.W_O[layer, head] @ alpha_U_cent
)
specific_head_dla.shape
# %%
alphalist = [i for i in alphabet]
line(
    specific_head_dla.T,
    x=nutils.process_tokens_index(model.to_str_tokens(tokens[:, WORD])),
    line_labels=alphalist,
)
# %%
mask = specific_head_dla.argmax(1) != answer_tokens[:, 0]
mask.shape
line(
    specific_head_dla.T[:, mask],
    x=nutils.process_tokens_index(model.to_str_tokens(tokens[mask, WORD])),
    line_labels=alphalist,
)
# %%
specific_head_probe_out = (
    eff_embed @ model.W_V[layer, head] @ model.W_O[layer, head] @ alpha_U_cent
)
specific_head_probe_out.shape
# %%
vocab_df_temp = copy.deepcopy(vocab_df)
vocab_df_temp["head_argmax"] = to_numpy(specific_head_probe_out.argmax(dim=1))
vocab_df_temp["is_first"] = vocab_df_temp["head_argmax"] == vocab_df_temp["let0"]
print(vocab_df_temp.query("is_word")["is_first"].mean())
px.box(vocab_df_temp.query("is_word"), x="num_chars", y="is_first")
# %%
x = []
for l in range(17):
    specific_head_probe_out = (
        cache["resid_pre", l][:, WORD, :]
        @ model.W_V[layer, head]
        @ model.W_O[layer, head]
        @ alpha_U_cent
    )
    x.append(
        (specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean().item()
    )
line(
    x,
    xaxis="Layer",
    title="Head L16H5 as a mechanistic probe on word token at resid_post at layer X",
)
# %%
specific_head_probe_out = (
    (
        cache["resid_pre", 0][:, WORD, :]
        + cache["attn_out", 0][:, WORD, :]
        + cache["mlp_out", 0][:, WORD, :]
    )
    @ model.W_V[layer, head]
    @ model.W_O[layer, head]
    @ alpha_U_cent
)
print((specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean())
specific_head_probe_out = (
    (
        cache["resid_pre", 0][:, WORD, :]
        # + cache["attn_out", 0][:, WORD, :]
        + cache["mlp_out", 0][:, WORD, :]
    )
    @ model.W_V[layer, head]
    @ model.W_O[layer, head]
    @ alpha_U_cent
)
print((specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean())
specific_head_probe_out = (
    (
        cache["resid_pre", 0][:, WORD, :]
        + cache["attn_out", 0][:, WORD, :]
        # + cache["mlp_out", 0][:, WORD, :]
    )
    @ model.W_V[layer, head]
    @ model.W_O[layer, head]
    @ alpha_U_cent
)
print((specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean())
specific_head_probe_out = (
    (
        # cache["resid_pre", 0][:, WORD, :]
        +cache["attn_out", 0][:, WORD, :]
        + cache["mlp_out", 0][:, WORD, :]
    )
    @ model.W_V[layer, head]
    @ model.W_O[layer, head]
    @ alpha_U_cent
)
print((specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean())
# %%
specific_head_probe_out = (
    (
        # cache["resid_pre", 0][:, WORD, :]
        # + cache["attn_out", 0][:, WORD, :]
        eff_embed[tokens[:, WORD]]
    )
    @ model.W_V[layer, head]
    @ model.W_O[layer, head]
    @ alpha_U_cent
)
print((specific_head_probe_out.argmax(1) == answer_tokens[:, 0]).float().mean())
# %%
sub_vocab_df = vocab_df.query("is_word & num_chars>=4")
head_probe_outs = eff_embed[sub_vocab_df.token.values] @ model.OV @ alpha_U
head_probe_outs = head_probe_outs.AB

# %%
imshow(
    (
        head_probe_outs.argmax(dim=-1)
        == torch.tensor(sub_vocab_df.let0.values, device="cuda")
    )
    .float()
    .mean(-1),
    yaxis="Layer",
    xaxis="Head",
    title="Accuracy of Head Probes on First Letter",
)
imshow(
    (
        head_probe_outs.argmax(dim=-1)
        == torch.tensor(sub_vocab_df.let1.values, device="cuda")
    )
    .float()
    .mean(-1),
    yaxis="Layer",
    xaxis="Head",
    title="Accuracy of Head Probes on Second Letter",
)
imshow(
    (
        head_probe_outs.argmax(dim=-1)
        == torch.tensor(sub_vocab_df.let2.values, device="cuda")
    )
    .float()
    .mean(-1),
    yaxis="Layer",
    xaxis="Head",
    title="Accuracy of Head Probes on Third Letter",
)
imshow(
    (
        head_probe_outs.argmax(dim=-1)
        == torch.tensor(sub_vocab_df.let3.values, device="cuda")
    )
    .float()
    .mean(-1),
    yaxis="Layer",
    xaxis="Head",
    title="Accuracy of Head Probes on Fourth Letter",
)
imshow(
    (
        head_probe_outs.argmax(dim=-1)
        == torch.tensor(sub_vocab_df.let4.values, device="cuda")
    )
    .float()
    .mean(-1),
    yaxis="Layer",
    xaxis="Head",
    title="Accuracy of Head Probes on Fifth Letter",
)

# %%
line(
    eff_embed[model.to_single_token(" broch")]
    @ model.W_V[20, 6]
    @ model.W_O[20, 6]
    @ alpha_U_cent,
    x=alphalist,
)
# %%
sub_vocab_df = vocab_df.query("is_word & num_chars>=5").reset_index(drop=True)
sub_eff_embed = eff_embed[sub_vocab_df.index]
sub_eff_embed.shape, sub_vocab_df.shape
# %%
# for letter in range(5):
#     train_indices = (sub_vocab_df.num_chars != 5).values
#     test_indices = ~train_indices

#     X_train = sub_eff_embed[train_indices]
#     X_test = sub_eff_embed[test_indices]
#     y_train = torch.tensor(sub_vocab_df[f"let{letter}"].values[train_indices]).cuda()
#     y_test = torch.tensor(sub_vocab_df[f"let{letter}"].values[test_indices]).cuda()
#     torch.set_grad_enabled(True)
#     probe = nn.Linear(d_model, 26).cuda()
#     optim = torch.optim.AdamW(
#         probe.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1
#     )

#     def train_step():
#         loss = (
#             -probe(X_train).log_softmax(dim=-1)[np.arange(len(X_train)), y_train].mean()
#         )
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         return loss.item()

#     @torch.no_grad()
#     def test_loss_and_acc():
#         log_probs = probe(X_test).log_softmax(dim=-1)

#         clps = -log_probs[np.arange(len(X_test)), y_test].mean()
#         acc = (log_probs.argmax(dim=-1) == y_test).float().mean()
#         return acc.item(), clps.item()

#     print()
#     print()
#     print()
#     print("Letter", letter)
#     print(train_step())
#     print(test_loss_and_acc())

#     train_losses = []
#     test_losses = []
#     for i in tqdm.trange(2000):
#         train_losses.append(train_step())
#         if i % 100 == 0:
#             test_losses.append(test_loss_and_acc())
#             print(test_losses[-1], train_losses[-1])
#     print(test_loss_and_acc())
#     torch.save(probe.state_dict(), f"/workspace/GPT-Spelling-Bee/probe_let{letter}.pt")
# %%
probes = []
for i in range(5):
    probe = nn.Linear(d_model, 26).cuda()
    probe.load_state_dict(torch.load(f"/workspace/GPT-Spelling-Bee/probe_let{i}.pt"))
    probes.append(probe)
probew = torch.stack([p.weight for p in probes], dim=0)
probeb = torch.stack([p.bias for p in probes], dim=0)
probew.shape, probeb.shape
# %%
line(probeb, x=alphalist, line_labels=[f"let{i}" for i in range(5)])
# %%
sub_vocab_df.let0.value_counts().sort_index()
# %%
for i in range(5):
    px.scatter(
        x=sub_vocab_df[f"let{i}"].value_counts().sort_index().values,
        y=to_numpy(probeb[i]),
        trendline="ols",
    ).show()
# px.scatter(x=np.log10(sub_vocab_df.let0.value_counts().sort_index().values), y=to_numpy(probeb[0]), trendline="ols").show()
# %%
x = []
for i in range(4):
    x.append(sub_vocab_df[f"let{i}"].value_counts().sort_index().values)
line(x, x=alphalist, title="Frequency Statistics by letter")
# %%
tokens_5 = sub_vocab_df.token[sub_vocab_df.num_chars == 5].values
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


records = []
for probe_i in range(5):
    probe = probes[probe_i]
    for letter_i in range(5):
        record = {
            "probe": probe_i,
            "letter": letter_i,
            "label": f"P{probe_i}Let{letter_i}",
        }
        record.update(get_probe_metrics(probe, letter_i))
        records.append(record)
temp_df = pd.DataFrame(records)
px.line(temp_df, x="label", y=temp_df.columns[3:])
# %%
within_probe_sims = torch.stack(
    [nutils.cos(probew[i][:, None, :], probew[i][None, :, :]) for i in range(5)]
)
imshow(within_probe_sims, facet_col=0, x=alphalist, y=alphalist)

# %%
between_probe_sims = torch.stack(
    [nutils.cos(probew[:, i][:, None, :], probew[:, i][None, :, :]) for i in range(26)]
)
imshow(between_probe_sims, facet_col=0, facet_labels=alphalist)

# %%
x = [f"{a}/{i}" for i in range(5) for a in alphabet]
imshow(
    nutils.cos(
        probew.reshape(-1, d_model)[:, None, :], probew.reshape(-1, d_model)[None, :, :]
    ),
    x=x,
    y=x,
)
# %%
from scipy.cluster import hierarchy


def plot_cosine_similarity_heatmap(
    data_array, restricted_labels, title="Pairwise Cosine Similarity Heatmap"
):
    # data_array = df.to_numpy()
    linkage = hierarchy.linkage(data_array)
    dendrogram = hierarchy.dendrogram(linkage, no_plot=True, color_threshold=-np.inf)
    reordered_ind = dendrogram["leaves"]
    # if reorder:
    #     # reorder df by ind
    #     df = df.iloc[reordered_ind, reordered_ind]
    #     # data_array = df.to_numpy()

    # plot the cosine similarity matrix
    fig = px.imshow(
        df.values,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        labels={"color": "Cosine Similarity"},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(restricted_labels))),
        ticktext=restricted_labels,
        showgrid=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(restricted_labels))),
        ticktext=restricted_labels,
        showgrid=False,
    )

    # don't show axes if there are more than 20 rows
    # if df.shape[0] > 20:
    #     fig.update_xaxes(
    #         visible=False,
    #     )
    #     fig.update_yaxes(
    #         visible=False,
    #     )
    return fig


# plot_cosine_similarity_heatmap(, x)
arr = to_numpy(
    nutils.cos(
        probew.reshape(-1, d_model)[:, None, :], probew.reshape(-1, d_model)[None, :, :]
    )
)
labels = x
linkage = hierarchy.linkage(arr)
dendrogram = hierarchy.dendrogram(linkage, no_plot=True, color_threshold=-np.inf)
reordered_ind = list(reversed(dendrogram["leaves"]))
reordered_labels = [labels[i] for i in reordered_ind]
reordered_arr = arr[reordered_ind][:, reordered_ind]
imshow(reordered_arr, x=reordered_labels, y=reordered_labels)
# %%
layer = 30
head = 4
pattern = cache["pattern", layer][:, head, :, :]
imshow(pattern[:5], facet_col=0, facet_labels=model.to_str_tokens(tokens[:5, WORD]), x=nutils.process_tokens_index(tokens[0]), y=nutils.process_tokens_index(tokens[0]))
imshow(pattern.mean(0), x=nutils.process_tokens_index(tokens[0]), y=nutils.process_tokens_index(tokens[0]))
imshow(pattern.std(0), x=nutils.process_tokens_index(tokens[0]), y=nutils.process_tokens_index(tokens[0]))

# %%
letter = 2
batch_word_list = nutils.process_tokens_index(model.to_str_tokens(tokens[:, WORD]))
log_probs = logits.log_softmax(dim=-1)[:, -6:-1, alpha_tokens][:, letter, :]
line(log_probs.gather(-1, answer_tokens).T, x=batch_word_list)
# %%
batch_alpha_U = alpha_U_cent[:, answer_tokens]
for letter in range(5):
    resid_stack, resid_labels = cache.accumulated_resid(pos_slice=-6+letter, apply_ln=True, return_labels=True)
    
    logit_lens = einops.einsum(resid_stack, batch_alpha_U, "layer batch d_model, d_model batch letter -> letter layer batch")
    imshow(logit_lens, facet_col=0, y=resid_labels, x=batch_word_list, title=f"Logit lens for letter {letter}")
    line(logit_lens.mean(-1), title=f"Logit lens for letter {letter}")
# %%
# %%
z_stack = cache.stack_activation("z")[:, :, -6:-1, :, :]
print(z_stack.shape)
W_O_alpha = model.W_O @ alpha_U_cent
print(W_O_alpha.shape)
W_O_answer = W_O_alpha[..., answer_tokens]
print(W_O_answer.shape)
head_dla = (
    einops.einsum(
        z_stack,
        W_O_answer,
        1 / cache["scale"][:, -6:-1, 0],
        "layer batch letter head d_head, layer head d_head batch letter2, batch letter2 -> layer head letter letter2",
    )
    / batch_size
)
head_dla = head_dla.reshape(-1, 5, 5)
head_labels = model.all_head_labels()
head_dla.shape

# %%
mlp_out = cache.stack_activation("mlp_out")[:, :, -6:-1, :] 
mlp_dla = einops.einsum(mlp_out, batch_alpha_U, 1 / cache["scale"][:, -6:-1, 0], "layer batch letter d_model, d_model batch letter2, batch letter2 -> layer letter letter2")/ batch_size
mlp_labels = [f"MLP{i}" for i in range(n_layers)]
dla_labels = head_labels + mlp_labels
dla = torch.cat([head_dla, mlp_dla], dim=0)
dla.shape
# %%
dla_diffs = []
for letter in range(4):
    dla_diff = dla[:, letter+1, letter]
    # dla_diff = dla[:, letter, letter] - dla[:, letter+1, letter]
    dla_diffs.append(dla_diff)
line(dla_diffs, x=dla_labels, line_labels=[f"Let{i}" for i in range(4)], title="DLA for Let k at pos k+1")
temp_df = pd.DataFrame(index=dla_labels)
for i in range(4):
    temp_df[f"Let{i}P{i}"] = to_numpy(dla[:, i, i])
for i in range(4):
    temp_df[f"Let{i}P{i+1}"] = to_numpy(dla[:, i+1, i])
for i in range(4):
    temp_df[f"Let{i}Diff"] = to_numpy(dla[:, i, i] - dla[:, i+1, i])
temp_df["abs_max"] = temp_df.abs().max(axis=1)
nutils.show_df(temp_df.sort_values("abs_max", ascending=False).head(30))





# %%
# %%
clean_logits = model(clean_tokens)
corr_logits = model(corr_tokens)

line([clean_logits[np.arange(batch_size), -1, clean_answers] - clean_logits[np.arange(batch_size), -1, corr_answers], corr_logits[np.arange(batch_size), -1, clean_answers] - corr_logits[np.arange(batch_size), -1, corr_answers]])

CLEAN_BASELINE_DIFF = (clean_logits[np.arange(batch_size), -1, clean_answers] - clean_logits[np.arange(batch_size), -1, corr_answers]).mean()
CORR_BASELINE_DIFF = (corr_logits[np.arange(batch_size), -1, clean_answers] - corr_logits[np.arange(batch_size), -1, corr_answers]).mean()
print("Clean Baseline Diff:", CLEAN_BASELINE_DIFF)
print("Corr Baseline Diff:", CORR_BASELINE_DIFF)

def metric(logits):
    logit_diff = (logits[np.arange(batch_size), -1, clean_answers] - logits[np.arange(batch_size), -1, corr_answers]).mean()
    return (logit_diff - CORR_BASELINE_DIFF) / (CLEAN_BASELINE_DIFF - CORR_BASELINE_DIFF)

filter_not_qkv_input = lambda name: "_input" not in name and "_result" not in name and "_attn_in" not in name and "_mlp_in" not in name
def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    torch.set_grad_enabled(True)
    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    torch.set_grad_enabled(False)
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)