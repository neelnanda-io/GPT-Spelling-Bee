# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

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

example_prompt = """The string " heaven" begins with the letter " H"
The string " same" begins with the letter " S"
The string " bath" begins with the letter \""""
example_answer = "B"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

def create_full_spelling_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuples are of the form (word, spelling).
    Creates a prompt that asks for the full spelling of a word.

    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    _, answer = [w for w in word_list if w[0] == word][0] # Assumes unique words in word_list. TODO: Improve?
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: How do you spell '{sample[0]}'? A: {sample[1]}\n\n"
    prompt += f"Q: How do you spell '{word}'? A:"
    return prompt, answer


word = 'cat'
word_list = [('cat', 'C A T'), ('dog', 'D O G'), ('bird', 'B I R D'), ('mouse', 'M O U S E')]
num_shots = 3

prompt, answer = create_full_spelling_prompt(word, word_list, num_shots)
print(prompt)
print(answer)
# %%
import circuitsvis as cv
example_prompt = """ string: S T R I N G
 heaven: H E A V E N
 antidisestablishmentarianism: A N T I D I S E S T A B L I S H M E N T A R I A N I S M"""
logits, cache = model.run_with_cache(example_prompt)
cv.logits.token_log_probs(model.to_tokens(example_prompt), model(example_prompt)[0].log_softmax(dim=-1), model.to_string)
# %%
for i, s in enumerate(model.to_str_tokens(example_prompt)):
    print(i, s)
# %%
resid_stack, labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=34, return_labels=True)
T_dir = model.W_U[:, model.to_single_token(" T")]
E_dir = model.W_U[:, model.to_single_token(" E")]
diff_dir = T_dir - E_dir
resid_stack = resid_stack.squeeze()
line([resid_stack @ T_dir, resid_stack @ E_dir, resid_stack @ diff_dir], line_labels=["T", "E", "T-E"], x=labels, title="DLA on S in ESTABL")
# %%
dla_df = pd.DataFrame({
    "label": labels,
    "Tdla": to_numpy(resid_stack @ T_dir),
    "Edla": to_numpy(resid_stack @ E_dir),
    "diff_dla": to_numpy(resid_stack @ diff_dir),
    "is_head": [l.startswith("L") for l in labels],
})
dla_df
# %%
nutils.show_df(dla_df.sort_values("diff_dla", ascending=False).head(30))
# %%
key_heads = dla_df.query("is_head").sort_values("diff_dla", ascending=False).head(10)
def unpack_label(label):
    x = re.match(r'L(\d+)H(\d+)', label)
    return int(x.group(1)), int(x.group(2))
attns = []
for i in key_heads.label.values:
    L, H = unpack_label(i)
    attn = cache["pattern", L][0, H, 34]
    attns.append(attn)
line(attns, x=nutils.process_tokens_index(example_prompt), line_labels=key_heads.label.values, title="Attention on S in ESTABL")
# %%
vocab_df = pd.DataFrame({
    "token": np.arange(d_vocab),
    "string": model.to_str_tokens(np.arange(d_vocab)),
})
vocab_df["is_alpha"] = vocab_df.string.str.match(r'^( ?)[a-z]+$')
vocab_df["is_word"] = vocab_df.string.str.match(r'^ [a-z]+$')
vocab_df["is_fragment"] = vocab_df.string.str.match(r'^[a-z]+$')
vocab_df["has_space"] = vocab_df.string.str.match(r'^ [A-Za-z]+$')
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
(letters_array!=-1).sum(-1)

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
eff_embed = model.W_E + model.blocks[0].mlp(model.blocks[0].ln2(model.W_E[None] + model.blocks[0].attn.b_O))
eff_embed = eff_embed.squeeze()
eff_embed.shape
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
char_index = 0
col_label = f"let{char_index}"
X = to_numpy(eff_embed[sub_vocab_df.token.values])
y = sub_vocab_df[col_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
probe = LogisticRegression(max_iter=100)
probe.fit(X_train, y_train)
probe.score(X_test, y_test)
# %%
lp_test = probe.predict_log_proba(X_test)
clp_test = lp_test[np.arange(5242), y_test]
# clp_test.sort()
line(clp_test)
# %%
temp_df = pd.DataFrame({
    "letter": y_test,
    "clp": to_numpy(clp_test),
    "max_lp": to_numpy(lp_test.max(-1)),
    "rank": (lp_test>clp_test[:, None]).sum(-1),
})
temp_df
# %%
