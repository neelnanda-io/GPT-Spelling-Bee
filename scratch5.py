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
layer = 16
head = 5
W_V = model.W_V[layer, head]
data = eff_embed_5 @ W_V
train_indices = np.random.uniform(size=(len(eff_embed_5),)) < 0.8
train_data = data[train_indices].data.clone()
test_data = data[~train_indices].data.clone()

first_letters = vocab_df.query("num_chars==5 & is_word").let0.values
train_first = first_letters[train_indices]
test_first = first_letters[~train_indices]

frequent_let0 = (np.arange(26)[vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index()>50])
frequent_let0
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
    "dict_size": 26,
    "enc_dtype":"fp32",
    "device": "cuda:0",
}
temp_encoder = AutoEncoder(cfg)
loss, reconstr, acts, l2, l1 = temp_encoder(test_data)
print(loss, l2, l1)
# loss.backward()
# %%
torch.set_grad_enabled(True)
encoder = AutoEncoder(cfg)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
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
    if (i+1)%10000 == 0:
        line([l0_losses, l1_losses, l2_losses, test_l0_losses, test_l1_losses, test_l2_losses], line_labels=["L0", "L1", "L2", "test L0", "test L1", "test L2"])
# line([l0_losses, l1_losses, l2_losses, test_l0_losses, test_l1_losses, test_l2_losses], line_labels=["L0", "L1", "L2", "test L0", "test L1", "test L2"])
# %%
# train_first_letter = 
# %%
train_loss, train_reconstr, train_acts, train_l2, train_l1 = encoder(train_data)
test_loss, test_reconstr, test_acts, test_l2, test_l1 = encoder(test_data)
histogram((train_acts!=0).float().mean(0))
histogram((test_acts!=0).float().mean(0))

train_one_hot_first = torch.tensor((train_first[:, None] == np.arange(26)[None, :])).cuda()
train_ave_act_by_letter = ((train_one_hot_first[:, None, :] * train_acts[:, :, None]).sum(0) / train_one_hot_first.sum(0)).T
line(train_ave_act_by_letter[frequent_let0], line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0], title="Train Ave Act by Letter")

train_act_firings = train_acts>0
train_ave_act_firing_by_letter = ((train_one_hot_first[:, None, :] * train_act_firings[:, :, None]).sum(0) / train_one_hot_first.sum(0)).T
line(train_ave_act_firing_by_letter[frequent_let0], line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0], title="Train Frac is Firing by Letter")

test_one_hot_first = torch.tensor((test_first[:, None] == np.arange(26)[None, :])).cuda()
test_ave_act_by_letter = ((test_one_hot_first[:, None, :] * test_acts[:, :, None]).sum(0) / (test_one_hot_first.sum(0)+1e-5)).T
line(test_ave_act_by_letter[frequent_let0], line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0], title="test Ave Act by Letter")

test_act_firings = test_acts>0
test_ave_act_firing_by_letter = ((test_one_hot_first[:, None, :] * test_act_firings[:, :, None]).sum(0) / (test_one_hot_first.sum(0)+1e-5)).T
line(test_ave_act_firing_by_letter[frequent_let0], line_labels=[i for c, i in enumerate(alphalist) if c in frequent_let0], title="test Frac is Firing by Letter")
# %%
test_one_hot_first = torch.tensor((test_first[:, None] == np.arange(26)[None, :])).cuda()
test_ave_act_by_letter = ((test_one_hot_first[:, None, :] * test_acts[:, :, None]).sum(0) / (test_one_hot_first.sum(0)+1e-5)).T
line(test_ave_act_by_letter, line_labels=alphalist, title="test Ave Act by Letter")
# %%
imshow(train_ave_act_by_letter, y=alphalist)
# %%
px.scatter(x=train_ave_act_by_letter.max(-1).values.cpu().detach().numpy(), y=test_ave_act_by_letter.max(-1).values.cpu().detach().numpy()).show()
px.scatter(x=vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index().values, y=test_ave_act_by_letter.max(-1).values.cpu().detach().numpy(), hover_name=alphalist).show()
# %%
px.scatter(x=vocab_df.query("num_chars==5 & is_word").let0.value_counts().sort_index().values, y=test_ave_act_firing_by_letter.max(-1).values.cpu().detach().numpy(), hover_name=alphalist).show()

# %%
x = test_ave_act_by_letter.argmax(dim=0).sort().indices
imshow(test_ave_act_by_letter[:, x])
# %%
