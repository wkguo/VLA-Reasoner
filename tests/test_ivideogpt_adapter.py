"""Bookkeeping of the iVideoGPT adapter (token/embedding sequences per tree node) with small fakes.

The fakes mimic upstream's sequence-length contract (context: 257 tokens per frame incl. separators;
dynamics: slot + 16 tokens per frame; `detokenize` asserts the same length invariant as upstream). They do NOT
exercise the real iVideoGPT networks or HF `generate`, which need the released weights.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vla_reasoner.mcts import MCTS  # noqa: E402
from vla_reasoner.world_model.ivideogpt import IVideoGPTWorldModel  # noqa: E402


class FakeTokenizer:
    context_length = 2

    def tokenize(self, x, ctx):
        assert ctx == self.context_length
        b, t = x.shape[:2]
        return torch.zeros(b, ctx * 257 - 1 + 17 * (t - ctx), dtype=torch.long), None

    def detokenize(self, idx, ctx):
        b, length = idx.shape
        assert (length + 1 - 257 * ctx) % 17 == 0  # upstream invariant
        future = (length + 1 - 257 * ctx) // 17
        assert future == 1  # one predicted frame per call
        v = idx[:, -16:].float().mean(1) / 10.0  # frame brightness encodes the predicted tokens
        return v.view(b, 1, 1, 1, 1).expand(b, ctx + future, 3, 8, 8)


class FakeLLM:
    def generate(self, inputs_embeds, max_new_tokens, output_hidden_states=False, **kwargs):
        b, _, e = inputs_embeds.shape
        tok = inputs_embeds[:, -1, 0].round().clamp(0, 9).long()  # token = sum of the injected action
        hidden = tuple((torch.zeros(b, 1, e), torch.ones(b, 1, e)) for _ in range(max_new_tokens))
        return SimpleNamespace(sequences=tok[:, None].expand(b, max_new_tokens), hidden_states=hidden)


class FakeHead(torch.nn.Module):
    def __init__(self, action_dim, e=4):
        super().__init__()
        self.llm = FakeLLM()
        self.emb = torch.nn.Embedding(32, e)
        torch.nn.init.zeros_(self.emb.weight)
        self.action_linear = torch.nn.Linear(action_dim, e)
        with torch.no_grad():
            self.action_linear.weight.zero_()
            self.action_linear.weight[0] = 1.0  # first embedding coordinate = sum(action)
            self.action_linear.bias.zero_()
        self.reward_linear = torch.nn.Linear(e, 1)
        with torch.no_grad():
            self.reward_linear.weight.fill_(0.5)
            self.reward_linear.bias.zero_()
        self.token_for_sdf = 31

    def get_input_embeddings(self, ids):
        return self.emb(ids)


def make_wm(reward_head=False):
    wm = IVideoGPTWorldModel.__new__(IVideoGPTWorldModel)  # skip loading real weights
    wm._torch, wm.context_length, wm.segment_length, wm.resolution = torch, 2, 12, 8
    wm.device, wm.temperature, wm.top_k = torch.device("cpu"), 1.0, 100
    wm.reward_head, wm.max_batch, wm._warned_depth = reward_head, None, False
    wm.tokenizer, wm.model = FakeTokenizer(), FakeHead(action_dim=2)
    return wm


def test_branching_sequences_and_frames():
    wm = make_wm()
    root = wm.init_state([np.zeros((16, 16, 3), np.uint8)] * 2)
    assert root.embeds.shape[1] == 2 * 257 and root.frame.shape == (8, 8, 3)
    children = wm.predict(root, np.array([[1.0, 0.0], [3.0, 0.0]]))
    assert [c.embeds.shape[1] for c in children] == [2 * 257 + 17] * 2
    assert [int(c.frame[0, 0, 0]) for c in children] == [round(255 * 0.1), round(255 * 0.3)]
    assert float(root.embeds[0, -1].abs().sum()) == 0.0  # the parent's sequence is not modified
    grand = wm.predict(children[0], np.array([[5.0, 0.0]]))[0]
    assert grand.embeds.shape[1] == 2 * 257 + 34 and grand.depth == 2
    assert int(grand.frame[0, 0, 0]) == round(255 * 0.5)
    assert torch.equal(grand.context_tokens, root.context_tokens)


def test_reward_head_and_max_batch():
    wm = make_wm(reward_head=True)
    wm.max_batch = 1
    root = wm.init_state([np.zeros((8, 8, 3), np.uint8)] * 2)
    kids = wm.predict(root, np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert len(kids) == 3 and all(k.reward == pytest.approx(2.0) for k in kids)  # 0.5 * sum(ones(4))


def test_plugs_into_mcts():
    wm = make_wm()

    class Sampler:
        k = 2

        def __call__(self, reference, rng):
            return np.stack([reference + 1.0, reference + 2.0]), np.zeros(2)

    def value(states):
        return np.array([s.frame.mean() / 255.0 for s in states])

    root = wm.init_state([np.zeros((8, 8, 3), np.uint8)] * 2)
    result = MCTS(wm, value, Sampler(), max_depth=2).search(root, np.array([0.0, 0.0]), np.random.default_rng(0))
    assert result.num_simulations == 6 and result.action.shape == (2,)
