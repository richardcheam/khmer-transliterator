"""Tests of E07 (`scripts/e07.py`): the confirmed latent defect (generation room for EOS), label alignment and
masking, adapter insertion and updates, checkpoint reload, split protection, resource caps and safe resume.

Model tests use a tiny random T5 (no download, seconds on CPU), never the pinned ByT5 checkpoint."""
from __future__ import annotations

import doctest
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e07  # noqa: E402
import e06_byt5 as B  # noqa: E402

try:
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
except ImportError:                                  # research extras not installed (research/requirements-byt5.txt)
    torch = None

ROWS = [("cat", "ខាត"), ("london", "ឡុងដ៍"), ("coffee", "កាហ្វេ"), ("ali", "អាលី")]
ADAPTER = {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["attn", "ffn"]}


def tiny_t5(adapter=ADAPTER, dropout=0.0, seed=0):
    torch.manual_seed(1)
    cfg = T5Config(vocab_size=384, d_model=32, d_kv=8, d_ff=48, num_layers=2, num_decoder_layers=1, num_heads=2,
                   feed_forward_proj="gated-gelu", dropout_rate=dropout, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1, tie_word_embeddings=False)
    return e07.insert_adapters(T5ForConditionalGeneration(cfg), adapter, seed)


class TestPure(unittest.TestCase):
    def test_doctests(self):
        self.assertEqual(doctest.testmod(e07, verbose=False).failed, 0)

    def test_generation_room_for_eos(self):
        """Confirmed latent E06 defect: 120 new tokens left no room for EOS after a 120-byte target."""
        self.assertGreaterEqual(e07.MAX_NEW, B.RECIPE["max_tgt_bytes"] + 1)
        self.assertLess(B.RECIPE["decode"]["max_new_tokens"], B.RECIPE["max_tgt_bytes"] + 1)   # E06 kept as it was
        full = B.encode_tgt("ក" * 40)                                              # exactly 120 bytes + EOS
        self.assertEqual(len(full), 121)
        self.assertEqual(e07.decode(full[:-1]), (None, "no_eos"))                 # truncation is flagged
        self.assertEqual(e07.decode(full)[1], "ok")

    def test_tiny_selection_rules(self):
        rows = [(f"w{'x' * n}", f"ក{n}") for n in range(12)] + [("dup", "ក99"), ("dup", "ក98"), ("same", "ក0")]
        cfg = {"bins": [[1, 5], [6, 99]], "per_bin": 2, "n": 4}
        out = e07.select_tiny(rows, cfg)
        self.assertEqual(len({w for w, _ in out}), 4)
        self.assertEqual(len({k for _, k in out}), 4)
        self.assertNotIn("dup", {w for w, _ in out})                               # two targets → excluded
        self.assertNotIn("ក0", {k for _, k in out})                                # shared target → excluded
        self.assertEqual(out, e07.select_tiny(list(reversed(rows)), cfg))         # order independent

    def test_split_protection(self):
        held = [mock.Mock(word="cat")]
        with mock.patch.object(e07.e06, "items", return_value=held):
            with self.assertRaises(ValueError):
                e07.check_disjoint(ROWS)
            with mock.patch.object(e07.e06, "items", return_value=[mock.Mock(word="zebra")]):
                self.assertEqual(e07.check_disjoint(ROWS)["overlap_with_dev_eval"], 0)

    def test_resource_caps(self):
        seq = iter([{"rss": 6 * e07.GiB, "available": 4 * e07.GiB}, {"rss": 1, "available": 4 * e07.GiB}] +
                   [{"rss": 1, "available": 1 * e07.GiB}] * 3)
        g = e07.Guard(read=lambda: next(seq))
        self.assertEqual([g.check() for _ in range(5)], [None, None, None, None, "memory_pressure"])  # sustained only
        led = [{"event": "start", "run": f"tiny_a{a}", "kind": "tiny", "t": 0.0} for a in (1, 2)] + \
              [{"event": "segment", "run": "tiny_a1", "kind": "tiny", "hours": 7.5, "t": 10.0}]
        b = e07.budget_left("tiny", led, now=13 * 3600)
        self.assertEqual(b["attempts_left"], 0)
        self.assertAlmostEqual(b["training_hours_left"], 0.5)
        self.assertLess(b["wall_hours_left"], 0)
        self.assertEqual(e07.budget_left("full", led, now=1.0)["attempts_left"], 1)

    def test_attempt2_changes_one_setting(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(e07, "REPORTS", Path(d)):
                (Path(d) / "tiny_attempt2_change.json").write_text(json.dumps({"change": {"lr": 3e-3, "batch": 8}}))
                with self.assertRaises(ValueError):
                    e07.tiny_cfg(2)
                (Path(d) / "tiny_attempt2_change.json").write_text(json.dumps({"change": {"adapter.r": 64}}))
                c = e07.tiny_cfg(2)
                self.assertEqual(c["adapter"]["r"], 64)
                self.assertEqual({k: v for k, v in c.items() if k != "adapter"},
                                 {k: v for k, v in e07.TINY.items() if k != "adapter"})
                self.assertEqual(e07.TINY["adapter"]["r"], 16)                     # the frozen config is untouched


@unittest.skipIf(torch is None, "research extras not installed")
class TestModel(unittest.TestCase):
    def test_labels_masking_and_shift(self):
        ids, mask, labels = B.batch_tensors(ROWS)
        m = tiny_t5()
        for n, (w, k) in enumerate(ROWS):
            self.assertEqual(B.decode_ids([x for x in labels[n].tolist() if x != -100])[0], k)        # aligned
            self.assertEqual(int(mask[n].sum()), len(w) + 1)                                         # + EOS
            last = len(k.encode())
            self.assertEqual(int(labels[n, last]), B.EOS)
        self.assertEqual(int(((labels == B.PAD)).sum()), 0)                       # padding is −100, never pad id
        out = m(input_ids=ids, attention_mask=mask, labels=labels)
        nll = torch.nn.functional.cross_entropy(out.logits.transpose(1, 2), labels, ignore_index=-100,
                                                reduction="sum")
        self.assertAlmostEqual(float(out.loss), float(nll / (labels != -100).sum()), places=5)
        dec = m._shift_right(labels)
        self.assertTrue(torch.equal(dec[:, 0], torch.zeros(len(ROWS), dtype=torch.long)))
        self.assertTrue(torch.equal(dec[:, 1:], labels[:, :-1].clamp(min=0)))

    def test_adapters_inserted_and_updated(self):
        m = tiny_t5()
        groups = e07.adapter_groups(m)
        self.assertEqual(set(groups), {"encoder_self", "encoder_ffn", "decoder_self", "decoder_cross", "decoder_ffn"})
        names = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.endswith((".A", ".B")) for n in names))
        frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
        before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        ids, mask, labels = B.batch_tensors(ROWS)
        m.train()
        for _ in range(2):
            m(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
                                if p.requires_grad))
            opt.step()
            opt.zero_grad()
        now = dict(m.named_parameters())
        self.assertTrue(all(not torch.equal(before[n], now[n]) for n in names))
        self.assertTrue(all(torch.equal(frozen[n], now[n]) for n in frozen))
        attn_only = tiny_t5({**ADAPTER, "targets": ["attn"]})
        self.assertEqual(set(e07.adapter_groups(attn_only)), {"encoder_self", "decoder_self", "decoder_cross"})

    def test_zero_init_equals_base_and_reload(self):
        base = tiny_t5(None).eval()
        m = tiny_t5().eval()
        ids, mask, labels = B.batch_tensors(ROWS)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(input_ids=ids, attention_mask=mask, labels=labels).logits,
                                           m(input_ids=ids, attention_mask=mask, labels=labels).logits, atol=1e-6))
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith(".B"):
                    p.normal_()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.pt"
            torch.save(B.lora_state(m), p)
            r = tiny_t5(seed=5)
            e07.load_adapter(r, p)
            src = [B.encode_src(w) for w, _ in ROWS]
            self.assertEqual([x[0][3] for x in e07.gen(m, src, max_new=8)], [x[0][3] for x in e07.gen(r, src, max_new=8)])
            with self.assertRaises(RuntimeError):
                e07.load_adapter(tiny_t5({**ADAPTER, "targets": ["attn"]}), p)    # unknown keys are refused
            sd = torch.load(p)
            sd.pop(next(iter(sd)))
            torch.save(sd, p)
            with self.assertRaises(RuntimeError):
                e07.load_adapter(tiny_t5(), p)                                     # missing keys are refused

    def test_greedy_scores_are_sum_logprob(self):
        m = tiny_t5().eval()
        src = [B.encode_src(w) for w, _ in ROWS]
        res = e07.gen(m, src, max_new=6)
        for s, r in zip(src, res):
            raw = r[0][3]
            tl = e07.token_losses(m, [s], [raw])[0]
            self.assertAlmostEqual(-sum(v for _, v in tl), r[0][2], places=3)

    def test_safe_resume(self):
        cfg = {**e07.TINY, "adapter": ADAPTER, "max_updates": 6, "batch": 2, "warmup": 1, "eval_every": 3}
        evals = []

        def evaluate(model):
            evals.append(1)
            return {"key": (len(evals),), "n": len(evals)}

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
            d = Path(d)
            with mock.patch.multiple(e07, LEDGER=d / "ledger.jsonl", STATE=d / "state"):
                make = lambda *a: tiny_t5()  # noqa: E731
                calls = iter([None] * 4 + ["memory_pressure"] * 10)
                g = e07.Guard(read=lambda: {"rss": 1, "available": 1 << 40})
                g.check = lambda: next(calls)
                r1 = e07.fit("t", "tiny", cfg, ROWS, evaluate, {3, 6}, d / "run", guard=g, make=make)
                self.assertTrue(r1.get("resumable"))
                self.assertEqual(r1["updates"], 4)
                ck = torch.load(d / "run" / "last.pt", weights_only=False)
                self.assertEqual(ck["state"]["update"], 4)
                with self.assertRaises(RuntimeError):                               # a changed config cannot resume
                    e07.fit("t", "tiny", {**cfg, "lr": 5e-3}, ROWS, evaluate, {3, 6}, d / "run", make=make)
                r2 = e07.fit("t", "tiny", cfg, ROWS, evaluate, {3, 6}, d / "run", make=make)
                self.assertEqual(r2["updates"], 6)
                self.assertEqual([h["update"] for h in r2["history"]], [3, 6])     # no evaluation under memory pressure
                led = e07.ledger()
                self.assertEqual(sum(e["event"] == "start" for e in led), 1)       # one attempt, two segments
                self.assertEqual(sum(e["event"] == "segment" for e in led), 2)
                self.assertEqual(e07.fit("t", "tiny", cfg, ROWS, evaluate, {3, 6}, d / "run", make=make), r2)
                for k in ("t2", "t3"):
                    if k == "t3":
                        with self.assertRaises(RuntimeError):                       # attempt cap enforced
                            e07.fit(k, "tiny", cfg, ROWS, evaluate, {3, 6}, d / k, make=make)
                    else:
                        e07.fit(k, "tiny", cfg, ROWS, evaluate, {3, 6}, d / k, make=make)


if __name__ == "__main__":
    unittest.main()
