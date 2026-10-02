"""Hugging Face transformers のオープンウェイトモデル用バックエンド。

層番号 ℓ は 1..n_layers で、「ℓ 番目のデコーダブロックの出力」を指す
（hidden_states[ℓ] に一致）。ステアリングはそのブロックの出力に α·v を全位置で加える。
"""
from __future__ import annotations

import contextlib

import numpy as np

from .base import Backend, Chat, Steer


def _import_torch():
    import torch  # noqa: WPS433
    return torch


def find_layers(model):
    """デコーダブロックの ModuleList を探す（Llama/Qwen/Gemma/Mistral/GPT-2/GPT-NeoX）。"""
    torch = _import_torch()
    for path in ("model.layers", "model.language_model.layers", "transformer.h",
                 "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue
    best = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m
    if best is None:
        raise ValueError("decoder layers not found")
    return best


class HFBackend(Backend):
    has_activations = True

    def __init__(self, path: str | None = None, *, model=None, tokenizer=None, name: str | None = None,
                 dtype: str = "auto", device: str | None = None, batch_size: int = 8,
                 max_length: int = 4096, trust_remote_code: bool = False):
        torch = _import_torch()
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            dt = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[dtype]
            kw = {"dtype": dt, "trust_remote_code": trust_remote_code}
            if device is None and torch.cuda.is_available():
                kw["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(path, **kw)
            if device is not None:
                model = model.to(device)
        self.model = model.eval()
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.name = name or path or "hf"
        self.layers = find_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = int(self.model.config.get_text_config().hidden_size
                           if hasattr(self.model.config, "get_text_config")
                           else self.model.config.hidden_size)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = next(self.model.parameters()).device
        self._label_cache: dict[tuple, list[int] | None] = {}

    # ------------------------------------------------------------ 書式
    def _has_template(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None))

    def render(self, chat: Chat) -> str:
        if not self._has_template():
            return f"{chat.system}\n\n{chat.user}\n\n{chat.prefix}"
        msgs = [{"role": "system", "content": chat.system}, {"role": "user", "content": chat.user}]
        try:
            s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # system ロールを受け付けないテンプレート（Gemma など）
            msgs = [{"role": "user", "content": chat.system + "\n\n" + chat.user}]
            s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return s + chat.prefix

    def _encode(self, texts: list[str]):
        torch = _import_torch()
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_length, add_special_tokens=not self._has_template())
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    # ------------------------------------------------------------ ステアリング
    @contextlib.contextmanager
    def _steering(self, steer: Steer | None):
        if steer is None or steer.alpha == 0:
            yield
            return
        torch = _import_torch()
        block = self.layers[steer.layer - 1]
        p = next(block.parameters())
        vec = torch.as_tensor(np.asarray(steer.vector) * steer.alpha, dtype=p.dtype, device=p.device)

        def hook(_m, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec

        h = block.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    # ------------------------------------------------------------ API
    def hidden(self, chats, layers):
        torch = _import_torch()
        out = {l: [] for l in layers}
        texts = [self.render(c) for c in chats]
        for i in range(0, len(texts), self.batch_size):
            ids, mask, pos = self._encode(texts[i:i + self.batch_size])
            with torch.no_grad():
                r = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               output_hidden_states=True)
            for l in layers:
                out[l].append(r.hidden_states[l][:, -1].float().cpu().numpy())
        return {l: np.concatenate(v, 0) for l, v in out.items()}

    def _label_ids(self, prompt: str, label: str) -> list[int]:
        """prompt の直後に label が続くときのトークン列（境界での結合に対応）。"""
        key = (prompt[-32:], label)
        if key not in self._label_cache:
            add = not self._has_template()
            a = self.tok(prompt, add_special_tokens=add)["input_ids"]
            b = self.tok(prompt + label, add_special_tokens=add)["input_ids"]
            if b[:len(a)] == a and len(b) > len(a):
                self._label_cache[key] = b[len(a):]
            else:
                self._label_cache[key] = self.tok(label, add_special_tokens=False)["input_ids"]
        return self._label_cache[key]

    def choice_logprobs(self, chats, labels, steer=None):
        torch = _import_torch()
        texts = [self.render(c) for c in chats]
        lab_ids = [[self._label_ids(t, lab) for lab in labels] for t in texts]
        single = all(len(x) == 1 for row in lab_ids for x in row)
        res = np.zeros((len(chats), len(labels)))
        with self._steering(steer):
            if single:
                for i in range(0, len(texts), self.batch_size):
                    ids, mask, pos = self._encode(texts[i:i + self.batch_size])
                    with torch.no_grad():
                        logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
                    lp = torch.log_softmax(logits[:, -1].float(), -1).cpu().numpy()
                    for j in range(lp.shape[0]):
                        res[i + j] = [lp[j, lab_ids[i + j][k][0]] for k in range(len(labels))]
            else:
                seqs, owners = [], []
                for n, t in enumerate(texts):
                    for k, lab in enumerate(labels):
                        seqs.append((t, lab_ids[n][k]))
                        owners.append((n, k))
                for i in range(0, len(seqs), self.batch_size):
                    chunk = seqs[i:i + self.batch_size]
                    add = not self._has_template()
                    rows = [self.tok(t, add_special_tokens=add)["input_ids"] + li for t, li in chunk]
                    L = max(len(r) for r in rows)
                    pad = self.tok.pad_token_id
                    ids = torch.tensor([[pad] * (L - len(r)) + r for r in rows], device=self.device)
                    mask = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows], device=self.device)
                    pos = (mask.cumsum(-1) - 1).clamp(min=0)
                    with torch.no_grad():
                        logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
                    lp = torch.log_softmax(logits.float(), -1)
                    for j, (_, li) in enumerate(chunk):
                        n_lab = len(li)
                        tgt = torch.tensor(li, device=self.device)
                        sl = lp[j, L - n_lab - 1:L - 1]
                        res[owners[i + j]] = float(sl.gather(-1, tgt[:, None]).sum())
        return res

    def text_nll(self, texts, steer=None):
        torch = _import_torch()
        tot, n = 0.0, 0
        with self._steering(steer):
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_length)
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
                with torch.no_grad():
                    logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
                lp = torch.log_softmax(logits[:, :-1].float(), -1)
                tgt = ids[:, 1:]
                m = mask[:, 1:].bool() & mask[:, :-1].bool()
                ll = lp.gather(-1, tgt[..., None])[..., 0]
                tot += float(-(ll * m).sum())
                n += int(m.sum())
        return tot / max(n, 1)
