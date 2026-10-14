"""用本机GGUF词表重建Qwen35分词，避免把UTF-8字节上界当作实测token。

tokenizers是可选实验依赖。没有匹配词表或校准时仍用原字节上界，不冒险放宽。
分词正则来自llama.cpp的LLAMA_VOCAB_PRE_TYPE_QWEN35；依赖及源码另行留档。
"""

import hashlib
import json

from scripts.local_fund_analysis_probe import MAX_CONTEXT, MODEL, PREDICT_TOKENS

PRETOKEN_SOURCE = "https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp"
PRETOKEN_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)


class LocalTokenBudget:
    """固定词表与64K上限；正文分词计数之外另预留模板、schema和完整输出空间。"""

    def __init__(self, info: dict):
        from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

        if info.get("tokenizer.ggml.model") != "gpt2" or info.get("tokenizer.ggml.pre") != "qwen35":
            raise ValueError("LOCAL_TOKENIZER_NOT_QWEN35")
        tokens = info["tokenizer.ggml.tokens"]
        vocabulary = {token: index for index, token in enumerate(tokens)}
        if len(vocabulary) != len(tokens):
            raise ValueError("LOCAL_TOKENIZER_DUPLICATE_VOCAB")
        pairs = [tuple(pair.split(" ")) for pair in info["tokenizer.ggml.merges"]]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("LOCAL_TOKENIZER_MERGE_INVALID")
        self.tokenizer = Tokenizer(models.BPE(vocabulary, pairs))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(PRETOKEN_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        self.tokenizer.decoder = decoders.ByteLevel()
        # CONTROL与USER_DEFINED按GGUF标记匹配；不从云端下载另一套模型词表。
        self.tokenizer.add_special_tokens(
            [
                AddedToken(t, special=True, normalized=False)
                for t, kind in zip(tokens, info["tokenizer.ggml.token_type"], strict=True)
                if kind in (3, 4)
            ]
        )
        self.vocab_hash = hashlib.sha256(json.dumps(info, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.calibrated = False

    def count(self, text: str) -> int:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if self.tokenizer.decode(encoded.ids, skip_special_tokens=False) != text:
            raise ValueError("LOCAL_TOKENIZER_ROUNDTRIP_FAILED")
        return len(encoded.ids)

    def content_tokens(self, request: dict) -> int:
        return sum(self.count(message["content"]) for message in request["messages"])

    def calibrate(self, pairs: list[tuple[dict, dict]]) -> list[dict]:
        """用两种固定文本核对运行器开销；不能用校准结果宣称所有输入计数精确一致。"""
        receipts = []
        for request, response in pairs:
            count = self.content_tokens(request)
            observed = response.get("prompt_eval_count")
            if not response.get("done") or not isinstance(observed, int) or not count <= observed <= count + 1024:
                raise ValueError("LOCAL_TOKENIZER_RUNTIME_CALIBRATION_FAILED")
            receipts.append(
                {"content_tokens": count, "runtime_prompt_tokens": observed, "template_delta": observed - count}
            )
        if len(receipts) < 2:
            raise ValueError("LOCAL_TOKENIZER_CALIBRATION_INSUFFICIENT")
        self.calibrated = True
        return receipts

    def request(self, prompt: str, value: dict, schema: dict) -> dict:
        """完整输入不裁剪；模板预留1024、schema按字节上界，输出固定4096。"""
        if not self.calibrated:
            raise ValueError("LOCAL_TOKENIZER_NOT_CALIBRATED")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))},
        ]
        content_count = sum(self.count(m["content"]) for m in messages)
        reserve = len(json.dumps(schema, ensure_ascii=False).encode()) + 1024 + PREDICT_TOKENS
        upper = content_count + reserve
        context = next((size for size in (16384, 32768, MAX_CONTEXT) if upper < size), None)
        if context is None:
            raise ValueError(f"LOCAL_FULL_INPUT_TOO_LARGE_TOKEN_BUDGET:{upper}")
        return {
            "model": MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "format": schema,
            "options": {"temperature": 0, "seed": 42, "num_ctx": context, "num_predict": PREDICT_TOKENS},
            "messages": messages,
        }

    def receipt(self, request: dict, response: dict | None = None) -> dict:
        """分别记录本地重建计数、保守上界和Ollama实测值，防止混用。"""
        count = self.content_tokens(request)
        return {
            "method": "GGUF_BPE_CONTENT_PLUS_SCHEMA_BYTES_AND_TEMPLATE_RESERVE",
            "vocab_sha256": self.vocab_hash,
            "content_tokens": count,
            "upper_with_reserve": count
            + len(json.dumps(request.get("format", {}), ensure_ascii=False).encode())
            + 1024
            + request["options"]["num_predict"],
            "context": request["options"]["num_ctx"],
            "runtime_prompt_tokens": (response or {}).get("prompt_eval_count"),
            "runtime_output_tokens": (response or {}).get("eval_count"),
        }
