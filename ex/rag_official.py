
"""Run only the proposed method (ours_rrf_bm25_kure: weighted RRF of BM25 + original-query
KURE + compressed-query KURE, weights 0.495/0.01/0.495) over every query that has gold
documents, and measure recall@20.

The number of needles is fixed to 1. For every query with at least one gold document, each
way of picking one of its gold documents (C(k,1) = k combinations) becomes a separate
needle=1 instance. For example, a query with 3 gold documents yields 3 needle=1 instances
(each one uses a single gold document as the answer and leaves the other gold documents out
of the corpus).

Query compression results (GPT calls) are cached per qid. The cache file is keyed by
direction + query_set and shared across tags; qids already in the cache are not sent to GPT
again. Each cache entry stores the compressed text together with API response metadata
(request id, creation time, model version, system_fingerprint) and a SHA-256 hash of the
entry. Every time the cache is loaded the hashes are recomputed and compared with the stored
values; if any of them differ (i.e. the cache appears to have changed since it was written),
the run is aborted.

BM25 (CPU) and KURE (one GPU given by --device, plus optional GPUs from --extra-devices) run
concurrently in separate processes. The child processes use CUDA, so the multiprocessing
start method is forced to spawn.

When --extra-devices lists more GPUs, corpus encoding (distractor + needle documents) is split
into contiguous ranges across all those GPUs and encoded concurrently in threads (one model
copy per GPU; this helps most in the direction with more documents). The built-in
sentence-transformers multi-process pool is not used: calling it inside a process that has
already initialised CUDA by loading the model on --device (and that is itself a child process
we spawned) caused the child process to crash. Queries are far fewer than documents, so query
encoding runs on --device only.

--fp16 loads KURE in half precision (bf16, or fp16 if bf16 is unavailable) to save memory and
allow larger batches. The default is fp32.

For BM25, each needle=1 instance (entry) has a different corpus (distractors + that entry's
needle document), so a new BM25Okapi index must be built per entry. With many entries this is
CPU-heavy, so entries are split across --bm25-workers CPU processes (default: all cores). The
scoring itself is unchanged; only the parallelism is added.

Run from the project root (normally ex/run_official_once.sh runs both directions at once).

    python ex/rag_official.py --direction fin_gov \
        --qrels work/combined/finance/full/qrels.json \
        --device cuda:0

    python ex/rag_official.py --direction gov_fin \
        --qrels work/combined/government/full/qrels.json \
        --device cuda:1 --extra-devices cuda:2 --fp16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_corpus
import config
import evaluate as evaluate_mod
from config import DIRECTIONS
from ood_retrieval.data import load_queries

TOP_K = 20
RRF_CONSTANT = 60
RRF_CANDIDATE_K = 2 * TOP_K
KURE_MODEL_NAME = "nlpai-lab/KURE-v1"
GPT_MODEL = "gpt-5"
N_NEEDLES = 1  # Fixed number of needles. No longer a CLI option.

# RRF weights of the proposed method: BM25 49.5% + original-query KURE 1% + compressed-query KURE 49.5%
OURS_WEIGHTS = {"bm25": 0.495, "orig_kure": 0.01, "comp_kure": 0.495}
OURS_METHOD_NAME = "ours_rrf_bm25_kure"


# The prompt is intentionally kept in Korean: the call transcripts are Korean, and this exact
# prompt is part of the experimental setup. Changing it changes the compressed queries.
# English summary: "Compress the full transcript of a suspected voice-phishing call into a
# search query. List the key situations and expressions as short phrases in the order of the
# call, without categories or headings. Keep verbatim (do not summarise): institution / bank /
# product names; proper nouns such as carriers and card companies; concrete amounts, limits and
# interest rates; specific threatening or persuasive phrasing. Replace only randomly generated
# identifiers (case numbers, phone numbers, IP addresses) with their type. Remove only
# conversational noise (greetings, repetition, hesitation). Output only the compressed query."
COMPRESSION_PROMPT_TEMPLATE = """Instruction: 다음은 보이스피싱 의심 전화 통화의 전체 녹취입니다.
검색에 사용할 핵심 쿼리로 압축하려고 합니다.
통화에서 있었던 핵심적인 상황과 표현들을 짧은 구절 단위로 나열하세요.
카테고리별로 분류하거나 소제목을 달지 말고, 통화의 흐름을 따라 자연스럽게 이어서 쓰세요.

다음은 원문의 표현 그대로, 요약하지 말고 포함하세요:
- 기관명·은행명·상품명
- 통신사, 카드사 등 고유명사
- 구체적 금액, 한도, 금리(%) 수치
- 협박/유도에 사용된 구체적 화법 (예: "다른 상담원은 확인 불가하다고 주장")

사건번호, 전화번호, IP주소처럼 매번 무작위로 생성되는 식별 숫자만 유형으로 표현하고,
나머지는 문맥을 살려서 그대로 남기세요.

대화의 잡음(인사말, 반복, 망설임)만 제거하세요.

통화 녹취: {full_call_transcript}

핵심 요약 쿼리:
(다른 설명 없이 압축된 쿼리 텍스트만 출력하세요.)"""


# Hangul syllables, Latin letters, digits and underscore.
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")


def tokenize_regex(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return _TOKEN_RE.findall(text)


_kiwi_instance = None


def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        from kiwipiepy import Kiwi
        _kiwi_instance = Kiwi()
    return _kiwi_instance


_KIWI_KEEP_TAGS = {"NNG", "NNP", "NNB", "VV", "VA", "SN", "SL"}


def tokenize_kiwi(text: str) -> list[str]:
    kiwi = _get_kiwi()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [t.form.lower() for t in kiwi.tokenize(text) if t.tag in _KIWI_KEEP_TAGS]


def get_tokenizer(name: str):
    """Tokenizer name -> function. The name is passed explicitly instead of using a global so
    that spawned child processes always use the right tokenizer."""
    if name == "kiwi":
        return tokenize_kiwi
    return tokenize_regex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=list(DIRECTIONS), required=True)
    p.add_argument("--qrels", required=True,
                   help="qrels.json mapping each query ID to its gold document IDs.")
    p.add_argument("--query-set", choices=["full"], default="full")
    p.add_argument("--device", default="cuda:0", help="Primary GPU for KURE embeddings (BM25 runs on CPU).")
    p.add_argument("--extra-devices", default="",
                   help="Additional GPUs used only for corpus (document) encoding, comma-separated "
                        "(e.g. cuda:1,cuda:2). Together with --device, document encoding is "
                        "split across these GPUs in threads. Query encoding runs on --device "
                        "only, since there are far fewer queries than documents.")
    p.add_argument("--fp16", action="store_true",
                   help="Load KURE in half precision (bf16 if supported, otherwise fp16) to "
                        "save memory and allow larger batches. Default is fp32.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="KURE encoding batch size. Larger is faster as long as it fits in memory.")
    p.add_argument("--tokenizer", choices=["regex", "kiwi"], default="regex",
                   help="BM25 tokenizer: regex (pattern-based) or kiwi (Korean morphological "
                        "analysis; drops particles/endings to normalise inflected forms).")
    p.add_argument("--bm25-workers", type=int, default=0,
                   help="Number of CPU processes for BM25 (a new BM25Okapi index is built per "
                        "entry because each entry has a different corpus, which is expensive). "
                        "0 (default) uses all CPU cores.")
    p.add_argument("--tag", default="",
                   help="Tag appended to output file names (as _{tag}) so repeated runs with the "
                        "same settings do not overwrite each other (e.g. run1, run2). The query "
                        "compression cache ignores the tag and is shared per "
                        "direction + query_set.")
    args = p.parse_args()
    return args


# ------------------------------------------------------------------
# GPT compression (+ per-qid caching with a SHA-256 integrity hash)
# ------------------------------------------------------------------
def compress_query(client, full_call_transcript: str, retries: int = 3) -> dict:
    """Call GPT to compress one query; return the compressed text plus API response metadata."""
    prompt = COMPRESSION_PROMPT_TEMPLATE.format(full_call_transcript=full_call_transcript)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL, messages=[{"role": "user", "content": prompt}])
            out = resp.choices[0].message.content.strip()
            # Strip the "핵심 요약 쿼리:" ("compressed query:") label if the model echoes it.
            if out.lower().startswith("핵심 요약 쿼리"):
                out = out.split(":", 1)[-1].strip()
            return {
                "text": out,
                "request_id": getattr(resp, "id", None),
                "created": getattr(resp, "created", None),
                "model": getattr(resp, "model", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
            }
        except Exception as e:
            print(f"  GPT call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("GPT call failed repeatedly.")


def _compute_entry_hash(qid: str, record: dict) -> str:
    """SHA-256 integrity hash of a cache entry: the qid and every field of the record (except
    sha256 itself), serialised as key-sorted JSON. Recomputing it on load and comparing with the
    stored value shows whether the entry changed after it was written."""
    payload = {"qid": qid, **{k: v for k, v in record.items() if k != "sha256"}}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compress_all(pairs, cache_path: Path) -> dict[str, str]:
    """Cache GPT compression results per qid. qids already in the cache file are reused without
    calling GPT; only missing qids are sent and added to the cache (the cache is written to disk
    after every call, so an interrupted run can resume).

    On load, each entry's SHA-256 hash is recomputed and compared with the stored hash. If any
    differ, the cache is treated as modified after creation and an exception is raised.
    """
    from openai import OpenAI

    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for qid, record in cache.items():
            stored_hash = record.get("sha256")
            recomputed = _compute_entry_hash(qid, record)
            if stored_hash != recomputed:
                raise RuntimeError(
                    f"Cache integrity check failed: hash mismatch for qid={qid} "
                    f"(stored={stored_hash}, recomputed={recomputed}). The cache file may have "
                    f"been modified after it was created: {cache_path}"
                )
        print(f"  Cache integrity check passed for {len(cache)} entries: {cache_path}")

    n_cached = sum(1 for query, _ in pairs if query.qid in cache)
    print(f"  Compression cache hits: {n_cached}/{len(pairs)} (the rest will call GPT)")

    # If every query hits the cache GPT is never called, so this runs without OPENAI_API_KEY.
    client = None
    seen_qids: set[str] = set()
    for query, _ in pairs:
        if query.qid in cache or query.qid in seen_qids:
            continue
        seen_qids.add(query.qid)
        if client is None:
            client = OpenAI()
        record = compress_query(client, query.text)
        record["cached_at"] = datetime.now(timezone.utc).isoformat()
        record["sha256"] = _compute_entry_hash(query.qid, record)
        cache[query.qid] = record
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    return {qid: record["text"] for qid, record in cache.items()}


# ------------------------------------------------------------------
# Instance selection — for every query with at least one gold document, each way of picking
# one of its gold documents becomes a separate needle=1 instance.
# ------------------------------------------------------------------
def load_selected_pairs(direction, args):
    qrels_raw = json.loads(Path(args.qrels).read_text(encoding="utf-8"))

    gold_docs = build_corpus.load_gold_docs()
    build_corpus.verify_against_delivered(gold_docs)
    doc_by_id = {doc.doc_id: doc for docs in gold_docs.values() for doc in docs}

    queries = load_queries(direction.query_csv(args.query_set), direction.key)
    query_by_id = {q.qid: q for q in queries}

    pairs = []
    missing_q = missing_d = 0

    # The same query (qid) can appear in pairs once per gold document. compress_all() caches
    # per qid, so GPT compression is called only once per qid.
    base_qids = [qid for qid, docs in qrels_raw.items() if len(docs) >= N_NEEDLES]
    for qid in base_qids:
        docs = sorted(qrels_raw[qid])
        query = query_by_id.get(qid)
        if query is None:
            missing_q += 1
            continue
        for combo in combinations(docs, N_NEEDLES):
            gold_ids = list(combo)
            if not all(did in doc_by_id for did in gold_ids):
                missing_d += 1
                continue
            pairs.append((query, gold_ids))
    print(f"{len(base_qids)} of {len(qrels_raw)} queries in qrels have >= {N_NEEDLES} gold "
          f"-> {len(pairs)} needle={N_NEEDLES} instances (all C(k,{N_NEEDLES}) combinations, no sampling)")

    if missing_q or missing_d:
        print(f"Warning: {missing_q} queries and {missing_d} gold documents could not be matched")
    if not pairs:
        raise SystemExit("No instances to evaluate.")
    return pairs, gold_docs, doc_by_id


# ------------------------------------------------------------------
# RRF
# ------------------------------------------------------------------
def weighted_rrf(weighted_lists: list[tuple[float, list[str]]], k: int = RRF_CONSTANT) -> list[str]:
    scores: dict[str, float] = {}
    for weight, ranked in weighted_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank + 1))
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ------------------------------------------------------------------
# Per-retriever rankings — the BM25 and KURE rankings the proposed RRF needs.
# ------------------------------------------------------------------
def _bm25_score_chunk(distractor_ids, distractor_tokens, needle_tokens, tokenizer_name: str,
                      chunk_pairs, comp_texts) -> tuple[list, list]:
    """Compute BM25 for a subset of entries (needle=1 instances). Each entry has its own corpus
    (distractors + that entry's single needle document), so BM25Okapi has to be rebuilt every
    time (required for an exact needle-in-a-haystack evaluation). That is CPU-heavy, so this
    function is run in several processes to use multiple cores."""
    from rank_bm25 import BM25Okapi
    tok = get_tokenizer(tokenizer_name)

    orig_ranked, comp_ranked = [], []
    for query, gold_ids in chunk_pairs:
        corpus_tokens = distractor_tokens + [needle_tokens[did] for did in gold_ids]
        corpus_ids = distractor_ids + gold_ids
        bm25 = BM25Okapi(corpus_tokens)

        o_scores = bm25.get_scores(tok(query.text))
        orig_ranked.append([corpus_ids[i] for i in np.argsort(-o_scores)])

        c_scores = bm25.get_scores(tok(comp_texts[query.qid]))
        comp_ranked.append([corpus_ids[i] for i in np.argsort(-c_scores)])
    return orig_ranked, comp_ranked


def run_bm25(distractors, pairs, doc_by_id, comp_texts, tokenizer_name: str,
            n_workers: int | None = None) -> tuple[list, list]:
    """A new BM25Okapi must be built per entry (the corpus differs per needle), so the cost
    grows with (number of entries x corpus size). Entries are split into one chunk per CPU
    worker and scored concurrently in separate processes (scoring and corpus construction are
    unchanged — only parallelism is added)."""
    tok = get_tokenizer(tokenizer_name)   # chosen explicitly inside this process
    distractor_ids = [d.doc_id for d in distractors]
    distractor_tokens = [tok(d.text) for d in distractors]
    all_needle_ids = sorted({did for _, gold_ids in pairs for did in gold_ids})
    needle_tokens = {did: tok(doc_by_id[did].text) for did in all_needle_ids}

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(pairs))) if pairs else 1

    if n_workers <= 1:
        orig_ranked, comp_ranked = _bm25_score_chunk(
            distractor_ids, distractor_tokens, needle_tokens, tokenizer_name, pairs, comp_texts)
    else:
        chunk_size = (len(pairs) + n_workers - 1) // n_workers
        chunks = [pairs[i * chunk_size:(i + 1) * chunk_size] for i in range(n_workers)]
        chunks = [c for c in chunks if c]
        orig_ranked, comp_ranked = [], []
        with ProcessPoolExecutor(max_workers=len(chunks),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = [ex.submit(_bm25_score_chunk, distractor_ids, distractor_tokens,
                              needle_tokens, tokenizer_name, chunk, comp_texts)
                    for chunk in chunks]
            for fut in futs:
                o, c = fut.result()
                orig_ranked.extend(o)
                comp_ranked.extend(c)
    print(f"  bm25 done ({len(pairs)} instances, tokenizer={tokenizer_name}, workers={n_workers})",
          flush=True)
    return orig_ranked, comp_ranked


def _cast_embedder_dtype(embedder) -> None:
    """Cast embedder.model (SentenceTransformer) to bf16 if supported, otherwise fp16.
    evaluate.DenseEmbedder has no dtype option, so the model is cast after loading."""
    import torch

    model = embedder.model
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(dtype=torch.bfloat16)
    else:
        model.half()


def _encode_corpus(model_name: str, primary_embedder, texts: list[str],
                   devices: list[str], fp16: bool) -> np.ndarray:
    """Encode a large set of documents. With two or more devices, the documents are split into
    one contiguous range per GPU and encoded concurrently in threads (one model copy per GPU).
    With a single device, primary_embedder encodes everything.

    The built-in sentence-transformers multi-process pool (start_multi_process_pool) is not
    used: running it inside a process that has already initialised CUDA by loading the model
    on --device (and which is itself a child spawned by our ProcessPoolExecutor) can crash the
    child with BrokenProcessPool. Threads release the GIL during GPU work, so multiple GPUs can
    be used concurrently without that problem.
    """
    if len(devices) <= 1 or not texts:
        return primary_embedder.encode(texts, is_query=False)

    n = len(devices)
    chunk_size = (len(texts) + n - 1) // n
    jobs = [(i, devices[i], texts[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]
    jobs = [(i, d, c) for i, d, c in jobs if c]  # skip empty chunks when there are fewer documents than GPUs

    def _encode_one(idx, device, chunk):
        if idx == 0:
            emb = primary_embedder  # reuse the already-loaded model (no extra load)
        else:
            emb = evaluate_mod.DenseEmbedder(model_name, device=device,
                                             batch_size=primary_embedder.batch_size,
                                             query_prompt_name=None)
            if fp16:
                _cast_embedder_dtype(emb)
        return emb.encode(chunk, is_query=False)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(jobs)) as tpe:
        futs = [tpe.submit(_encode_one, i, d, c) for i, d, c in jobs]
        results = [f.result() for f in futs]

    return np.vstack(results)


def run_dense(model_name: str, distractors, pairs, doc_by_id, comp_texts,
             device: str, label: str, extra_devices: list[str] | None = None,
             fp16: bool = False, batch_size: int = 8) -> tuple[list, list]:
    devices = [device] + [d for d in (extra_devices or []) if d]
    embedder = evaluate_mod.DenseEmbedder(model_name, device=device,
                                          batch_size=batch_size, query_prompt_name=None)
    if fp16:
        _cast_embedder_dtype(embedder)

    distractor_vecs = _encode_corpus(model_name, embedder, [d.text for d in distractors],
                                     devices, fp16)
    distractor_ids = [d.doc_id for d in distractors]

    all_needle_ids = sorted({did for _, gold_ids in pairs for did in gold_ids})
    needle_docs = [doc_by_id[did] for did in all_needle_ids]
    needle_vecs_all = _encode_corpus(model_name, embedder, [d.text for d in needle_docs],
                                     devices, fp16)
    needle_idx = {did: i for i, did in enumerate(all_needle_ids)}

    # There are far fewer queries than documents, so queries are encoded on the primary device only.
    orig_q_vecs = embedder.encode([q.text for q, _ in pairs], is_query=True)
    comp_q_vecs = embedder.encode([comp_texts[q.qid] for q, _ in pairs], is_query=True)

    orig_ranked, comp_ranked = [], []
    for i, (query, gold_ids) in enumerate(pairs):
        nv = needle_vecs_all[[needle_idx[did] for did in gold_ids]]
        doc_vecs = np.vstack([distractor_vecs, nv])
        doc_ids = distractor_ids + gold_ids

        o_scores = orig_q_vecs[i] @ doc_vecs.T
        orig_ranked.append([doc_ids[j] for j in np.argsort(-o_scores)])
        c_scores = comp_q_vecs[i] @ doc_vecs.T
        comp_ranked.append([doc_ids[j] for j in np.argsort(-c_scores)])
    dtype_note = " fp16/bf16" if fp16 else ""
    print(f"  {label} done ({len(pairs)} instances, devices={devices}{dtype_note})", flush=True)
    return orig_ranked, comp_ranked


def recall_from_topk(top_ids_per_query: list[list[str]], pairs) -> tuple[float, list[dict]]:
    """The same query appears in several needle instances, so the needle document ID is
    appended to the output qid to tell them apart."""
    rows = []
    for (query, gold_ids), top_ids in zip(pairs, top_ids_per_query):
        hit_count = len(set(top_ids[:TOP_K]) & set(gold_ids))
        qid_out = f"{query.qid}::{'-'.join(gold_ids)}"
        rows.append({"qid": qid_out, "base_qid": query.qid, "hit_count": hit_count,
                     "recall": round(hit_count / len(gold_ids), 4)})
    macro = float(np.mean([r["recall"] for r in rows])) if rows else 0.0
    return macro, rows


# ------------------------------------------------------------------
# Per-entry log (one JSONL line per entry) + confusion matrix
# ------------------------------------------------------------------
def log_entries_and_build_confusion_matrix(pairs, ranked, comp_texts, log_path: Path) -> dict:
    """Write one line per entry (needle=1 instance in pairs) to both stdout and log_path
    (JSONL). Each line starts with the answer (gold document ID) and the model's prediction
    (top-1 predicted document ID, hit within recall@20), followed by the full original query
    (query) and the full GPT-compressed query (compressed_query), both untruncated.

    Every entry in this recall evaluation has exactly one gold document, so the true class is
    always Positive. The confusion matrix therefore only splits into found within top-20 (TP)
    and not found (FN); there are no Negative examples, so TN/FP are always 0.
    """
    tp = fn = 0
    with log_path.open("w", encoding="utf-8") as f:
        for i, (query, gold_ids) in enumerate(pairs):
            gold_id = gold_ids[0]  # N_NEEDLES == 1, so gold_ids always has one element
            top_ids = ranked[i]
            hit = gold_id in top_ids[:TOP_K]
            rank = (top_ids.index(gold_id) + 1) if gold_id in top_ids else None
            pred_top1 = top_ids[0] if top_ids else None

            if hit:
                tp += 1
            else:
                fn += 1

            entry = {
                "entry": i + 1,
                "hit_at_20": hit,
                "predicted_rank_of_gold": rank,
                "gold": gold_id,
                "predicted_top1": pred_top1,
                "qid": f"{query.qid}::{gold_id}",
                "base_qid": query.qid,
                "query": query.text,
                "compressed_query": comp_texts[query.qid],
            }
            line = json.dumps(entry, ensure_ascii=False)
            print(line)
            f.write(line + "\n")

    return {"TP": tp, "FN": fn, "TN": 0, "FP": 0}


# ------------------------------------------------------------------
# Entry points for parallel execution (passed directly to ProcessPoolExecutor)
# ------------------------------------------------------------------
def _run_bm25_entry(args):
    distractors, pairs, doc_by_id, comp_texts, tokenizer_name, bm25_workers = args
    return run_bm25(distractors, pairs, doc_by_id, comp_texts, tokenizer_name,
                    n_workers=bm25_workers)


def _run_kure_entry(args):
    distractors, pairs, doc_by_id, comp_texts, device, extra_devices, fp16, batch_size = args
    return run_dense(KURE_MODEL_NAME, distractors, pairs, doc_by_id, comp_texts, device, "kure",
                     extra_devices=extra_devices, fp16=fp16, batch_size=batch_size)


def main() -> None:
    args = parse_args()

    config.check_data_root()
    direction = DIRECTIONS[args.direction]
    pairs, gold_docs, doc_by_id = load_selected_pairs(direction, args)
    distractors = gold_docs[direction.corpus_category]
    print(f"{direction.key} ({args.query_set}) needle={N_NEEDLES} (all combinations): "
          f"{len(pairs)} instances, {len(distractors)} distractors ({direction.corpus_category})")

    extra_devices = [d.strip() for d in args.extra_devices.split(",") if d.strip()]
    kure_devices = [args.device] + extra_devices
    print(f"GPU assignment -> kure: {kure_devices} (only document encoding is split; queries on "
          f"{args.device}), batch_size={args.batch_size}, fp16={args.fp16} (BM25 on CPU)")

    tag_suffix = f"_{args.tag}" if args.tag else ""

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The compression cache is shared per direction + query_set (independent of the tag —
    # runs with different tags reuse the same cache and only call GPT for missing qids).
    cache_path = out_dir / f"compression_cache_{direction.key}_{args.query_set}.json"
    print(f"Query compression cache: {cache_path}")
    comp_texts = compress_all(pairs, cache_path)  # GPT calls are sequential (network I/O; avoids cache file conflicts)

    # The proposed method (ours_rrf_bm25_kure) only needs the BM25 and KURE rankings.
    print("Starting BM25 / KURE in parallel...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
        bm25_workers = args.bm25_workers if args.bm25_workers > 0 else None
        fut_bm25 = ex.submit(_run_bm25_entry,
                             (distractors, pairs, doc_by_id, comp_texts, args.tokenizer,
                              bm25_workers))
        fut_kure = ex.submit(_run_kure_entry,
                             (distractors, pairs, doc_by_id, comp_texts, args.device,
                              extra_devices, args.fp16, args.batch_size))

        orig_bm25, comp_bm25 = fut_bm25.result()
        orig_kure, comp_kure = fut_kure.result()
    print(f"BM25 / KURE finished ({time.time()-t0:.1f}s)")

    # Proposed method: weighted RRF of BM25 49.5% + original-query KURE 1% + compressed-query KURE 49.5%.
    ours_ranked = []
    for i in range(len(pairs)):
        ids_bm25 = orig_bm25[i][:RRF_CANDIDATE_K]
        ids_orig_kure = orig_kure[i][:RRF_CANDIDATE_K]
        ids_comp_kure = comp_kure[i][:RRF_CANDIDATE_K]

        ours_ranked.append(weighted_rrf([
            (OURS_WEIGHTS["bm25"], ids_bm25),
            (OURS_WEIGHTS["orig_kure"], ids_orig_kure),
            (OURS_WEIGHTS["comp_kure"], ids_comp_kure),
        ]))

    summary = {"direction": args.direction, "query_set": args.query_set,
              "n_needles": N_NEEDLES, "needle_selection": "all_gold_combinations",
              "n_queries": len(pairs), "top_k": TOP_K, "method": OURS_METHOD_NAME,
              "rrf_weights": OURS_WEIGHTS}

    macro, rows = recall_from_topk(ours_ranked, pairs)
    summary[OURS_METHOD_NAME] = {"recall@20_macro": round(macro, 4),
                                 "total_hits": sum(r["hit_count"] for r in rows),
                                 "total_gold": sum(len(g) for _, g in pairs)}

    file_stem = f"rag_ours_only_{direction.key}_{args.query_set}_n{N_NEEDLES}_{args.tokenizer}{tag_suffix}"

    # Write one line per entry (needle=1 instance) with input / answer / prediction to stdout
    # and a file, and build the confusion matrix from it.
    entries_log_path = out_dir / f"{file_stem}.entries.jsonl"
    print()
    print(f"Writing {len(pairs)} per-entry log lines (file: {entries_log_path})")
    confusion = log_entries_and_build_confusion_matrix(pairs, ours_ranked, comp_texts, entries_log_path)
    summary["confusion_matrix"] = confusion

    out_path = out_dir / f"{file_stem}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "rows": {OURS_METHOD_NAME: rows}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    print()
    print(f"[{direction.key} {args.query_set} n={N_NEEDLES}] {len(pairs)} instances")
    m = summary[OURS_METHOD_NAME]
    print("Confusion Matrix (at recall@20; every entry has a gold document, so TN/FP=0)")
    print(f"  TP(Hit)  = {confusion['TP']}")
    print(f"  FN(Miss) = {confusion['FN']}")
    print(f"  TN       = {confusion['TN']}")
    print(f"  FP       = {confusion['FP']}")
    print(f"  Overall recall@{TOP_K} = {m['recall@20_macro']:.4f}  "
          f"({m['total_hits']}/{m['total_gold']})")
    print()
    print(f"Entry log saved: {entries_log_path}")
    print(f"Summary saved: {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
