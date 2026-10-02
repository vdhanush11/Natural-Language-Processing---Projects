from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os
import re
import time
import logging

import pycountry
import torch
from langdetect import detect_langs, DetectorFactory, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DetectorFactory.seed = 42
logger = logging.getLogger("translation_service")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
MODEL_DIR = os.getenv("MODEL_DIR", "").strip()

_NLLB_LANGUAGES = (
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn",
    "ajp_Arab", "aka_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab",
    "ary_Arab", "arz_Arab", "ars_Arab", "asm_Beng", "ast_Latn", "awa_Deva",
    "ayr_Latn", "azb_Arab", "azj_Latn",
    "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng",
    "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn",
    "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab",
    "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn",
    "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn",
    "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn", "fra_Latn",
    "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn",
    "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva",
    "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn",
    "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn",
    "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor",
    "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn",
    "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn",
    "kon_Latn", "kor_Hang", "lao_Laoo",
    "lij_Latn", "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn",
    "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "lvs_Latn",
    "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl",
    "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "nld_Latn",
    "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn",
    "nus_Latn", "nya_Latn", "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
    "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn",
    "prs_Arab", "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn",
    "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn",
    "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
    "spa_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn",
    "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu",
    "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn",
    "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng",
    "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
    "swh_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr",
    "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn",
)

_COMMON_NAMES = {
    "eng_Latn": "English", "fra_Latn": "French", "spa_Latn": "Spanish",
    "hin_Deva": "Hindi", "tam_Taml": "Tamil", "deu_Latn": "German",
    "ita_Latn": "Italian", "por_Latn": "Portuguese", "rus_Cyrl": "Russian",
    "arb_Arab": "Arabic", "ben_Beng": "Bengali", "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean", "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)", "tel_Telu": "Telugu",
    "mar_Deva": "Marathi", "urd_Arab": "Urdu", "vie_Latn": "Vietnamese",
}

_VARIANT_NAMES = {
    "ajp_Arab": "South Levantine Arabic",
    "arb_Arab": "Modern Standard Arabic",
    "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic",
    "acm_Arab": "Mesopotamian Arabic",
    "acq_Arab": "Taizzi-Adeni Arabic",
    "apc_Arab": "North Levantine Arabic",
    "ars_Arab": "Najdi Arabic",
    "azb_Arab": "South Azerbaijani",
    "azj_Latn": "North Azerbaijani",
    "pbt_Arab": "Southern Pashto",
    "pes_Arab": "Iranian Persian",
    "prs_Arab": "Dari Persian",
    "uzn_Latn": "Northern Uzbek",
    "ydd_Hebr": "Eastern Yiddish",
}

_SCRIPT_NAMES = {
    "Arab": "Arabic script", "Armn": "Armenian script", "Beng": "Bengali script",
    "Cyrl": "Cyrillic script", "Deva": "Devanagari script", "Ethi": "Ethiopic script",
    "Geor": "Georgian script", "Grek": "Greek script", "Guru": "Gurmukhi script",
    "Hang": "Hangul script", "Hans": "Simplified Chinese", "Hant": "Traditional Chinese",
    "Hebr": "Hebrew script", "Jpan": "Japanese script", "Khmr": "Khmer script",
    "Knda": "Kannada script", "Laoo": "Lao script", "Latn": "Latin script",
    "Mlym": "Malayalam script", "Mymr": "Myanmar script", "Olck": "Ol Chiki script",
    "Orya": "Odia script", "Sinh": "Sinhala script", "Taml": "Tamil script",
    "Telu": "Telugu script", "Thai": "Thai script", "Tibt": "Tibetan script",
    "Tfng": "Tifinagh script",
}

_LANGUAGE_VARIANT_COUNTS = {}
for _code in _NLLB_LANGUAGES:
    _LANGUAGE_VARIANT_COUNTS[_code.split("_")[0]] = _LANGUAGE_VARIANT_COUNTS.get(_code.split("_")[0], 0) + 1


def _language_name(code: str) -> str:
    if code in _COMMON_NAMES:
        return _COMMON_NAMES[code]
    if code in _VARIANT_NAMES:
        return _VARIANT_NAMES[code]
    base, script = code.split("_")
    language = pycountry.languages.get(alpha_3=base)
    name = language.name if language else base.upper()
    if _LANGUAGE_VARIANT_COUNTS[base] > 1:
        name = f"{name} ({_SCRIPT_NAMES.get(script, script + ' script')})"
    return name


LANGUAGES = {
    code: {"name": _language_name(code), "nllb": code}
    for code in _NLLB_LANGUAGES
}

_LANGUAGE_ALIASES = {
    "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "hi": "hin_Deva",
    "ta": "tam_Taml", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "bn": "ben_Beng", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "zh": "zho_Hans", "te": "tel_Telu", "mr": "mar_Deva",
    "ur": "urd_Arab", "vi": "vie_Latn",
}

SUPPORTED_PAIRS = [(source, target) for source in LANGUAGES for target in LANGUAGES if source != target]

_PROTECT_PATTERNS = [
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b(?:ERR|ERROR|CODE|E|SKU|ID)[-_ ]?\d{2,}\b", re.I),
    re.compile(r"\b[A-Z]{2,}[-_]?\d+[A-Z0-9-]*\b"),
    re.compile(r"#\w+|@\w+"),
]
_SENTINEL_PREFIX = "NLLBKEEP"

@dataclass
class DetectionResult:
    iso: Optional[str]
    confidence: float
    supported: bool
    candidates: List[Tuple[str, float]]
    note: str = ""

def detect_language(text: str, default_iso: str = "en") -> DetectionResult:
    if not text or not text.strip():
        return DetectionResult(default_iso, 0.0, True, [], "Empty input; defaulted to English.")
    try:
        ranked = detect_langs(text)
    except LangDetectException:
        return DetectionResult(default_iso, 0.0, True, [], "Detector could not decide; defaulted to English.")
    candidates = [(_LANGUAGE_ALIASES.get(c.lang, c.lang), float(c.prob)) for c in ranked]
    top_iso, top_prob = candidates[0]
    default_iso = _LANGUAGE_ALIASES.get(default_iso, default_iso)
    if top_iso not in LANGUAGES:
        return DetectionResult(
            default_iso, top_prob, False, candidates,
            f"Detected '{top_iso}' is outside the supported registry; defaulted to English."
        )
    note = ""
    if len(candidates) > 1 and candidates[1][1] > 0.30:
        note = f"Possible code-switching: runner-up '{candidates[1][0]}' (p={candidates[1][1]:.2f})."
    return DetectionResult(top_iso, top_prob, True, candidates, note)

def protect_terms(text: str):
    spans = []
    for pattern in _PROTECT_PATTERNS:
        for m in pattern.finditer(text):
            if m.group(0).strip():
                spans.append((m.start(), m.end(), m.group(0)))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept = []
    for st, en, val in spans:
        if any(not (en <= k0 or st >= k1) for k0, k1, _ in kept):
            continue
        kept.append((st, en, val))
    ordered = sorted(kept, key=lambda s: s[0])
    idx_of = {(st, en): i for i, (st, en, _) in enumerate(ordered)}
    mapping = {}
    for st, en, val in sorted(kept, key=lambda s: s[0], reverse=True):
        placeholder = f"{_SENTINEL_PREFIX}{idx_of[(st, en)]}"
        text = text[:st] + placeholder + text[en:]
        mapping[placeholder] = val
    return text, mapping

def restore_terms(text: str, mapping: Dict[str, str]) -> str:
    for placeholder, original in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        digits = placeholder[len(_SENTINEL_PREFIX):]
        pat = re.compile(re.escape(_SENTINEL_PREFIX) + r"\s*" + digits + r"(?!\d)", re.I)
        text = pat.sub(lambda _m, o=original: o, text)
    return text

def postprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

class TranslationService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_DIR if MODEL_DIR else DEFAULT_MODEL
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.loading_error = None
        self._load()

    def _load(self):
        try:
            logger.info("Loading translation model: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype)
            self.model.to(self.device)
            self.model.eval()
            for iso in LANGUAGES:
                _ = self.target_bos_id(iso)
            self.loaded = True
            logger.info("Model ready on %s", self.device)
        except Exception as exc:
            self.loading_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model load failed")

    def health(self):
        return {
            "status": "ok" if self.loaded else "degraded",
            "model_loaded": self.loaded,
            "device": self.device,
            "model": self.model_name,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "error": self.loading_error,
        }

    def model_info(self):
        return {
            "model_name": self.model_name,
            "base_model": DEFAULT_MODEL,
            "device": self.device,
            "architecture": "Transformer encoder-decoder (NLLB-200 distilled 600M)",
            "languages": LANGUAGES,
            "supported_pairs": [list(p) for p in SUPPORTED_PAIRS],
            "generation_defaults": {
                "num_beams": 5,
                "max_new_tokens": 512,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
        }

    def nllb_code(self, iso: str):
        iso = _LANGUAGE_ALIASES.get(iso, iso)
        if iso not in LANGUAGES:
            raise ValueError(f"Unsupported language '{iso}'. Supported: {sorted(LANGUAGES)}")
        return LANGUAGES[iso]["nllb"]

    def target_bos_id(self, tgt_iso: str):
        token = self.nllb_code(tgt_iso)
        tok_id = self.tokenizer.convert_tokens_to_ids(token)
        if tok_id is None or tok_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Could not resolve NLLB token id for '{token}'.")
        return tok_id

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        tgt_iso: str,
        src_iso: Optional[str] = None,
        protect: bool = True,
        generation: Optional[dict] = None,
    ):
        if not self.loaded:
            raise RuntimeError("Translation model is not loaded. Check the server logs.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text is empty.")
        tgt_iso = _LANGUAGE_ALIASES.get(tgt_iso, tgt_iso)
        if tgt_iso not in LANGUAGES:
            raise ValueError(f"Unsupported target '{tgt_iso}'.")
        text = text.strip()
        t0 = time.perf_counter()

        src_iso = _LANGUAGE_ALIASES.get(src_iso, src_iso) if src_iso else None
        det = detect_language(text) if src_iso is None else DetectionResult(
            src_iso, 1.0, True, [(src_iso, 1.0)], "Source language provided by user."
        )
        resolved_src = det.iso or "en"
        if resolved_src not in LANGUAGES:
            resolved_src = "en"

        src_code, tgt_code = self.nllb_code(resolved_src), self.nllb_code(tgt_iso)
        self.tokenizer.src_lang = src_code

        masked, mapping = protect_terms(text) if protect else (text, {})
        enc = self.tokenizer(
            masked,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        gen_cfg = {
            "num_beams": 5,
            "max_new_tokens": 512,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        if generation:
            gen_cfg.update(generation)

        generated = self.model.generate(
            **enc,
            forced_bos_token_id=self.target_bos_id(tgt_iso),
            **gen_cfg,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        if protect:
            decoded = restore_terms(decoded, mapping)
        output = postprocess(decoded)

        return {
            "source_text": text,
            "translated_text": output,
            "source_language": resolved_src,
            "target_language": tgt_iso,
            "source_language_name": LANGUAGES[resolved_src]["name"],
            "target_language_name": LANGUAGES[tgt_iso]["name"],
            "source_nllb": src_code,
            "target_nllb": tgt_code,
            "detection": asdict(det),
            "num_input_tokens": int(enc["input_ids"].shape[1]),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "protected_terms": list(mapping.values()),
            "generation": gen_cfg,
        }

    def translate_batch(
        self,
        messages: Sequence[str],
        tgt_iso: str,
        src_iso: Optional[str] = None,
        protect: bool = True,
    ):
        if not messages:
            return []
        # Preserve the notebook's batching idea: group by resolved source language.
        resolved = []
        for i, message in enumerate(messages):
            src = src_iso or detect_language(message).iso or "en"
            if src not in LANGUAGES:
                src = "en"
            resolved.append((i, src))

        results = [None] * len(messages)
        groups = {}
        for i, src in resolved:
            groups.setdefault(src, []).append(i)

        for src, indices in groups.items():
            self.tokenizer.src_lang = self.nllb_code(src)
            for start in range(0, len(indices), 8):
                chunk = indices[start:start+8]
                texts = [messages[i].strip() for i in chunk]
                masked, maps = [], []
                for text in texts:
                    mt, mp = protect_terms(text) if protect else (text, {})
                    masked.append(mt); maps.append(mp)

                enc = self.tokenizer(
                    masked, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                t0 = time.perf_counter()
                gen = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.target_bos_id(tgt_iso),
                    num_beams=5, max_new_tokens=512,
                    no_repeat_ngram_size=3, length_penalty=1.0,
                    early_stopping=True,
                )
                per_item_ms = (time.perf_counter() - t0) * 1000 / max(len(chunk), 1)
                decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)

                for j, i in enumerate(chunk):
                    out = restore_terms(decoded[j], maps[j]) if protect else decoded[j]
                    results[i] = {
                        "source_text": texts[j],
                        "translated_text": postprocess(out),
                        "source_language": src,
                        "target_language": tgt_iso,
                        "source_language_name": LANGUAGES[src]["name"],
                        "target_language_name": LANGUAGES[tgt_iso]["name"],
                        "latency_ms": round(per_item_ms, 1),
                        "num_input_tokens": int(enc["input_ids"][j].shape[0]),
                    }
        return results
