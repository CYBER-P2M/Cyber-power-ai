"""Evidence-grounded reports for Normal / Fault / Attack.

Modes:
    grounded_template:
        No LLM loading or generation.

    llm_draft_guarded:
        Generate an LLM draft, but keep the final narrative deterministic
        and grounded in the evidence packet.

This module does not claim general semantic hallucination detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time

from collections import OrderedDict
from pathlib import Path
from uuid import uuid4


try:
    from . import report_prompts, report_charts
except ImportError:
    import report_prompts
    import report_charts


ROOT = Path(__file__).resolve().parents[1]

SECTION_ORDER = [
    "Summary",
    "Key Evidence",
    "Interpretation",
    "Possible Impact",
    "Recommended Actions",
    "Confidence & Limitations",
]

LLM_SECTIONS = [
    name
    for name in SECTION_ORDER
    if name != "Key Evidence"
]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def _load_config():
    for module_name in ("src.config", "config"):
        try:
            return __import__(module_name, fromlist=["x"])
        except ImportError:
            continue

    return None


_CFG = _load_config()


def _cfg(name, default):
    if _CFG is None:
        return default

    return getattr(_CFG, name, default)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    return bool(value)


def settings():
    fallbacks = _cfg(
        "LLM_FALLBACKS",
        ["Qwen/Qwen2.5-0.5B-Instruct"],
    )

    if isinstance(fallbacks, str):
        fallbacks = [fallbacks]
    elif fallbacks is None:
        fallbacks = []

    # Respect an existing Hugging Face cache by default.
    # Configure LLM_CACHE_DIR explicitly to use another directory.
    cache_dir = _cfg("LLM_CACHE_DIR", None)

    return {
        "CLASS_NORMAL": _cfg("CLASS_NORMAL", "Normal"),
        "CLASS_FAULT": _cfg("CLASS_FAULT", "Fault"),
        "CLASS_ATTACK": _cfg("CLASS_ATTACK", "Attack"),

        "REPORT_MODE": str(
            _cfg("REPORT_MODE", "grounded_template")
        ).strip().lower(),

        "REPORT_LANG": str(
            _cfg("REPORT_LANG", "en")
        ).strip().lower(),

        "LLM_MODEL_NAME": _cfg(
            "LLM_MODEL_NAME",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),

        "LLM_FALLBACKS": list(fallbacks),

        "LLM_MAX_NEW_TOKENS": max(
            1,
            int(_cfg("LLM_MAX_NEW_TOKENS", 450)),
        ),

        "LLM_ALLOW_DOWNLOAD": _as_bool(
            _cfg("LLM_ALLOW_DOWNLOAD", True)
        ),

        "LLM_RETRY_SECONDS": max(
            0.0,
            float(_cfg("LLM_RETRY_SECONDS", 120)),
        ),

        "LLM_CACHE_DIR": (
            Path(cache_dir)
            if cache_dir is not None
            else None
        ),

        "REPORTS_DIR": Path(
            _cfg(
                "REPORTS_DIR",
                ROOT / "results" / "reports",
            )
        ),

        "LOG_DIR": Path(
            _cfg("LOG_DIR", ROOT / "logs")
        ),

        "BASELINE_PATH": Path(
            _cfg(
                "BASELINE_PATH",
                ROOT / "models" / "baseline_stats.json",
            )
        ),

        "FEATURE_DICT_PATH": Path(
            _cfg(
                "FEATURE_DICT_PATH",
                ROOT / "src" / "feature_dictionary.json",
            )
        ),

        "RELIABILITY_DIR": Path(
            _cfg(
                "REPORT_RELIABILITY_DIR",
                ROOT / "results" / "classification_reports",
            )
        ),

        # Backward-compatible RF-specific override.
        "RELIABILITY_CSV": Path(
            _cfg(
                "RELIABILITY_CSV",
                ROOT
                / "results"
                / "classification_reports"
                / "random_forest.csv",
            )
        ),

        "PACKET_FEATURES": max(
            1,
            int(_cfg("REPORT_PACKET_FEATURES", 20)),
        ),

        "TABLE_ROWS": max(
            1,
            int(_cfg("REPORT_TABLE_ROWS", 10)),
        ),

        "WEAK_Z": float(
            _cfg("WEAK_Z_THRESHOLD", 2.0)
        ),

        "LOW_RECALL": float(
            _cfg("LOW_RECALL_THRESHOLD", 0.5)
        ),

        "Z_CAP": max(
            1.0,
            float(_cfg("Z_CAP", 50.0)),
        ),
    }


# ---------------------------------------------------------------------
# Logging and exceptions
# ---------------------------------------------------------------------

def _logger():
    logger = logging.getLogger("powergrid.report")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        try:
            log_dir = settings()["LOG_DIR"]
            log_dir.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(
                log_dir / "report.log",
                encoding="utf-8",
            )

            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s"
                )
            )

            logger.addHandler(handler)

        except OSError:
            logger.addHandler(logging.NullHandler())

    return logger


class LLMUnavailable(RuntimeError):
    pass


class BaselineMissing(FileNotFoundError):
    pass


class BaselineMismatch(ValueError):
    pass


# ---------------------------------------------------------------------
# Normalization and numeric helpers
# ---------------------------------------------------------------------

def _normalize_name(value):
    return str(value).strip().casefold()


def _normalized_lookup(mapping, description):
    """Build a case-insensitive lookup without silently merging keys."""
    if not isinstance(mapping, dict):
        raise ValueError(
            f"{description} must be a dictionary."
        )

    lookup = {}

    for original_name, value in mapping.items():
        key = _normalize_name(original_name)

        if key in lookup:
            previous_name = lookup[key][0]

            raise ValueError(
                f"Ambiguous {description} keys: "
                f"{previous_name!r} and {original_name!r}."
            )

        lookup[key] = (str(original_name), value)

    return lookup


def _finite_float(value):
    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Expected a finite numeric value.")

    return number


def _sig(value, digits=4):
    number = _finite_float(value)

    if number == 0:
        return 0.0

    return float(f"{number:.{digits}g}")


def _guard_number(value):
    try:
        number = _finite_float(value)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"

    return f"{number:g}"


def _canonical_model(model_name):
    value = re.sub(
        r"[\s-]+",
        "_",
        str(model_name).strip().lower(),
    )

    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "xgb": "xgboost",
        "cat": "catboost",
    }

    value = aliases.get(value, value)

    if value not in {
        "random_forest",
        "xgboost",
        "catboost",
    }:
        raise ValueError(
            f"Unknown model for reporting: {model_name}"
        )

    return value


def _read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------
# Baseline, dictionary, reliability
# ---------------------------------------------------------------------

def load_baseline():
    path = settings()["BASELINE_PATH"]

    if not path.exists():
        raise BaselineMissing(
            f"{path} was not found.\n"
            "Run: python tools/build_baseline.py"
        )

    baseline = _read_json(path)

    if (
        not isinstance(baseline, dict)
        or not isinstance(baseline.get("features"), dict)
    ):
        raise BaselineMismatch(
            "Baseline must contain a 'features' dictionary."
        )

    return baseline


def load_feature_dictionary():
    path = settings()["FEATURE_DICT_PATH"]

    if not path.exists():
        _logger().warning(
            "Feature dictionary missing: %s",
            path,
        )
        return {}

    obj = _read_json(path)

    if not isinstance(obj, dict):
        raise ValueError(
            "Feature dictionary JSON must be an object."
        )

    features = obj.get("features", obj)

    if not isinstance(features, dict):
        raise ValueError(
            "Feature descriptions must be a dictionary."
        )

    return features


def load_reliability(model_name="random_forest"):
    import pandas as pd

    s = settings()
    model_name = _canonical_model(model_name)

    if model_name == "random_forest":
        path = s["RELIABILITY_CSV"]
    else:
        path = (
            s["RELIABILITY_DIR"]
            / f"{model_name}.csv"
        )

    result = {
        "model": model_name,
        "path": str(path),
        "per_class": OrderedDict(),
    }

    if not path.exists():
        result["warning"] = (
            f"Historical evaluation for {model_name} "
            "is unavailable. Run src/evaluate_models.py."
        )
        return result

    try:
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str).str.strip()

        if frame.index.duplicated().any():
            raise ValueError(
                "Evaluation report has duplicate class rows."
            )

        for cls in (
            s["CLASS_NORMAL"],
            s["CLASS_FAULT"],
            s["CLASS_ATTACK"],
        ):
            if cls not in frame.index:
                continue

            row = frame.loc[cls]

            precision = _finite_float(row["precision"])
            recall = _finite_float(row["recall"])
            support = _finite_float(row["support"])

            if not (
                0 <= precision <= 1
                and 0 <= recall <= 1
                and support >= 0
            ):
                raise ValueError(
                    f"Invalid evaluation values for {cls}."
                )

            result["per_class"][cls] = {
                "precision": _sig(precision),
                "recall": _sig(recall),
                "precision_pct": round(
                    precision * 100,
                    1,
                ),
                "recall_pct": round(
                    recall * 100,
                    1,
                ),
                "support": int(support),
            }

    except Exception as exc:
        _logger().exception(
            "Unable to load reliability report: %s",
            path,
        )

        result["per_class"] = OrderedDict()
        result["warning"] = (
            "Historical evaluation could not be read: "
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ---------------------------------------------------------------------
# Evidence packet
# ---------------------------------------------------------------------

def build_evidence(
    record,
    prediction,
    baseline=None,
    reliability=None,
    dictionary=None,
    synthetic=False,
):
    s = settings()

    if baseline is None:
        baseline = load_baseline()

    if dictionary is None:
        dictionary = load_feature_dictionary()

    model_name = _canonical_model(
        prediction.get("model", "random_forest")
    )

    if reliability is None:
        reliability = load_reliability(model_name)

    stats = baseline.get("features")

    if not isinstance(stats, dict):
        raise BaselineMismatch(
            "Baseline 'features' is missing or invalid."
        )

    baseline_lookup = _normalized_lookup(
        stats,
        "baseline feature",
    )

    dictionary_lookup = _normalized_lookup(
        dictionary,
        "feature dictionary",
    )

    # Do not silently accept duplicate input names after normalization.
    _normalized_lookup(record, "record feature")

    cls = prediction["predicted_class"]

    expected_classes = {
        s["CLASS_NORMAL"],
        s["CLASS_FAULT"],
        s["CLASS_ATTACK"],
    }

    if cls not in expected_classes:
        raise ValueError(
            f"Unexpected predicted class: {cls}"
        )

    probabilities = {
        str(name): _finite_float(value)
        for name, value
        in prediction["probabilities"].items()
    }

    if set(probabilities) != expected_classes:
        raise ValueError(
            "Probabilities must contain exactly "
            "Normal, Fault and Attack "
            "(or the configured class names)."
        )

    if any(
        value < 0 or value > 1
        for value in probabilities.values()
    ):
        raise ValueError(
            "Class probabilities must be in [0, 1]."
        )

    if not math.isclose(
        sum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "Class probabilities must sum to one."
        )

    risk = 1.0 - probabilities[s["CLASS_NORMAL"]]

    rows = []
    missing = []
    invalid = []
    undocumented = []
    notes = []

    for feature_name, value in record.items():
        key = _normalize_name(feature_name)

        if key not in baseline_lookup:
            missing.append(str(feature_name))
            continue

        baseline_name, feature_stats = baseline_lookup[key]

        try:
            if not isinstance(feature_stats, dict):
                raise ValueError(
                    "Invalid baseline feature statistics."
                )

            numeric_value = _finite_float(value)
            mean = _finite_float(feature_stats["mean"])
            std = _finite_float(feature_stats["std"])

            if std < 0:
                raise ValueError(
                    "Standard deviation cannot be negative."
                )

            delta = numeric_value - mean
            z_raw = delta / max(std, 1e-3)

            if not math.isfinite(z_raw):
                raise ValueError(
                    "Non-finite baseline deviation."
                )

            z_value = max(
                -s["Z_CAP"],
                min(s["Z_CAP"], z_raw),
            )

            dictionary_match = dictionary_lookup.get(key)
            info = (
                dictionary_match[1]
                if dictionary_match is not None
                else {}
            )

            if isinstance(info, str):
                info = {"meaning": info}

            if not isinstance(info, dict):
                info = {}

            meaning = str(
                info.get("meaning", "")
            ).strip()

            if not meaning:
                meaning = "undocumented feature"
                undocumented.append(str(feature_name))

            rows.append(
                {
                    "feature": str(feature_name),
                    "baseline_feature": baseline_name,
                    "meaning": meaning,
                    "kind": str(
                        info.get("kind", "unknown")
                    ),
                    "value": _sig(numeric_value),
                    "baseline_mean": _sig(mean),
                    "baseline_std": _sig(std),
                    "deviation": _sig(delta),
                    "z": round(z_value, 2),
                    "z_capped": (
                        abs(z_raw) > s["Z_CAP"]
                    ),
                }
            )

        except (
            TypeError,
            ValueError,
            KeyError,
            OverflowError,
        ):
            invalid.append(str(feature_name))

    rows.sort(
        key=lambda item: abs(item["z"]),
        reverse=True,
    )

    top_features = rows[:s["PACKET_FEATURES"]]

    evidence_weak = (
        not rows
        or all(
            abs(item["z"]) < s["WEAK_Z"]
            for item in rows[:5]
        )
    )

    fault_reliability = reliability.get(
        "per_class",
        {},
    ).get(
        s["CLASS_FAULT"],
        {},
    )

    low_reliability = (
        cls == s["CLASS_FAULT"]
        and fault_reliability.get(
            "recall",
            1.0,
        ) < s["LOW_RECALL"]
    )

    if missing:
        notes.append(
            f"Baseline statistics are missing for "
            f"{len(missing)} supplied features. "
            "No substitute statistics were invented."
        )

    if invalid:
        notes.append(
            f"Skipped {len(invalid)} features with "
            "invalid measurements or baseline statistics."
        )

    if undocumented:
        notes.append(
            f"{len(undocumented)} usable features lack "
            "a documented meaning. No meaning was inferred "
            "from their names."
        )

    if not rows:
        notes.append(
            "No usable baseline feature evidence is available. "
            "The physical or security cause cannot be inferred."
        )

    if reliability.get("warning"):
        notes.append(reliability["warning"])

    if "model" not in prediction:
        notes.append(
            "The prediction did not specify its model; "
            "random_forest was assumed for historical metrics."
        )

    if "risk_score" in prediction:
        try:
            supplied_risk = _finite_float(
                prediction["risk_score"]
            )

            if not math.isclose(
                supplied_risk,
                risk,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                notes.append(
                    "The supplied risk score differed from "
                    "1 - P(Normal); the report uses the "
                    "score derived from class probabilities."
                )

        except (TypeError, ValueError, OverflowError):
            notes.append(
                "The supplied risk score was invalid; "
                "the report derives it from P(Normal)."
            )

    return {
        "record_id": (
            f"{prediction.get('source', 'record')}"
            f"#{prediction.get('index', 0)}"
        ),
        "model": model_name,
        "predicted_class": cls,
        "probabilities_pct": {
            name: round(value * 100, 1)
            for name, value in probabilities.items()
        },
        "risk_score_pct": round(risk * 100, 1),
        "weak_z_threshold": s["WEAK_Z"],
        "evidence_weak": bool(evidence_weak),
        "low_recall_threshold_pct": round(
            s["LOW_RECALL"] * 100,
            1,
        ),
        "low_reliability_class": bool(low_reliability),
        "top_features": top_features,
        "model_reliability": reliability,
        "baseline_class": baseline.get(
            "baseline_class",
            "Normal",
        ),
        "synthetic": bool(synthetic),
        "evidence_notes": notes,
        "missing_baseline_features": missing,
        "invalid_features": invalid,
        "undocumented_features": undocumented,
    }


# ---------------------------------------------------------------------
# LLM loading and process-local reuse
# ---------------------------------------------------------------------

_LLM = None
_LLM_SIGNATURE = None
_LLM_FAILURE = None

_LOAD_LOCK = threading.Lock()
_GENERATE_LOCK = threading.Lock()


def load_llm(force=False):
    """Return a reusable run(messages) -> str callable.

    force=True clears process-local reuse/retry state.
    It does not request force_download from Hugging Face.
    """
    global _LLM
    global _LLM_SIGNATURE
    global _LLM_FAILURE

    s = settings()

    signature = (
        s["LLM_MODEL_NAME"],
        tuple(s["LLM_FALLBACKS"]),
        str(s["LLM_CACHE_DIR"]),
        s["LLM_MAX_NEW_TOKENS"],
        s["LLM_ALLOW_DOWNLOAD"],
    )

    with _LOAD_LOCK:
        if force:
            _LLM = None
            _LLM_SIGNATURE = None
            _LLM_FAILURE = None

        if (
            _LLM is not None
            and _LLM_SIGNATURE == signature
        ):
            return _LLM

        if (
            _LLM_FAILURE is not None
            and _LLM_FAILURE[0] == signature
        ):
            _, failed_at, message = _LLM_FAILURE

            if (
                time.monotonic() - failed_at
                < s["LLM_RETRY_SECONDS"]
            ):
                raise LLMUnavailable(
                    "A recent loading attempt failed; "
                    "retry cooldown is active. "
                    + message
                )

        _LLM = None
        _LLM_SIGNATURE = None

        try:
            import torch

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            cache_kwargs = {}

            if s["LLM_CACHE_DIR"] is not None:
                s["LLM_CACHE_DIR"].mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cache_kwargs["cache_dir"] = str(
                    s["LLM_CACHE_DIR"]
                )

            model_names = list(
                dict.fromkeys(
                    [
                        s["LLM_MODEL_NAME"],
                        *s["LLM_FALLBACKS"],
                    ]
                )
            )

            errors = []

            for model_name in model_names:
                try:
                    def get_pair(local_only):
                        kwargs = {
                            **cache_kwargs,
                            "local_files_only": local_only,
                        }

                        tokenizer = (
                            AutoTokenizer.from_pretrained(
                                model_name,
                                **kwargs,
                            )
                        )

                        model = (
                            AutoModelForCausalLM
                            .from_pretrained(
                                model_name,
                                **kwargs,
                            )
                        )

                        return tokenizer, model

                    try:
                        tokenizer, model = get_pair(True)
                        origin = "disk cache"

                    except OSError:
                        if not s["LLM_ALLOW_DOWNLOAD"]:
                            raise

                        _logger().info(
                            "Resolving model files via hub/cache: %s",
                            model_name,
                        )

                        tokenizer, model = get_pair(False)
                        origin = "hub/cache resolution"

                    model.eval()

                    def run(
                        messages,
                        _tokenizer=tokenizer,
                        _model=model,
                        _max_tokens=s["LLM_MAX_NEW_TOKENS"],
                    ):
                        with _GENERATE_LOCK:
                            prompt = (
                                _tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                            )

                            inputs = _tokenizer(
                                prompt,
                                return_tensors="pt",
                            )

                            generation_kwargs = {
                                "max_new_tokens": _max_tokens,
                                "do_sample": False,
                            }

                            if (
                                _tokenizer.pad_token_id is None
                                and _tokenizer.eos_token_id
                                is not None
                            ):
                                generation_kwargs[
                                    "pad_token_id"
                                ] = _tokenizer.eos_token_id

                            with torch.inference_mode():
                                output = _model.generate(
                                    **inputs,
                                    **generation_kwargs,
                                )

                            prompt_length = inputs[
                                "input_ids"
                            ].shape[1]

                            text = _tokenizer.decode(
                                output[0][prompt_length:],
                                skip_special_tokens=True,
                            )

                            if not text.strip():
                                raise LLMUnavailable(
                                    "The model returned an empty draft."
                                )

                            return text

                    run.model_name = model_name

                    _LLM = run
                    _LLM_SIGNATURE = signature
                    _LLM_FAILURE = None

                    _logger().info(
                        "LLM ready: model=%s source=%s device=cpu",
                        model_name,
                        origin,
                    )

                    return run

                except Exception as exc:
                    errors.append(
                        f"{model_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    _logger().exception(
                        "LLM loading failed for %s",
                        model_name,
                    )

            raise LLMUnavailable(
                " | ".join(errors)
            )

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )

            _LLM_FAILURE = (
                signature,
                time.monotonic(),
                message,
            )

            raise LLMUnavailable(message) from exc


# ---------------------------------------------------------------------
# Grounded narrative
# ---------------------------------------------------------------------

def _other_class(cls):
    s = settings()

    if cls == s["CLASS_ATTACK"]:
        return s["CLASS_FAULT"]

    return s["CLASS_ATTACK"]


def template_sections(packet):
    cls = packet["predicted_class"]
    other = _other_class(cls)
    probabilities = packet["probabilities_pct"]

    features = packet.get("top_features", [])
    low_reliability = bool(
        packet.get("low_reliability_class")
    )

    qualifier = (
        ", a low-reliability class,"
        if low_reliability
        else ""
    )

    sections = OrderedDict()

    sections["Summary"] = (
        f"The classifier predicts {cls}{qualifier} "
        "with model probability "
        f"{_guard_number(probabilities.get(cls))}%. "
        "The Non-Normal Score "
        "(one minus the Normal probability) is "
        f"{_guard_number(packet['risk_score_pct'])}%, "
        "not a calibrated probability of damage or attack."
    )

    evidence = "; ".join(
        (
            f"{item['feature']} "
            f"(z = {_guard_number(item['z'])})"
        )
        for item in features[:3]
    )

    if not evidence:
        evidence = "no usable baseline deviations"

    if packet.get("evidence_weak") or not features:
        weakness = (
            " Evidence is weak/uncertain under the "
            "configured baseline-deviation rule."
        )
    else:
        weakness = (
            " Baseline deviations alone do not "
            "identify the cause."
        )

    sections["Interpretation"] = (
        "The largest listed deviations from the "
        f"{packet.get('baseline_class', 'Normal')} "
        f"baseline are {evidence}. "
        f"The classifier assigns {cls} "
        f"{_guard_number(probabilities.get(cls))}% "
        f"and {other} "
        f"{_guard_number(probabilities.get(other))}%. "
        "This preference is not independent evidence "
        "that excludes the alternative class. "
        "The packet does not establish whether the cause "
        "is malicious manipulation or a physical disturbance."
        + weakness
    )

    sections["Possible Impact"] = (
        "The represented measurements warrant review. "
        "Their operational significance cannot be determined "
        "from this packet alone; equipment damage, "
        "service interruption and security-control bypass "
        "are not established."
    )

    sections["Recommended Actions"] = (
        "1. Have a qualified engineer review the listed "
        "measurements and preprocessing.\n"
        "2. Compare with independent measurements, if "
        "available; their availability is not established here.\n"
        "3. Review class-specific evaluation and baseline "
        "provenance before relying on this result.\n"
        "4. Do not initiate automated operational actions "
        "from this report."
    )

    reliability = packet.get(
        "model_reliability",
        {},
    )

    per_class = reliability.get("per_class", {})
    class_metrics = per_class.get(cls, {})

    precision = class_metrics.get("precision_pct")
    recall = class_metrics.get("recall_pct")

    if precision is not None and recall is not None:
        reliability_text = (
            f"Historical {cls} precision is "
            f"{_guard_number(precision)}% and recall is "
            f"{_guard_number(recall)}%. "
            "These aggregate metrics do not certify this "
            "individual prediction; the evaluation split's "
            "provenance must be checked."
        )
    else:
        reliability_text = (
            f"Historical precision/recall for {cls} "
            "are unavailable in the packet."
        )

    if low_reliability:
        reliability_text += (
            " This is a low-reliability class under "
            "the configured historical recall rule."
        )

    if packet.get("evidence_weak") or not features:
        strength = "Evidence is weak/uncertain. "
    else:
        strength = (
            "Deviation magnitude is not proof of an attack "
            "or prediction correctness. "
        )

    sections["Confidence & Limitations"] = (
        strength + reliability_text
    )

    return sections


def _escape_cell(value):
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def key_evidence_table(packet):
    lines = [
        "| Feature | Meaning | Value | Baseline | Deviation |",
        "|---|---|---|---|---|",
    ]

    for item in packet["top_features"][
        :settings()["TABLE_ROWS"]
    ]:
        capped = (
            " (capped)"
            if item.get("z_capped")
            else ""
        )

        lines.append(
            f"| {_escape_cell(item['feature'])} "
            f"| {_escape_cell(item['meaning'])} "
            f"| {item['value']} "
            f"| {item['baseline_mean']} "
            f"| {item['deviation']:+g} "
            f"({item['z']:+.2f} sigma{capped}) |"
        )

    if not packet["top_features"]:
        lines.append(
            "| Unavailable | No usable baseline evidence "
            "| - | - | - |"
        )

    return "\n".join(lines)


def parse_sections(text):
    pattern = re.compile(
        r"^\s{0,3}#{1,4}\s*"
        r"(?:\d+[.)]\s*)?"
        r"(.+?)\s*#*\s*$",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(text or ""))
    sections = OrderedDict()

    for index, match in enumerate(matches):
        title = (
            match.group(1)
            .replace("**", "")
            .strip()
            .lower()
        )

        canonical = next(
            (
                name
                for name in SECTION_ORDER
                if title.startswith(name.lower())
            ),
            None,
        )

        if canonical and canonical not in sections:
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            sections[canonical] = text[
                match.end():end
            ].strip()

    return sections


def verify_report(text, packet):
    """Conservative Guard V2 policy.

    The LLM draft is not used as authoritative final prose.
    No general semantic verification is claimed here.
    """
    notes = [
        "guard-v2: final narrative uses packet-grounded "
        "templates; the LLM draft is not relied upon"
    ]

    return template_sections(packet), notes


def assemble_markdown(
    packet,
    sections,
    notes,
    generator,
):
    head = [
        f"# Incident Report - {packet['predicted_class']}",
        "",
        f"- Record: {_escape_cell(packet['record_id'])}",
        f"- Model: {packet.get('model', 'unspecified')}",
        f"- Generated by: {generator}",
    ]

    if packet.get("synthetic"):
        head.append(
            "- SYNTHETIC RUN: not a confirmed operational event."
        )

    head.extend(
        f"- {note}"
        for note in notes
    )

    parts = ["\n".join(head), ""]

    for index, name in enumerate(
        SECTION_ORDER,
        start=1,
    ):
        content = (
            key_evidence_table(packet)
            if name == "Key Evidence"
            else sections[name]
        )

        parts.extend(
            [
                f"## {index}. {name}",
                content,
                "",
            ]
        )

    disclaimer = getattr(
        report_prompts,
        "DISCLAIMER",
        (
            "Decision support only; not an operational "
            "command or a confirmed incident diagnosis."
        ),
    )

    parts.append(f"_{disclaimer}_")

    return "\n".join(parts)


# ---------------------------------------------------------------------
# Public report entry point
# ---------------------------------------------------------------------

def generate_report(
    record,
    prediction,
    llm=None,
    save=True,
    synthetic=False,
    baseline=None,
    reliability=None,
    dictionary=None,
    mode=None,
):
    s = settings()

    if prediction["predicted_class"] == s["CLASS_NORMAL"]:
        return {
            "status": "skipped_normal",
            "markdown": None,
            "charts": [],
            "notes": [],
            "llm_used": False,
            "raw_llm_text": "",
            "error": None,
        }

    selected_mode = str(
        mode or s["REPORT_MODE"]
    ).strip().lower()

    # Injected test runners explicitly request the guarded draft path.
    if llm is not None and mode is None:
        selected_mode = "llm_draft_guarded"

    allowed_modes = {
        "grounded_template",
        "llm_draft_guarded",
    }

    if selected_mode not in allowed_modes:
        raise ValueError(
            "REPORT_MODE must be "
            "'grounded_template' or 'llm_draft_guarded'."
        )

    started = time.monotonic()

    # Baseline and input errors are not disguised as LLM failures.
    packet = build_evidence(
        record=record,
        prediction=prediction,
        baseline=baseline,
        reliability=reliability,
        dictionary=dictionary,
        synthetic=synthetic,
    )

    notes = list(packet.get("evidence_notes", []))
    raw_llm_text = ""
    error = None
    llm_used = False

    if selected_mode == "grounded_template":
        sections = template_sections(packet)

        status = "grounded_template"
        generator = (
            "packet-grounded template "
            "(LLM not invoked)"
        )

        notes.append(
            "Intentional template mode: no LLM loading "
            "or generation was requested."
        )

    else:
        try:
            runner = (
                llm
                if llm is not None
                else load_llm()
            )

            messages = report_prompts.build_messages(packet)
            raw_llm_text = runner(messages)

            if (
                not isinstance(raw_llm_text, str)
                or not raw_llm_text.strip()
            ):
                raise LLMUnavailable(
                    "The LLM returned no usable draft."
                )

            llm_used = True

            sections, guard_notes = verify_report(
                raw_llm_text,
                packet,
            )

            notes.extend(guard_notes)

            status = "llm"

            generator = (
                "packet-grounded final narrative; "
                "diagnostic LLM draft: "
                f"{getattr(runner, 'model_name', 'custom')}"
            )

        except Exception as exc:
            _logger().exception(
                "LLM failed; using grounded fallback"
            )

            error = {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            }

            sections = template_sections(packet)

            status = "template_fallback"
            generator = (
                "packet-grounded fallback "
                "(LLM failed)"
            )

            notes.append(
                f"LLM fallback: {error['type']}. "
                "See returned error details and logs/report.log."
            )

    lang_warning = s["REPORT_LANG"] != "en"

    if lang_warning:
        notes.append(
            "Language note: this module's grounded narrative "
            "is generated in English."
        )

    charts = []
    markdown_path = None
    draft_path = None

    if save:
        reports_dir = s["REPORTS_DIR"]
        reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        record_digest = hashlib.sha256(
            packet["record_id"].encode("utf-8")
        ).hexdigest()[:10]

        base_name = (
            f"report_{packet['model']}_"
            f"{record_digest}_"
            f"{uuid4().hex[:10]}"
        )

        try:
            probability_path = (
                reports_dir
                / f"{base_name}_probabilities.png"
            )

            generated = report_charts.probability_chart(
                packet,
                probability_path,
            )

            if generated is not None:
                charts.append(str(generated))

        except Exception as exc:
            _logger().exception(
                "Probability chart generation failed"
            )

            notes.append(
                "Probability chart unavailable: "
                f"{type(exc).__name__}."
            )

        if packet["top_features"]:
            try:
                features_path = (
                    reports_dir
                    / f"{base_name}_features.png"
                )

                generated = report_charts.feature_chart(
                    packet,
                    features_path,
                    top_n=s["TABLE_ROWS"],
                )

                if generated is not None:
                    charts.append(str(generated))

            except Exception as exc:
                _logger().exception(
                    "Feature chart generation failed"
                )

                notes.append(
                    "Feature chart unavailable: "
                    f"{type(exc).__name__}."
                )

        if raw_llm_text:
            draft_path = (
                reports_dir
                / f"{base_name}_llm_draft.txt"
            )

            draft_path.write_text(
                (
                    "UNVERIFIED LLM DRAFT\n"
                    "Not the authoritative incident report.\n\n"
                    + raw_llm_text
                ),
                encoding="utf-8",
            )

    markdown = assemble_markdown(
        packet=packet,
        sections=sections,
        notes=notes,
        generator=generator,
    )

    if save:
        markdown_path = (
            s["REPORTS_DIR"]
            / f"{base_name}.md"
        )

        markdown_path.write_text(
            markdown,
            encoding="utf-8",
        )

    duration = round(
        time.monotonic() - started,
        3,
    )

    _logger().info(
        (
            "report record=%s model=%s mode=%s "
            "status=%s llm_used=%s elapsed=%s"
        ),
        packet["record_id"],
        packet["model"],
        selected_mode,
        status,
        llm_used,
        duration,
    )

    return {
        "status": status,
        "mode": selected_mode,
        "markdown": markdown,
        "packet": packet,
        "charts": charts,
        "notes": notes,
        "md_path": (
            str(markdown_path)
            if markdown_path
            else None
        ),
        "draft_path": (
            str(draft_path)
            if draft_path
            else None
        ),
        "lang_warning": lang_warning,
        "synthetic": bool(synthetic),
        "raw_llm_text": raw_llm_text,
        "llm_used": llm_used,
        "final_narrative_source": (
            "packet_grounded_template"
        ),
        "error": error,
        "duration_seconds": duration,
        "generator": generator,
    }