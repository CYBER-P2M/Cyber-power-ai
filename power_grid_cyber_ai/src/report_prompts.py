"""Prompts for the incident report. English only (small LLM)."""

SYSTEM_ROLE = (
    "You are a power-grid cybersecurity and protection analyst. "
    "Use ONLY the data provided in the evidence packet. Never invent numbers, devices, "
    "events, logs, alerts, commands or measurements that are not in the packet. "
    "Copy every number exactly as written in the packet. "
    "If the evidence is weak, say it is uncertain. Be concise and follow the fixed format."
)

DISCLAIMER = "Decision-support only; requires review by a qualified engineer."

FORMAT_RULES = (
    "Write ONLY these five Markdown sections, with exactly these headings, in this order:\n"
    "## 1. Summary\n(2 sentences: the detection and the risk score.)\n"
    "## 3. Interpretation\n(short paragraph)\n"
    "## 4. Possible Impact\n(short paragraph)\n"
    "## 5. Recommended Actions\n(3 to 5 numbered items)\n"
    "## 6. Confidence & Limitations\n(1-2 lines)\n"
    "Do not write section 2 (the evidence table is added by code). "
    "Do not add any other text after section 6."
)

ATTACK_PROMPT = (
    "The model predicts ATTACK. Explain which signs in the provided features are consistent with "
    "malicious manipulation of measurements (for example data injection), name the provided features "
    "that show it, the possible impact, and the response. Also say why the evidence does NOT fit "
    "the Fault class."
)

FAULT_PROMPT = (
    "The model predicts FAULT. Explain which signs in the provided features are consistent with a "
    "physical/electrical disturbance, the affected measurements, the likely cause, and the response. "
    "Also say why the evidence does NOT fit the Attack class."
)


def packet_to_text(packet, max_features=10):
    p = packet
    strength = (f"WEAK (all top-5 |z| < {p['weak_z_threshold']:g})" if p["evidence_weak"] else "not weak")
    lines = [
        f"Predicted class: {p['predicted_class']}",
        "Class probabilities (percent): " + ", ".join(f"{k} {v}" for k, v in p["probabilities_pct"].items()),
        f"Risk score (percent): {p['risk_score_pct']}",
        f"Evidence strength: {strength}",
        "Top deviations from the Normal baseline (feature | meaning | value | baseline mean | z):",
    ]
    for f in p["top_features"][:max_features]:
        lines.append(f"- {f['feature']} | {f['meaning']} | {f['value']} | {f['baseline_mean']} | z={f['z']}")
    rel = p["model_reliability"]
    lines.append(f"Historical reliability of the model ({rel['model']}), per class:")
    for cls, m in rel["per_class"].items():
        lines.append(f"- {cls}: precision {m['precision_pct']} percent, recall {m['recall_pct']} percent, support {m['support']}")
    if p["low_reliability_class"]:
        lines.append(f"NOTE: {p['predicted_class']} is a low-reliability class (recall below {p['low_recall_threshold_pct']:g} percent). "
                     "The Summary must say 'low-reliability class' and the Confidence section must quote the historical precision and recall.")
    return "\n".join(lines)


def build_messages(packet):
    task = ATTACK_PROMPT if packet["predicted_class"] == "Attack" else FAULT_PROMPT
    user = f"{task}\n\n{FORMAT_RULES}\n\nEVIDENCE PACKET\n{packet_to_text(packet)}"
    return [{"role": "system", "content": SYSTEM_ROLE}, {"role": "user", "content": user}]
