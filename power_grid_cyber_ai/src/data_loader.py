from pathlib import Path
import re

import pandas as pd


# أسماء أعمدة الليبل الممكنة، مرتبة حسب الأولوية.
LABEL_CANDIDATES = [
    "label",
    "class",
    "target",
    "event",
    "category",
    "type",
    "marker",
]

# الفئات النهائية للمشروع.
FINAL_CLASSES = {"Normal", "Attack", "Fault"}


def _clean_column_name(column):
    """توحيد اسم العمود للمقارنة فقط."""
    return str(column).strip().lower()


def _compact_text(value):
    """تحويل القيمة إلى صيغة بسيطة للمقارنة."""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    compact = re.sub(r"[^a-z0-9]", "", text)
    return text, compact


def find_target(df, requested=None):
    """
    يحدد عمود الليبل داخل DataFrame واحد.

    إذا مرّرت requested، يجب أن يوجد الاسم في الملف نفسه.
    وإلا يبحث ضمن LABEL_CANDIDATES بدون حساسية لحالة الأحرف.
    """
    columns_lookup = {
        _clean_column_name(column): column
        for column in df.columns
    }

    if requested:
        requested_key = _clean_column_name(requested)

        if requested_key in columns_lookup:
            return columns_lookup[requested_key]

        raise ValueError(
            f"Requested target column '{requested}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    for candidate in LABEL_CANDIDATES:
        if candidate in columns_lookup:
            return columns_lookup[candidate]

    raise ValueError(
        "Target column not found. "
        f"Expected one of: {LABEL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )

def normalize_label(value):
    """توحيد قيم marker إلى Normal / Attack / Fault."""

    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)

    label_map = {
        # Normal
        "normal": "Normal",
        "noevents": "Normal",
        "noevent": "Normal",

        # Attack
        "attack": "Attack",
        "attacks": "Attack",
        "injection": "Attack",
        "intrusion": "Attack",

        # Fault
        "natural": "Fault",
        "naturalevent": "Fault",
        "naturalevents": "Fault",
        "fault": "Fault",
        "failure": "Fault",
        "disturbance": "Fault",
    }

    return label_map.get(compact)


def read_csvs(directory, requested_target=None, allow_numeric_marker=False):
    """
    يقرأ ملفات CSV ويجهز عمود label موحّد قبل الدمج.

    كل ملف يمكن أن يحتوي target مختلفًا، مثل:
    - class
    - marker
    - label

    يرجع:
    - DataFrame موحّد يحتوي على label.
    - قائمة بمسارات ملفات CSV.
    """
    paths = sorted(Path(directory).glob("*.csv"))

    if not paths:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []

    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]

        try:
            source_target = find_target(frame, requested=requested_target)
        except ValueError as error:
            raise ValueError(
                f"\nProblem in file: {path.name}\n{error}"
            ) from error

        # نحفظ معلومات المصدر للتدقيق فقط.
        frame["source_file"] = path.name
        frame["source_target"] = source_target

        # إنتاج الليبل النهائي الموحد.
        frame["label"] = frame[source_target].map(
            lambda value: normalize_label(
                value,
                allow_numeric_marker=allow_numeric_marker,
            )
        )

        # نحذف عمود الهدف الأصلي لتجنب استعماله كـ Feature.
        if source_target != "label":
            frame = frame.drop(columns=[source_target])

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    return combined, paths
def drop_leakage_columns(X):
    """
    حذف الأعمدة التي تحتوي على هوية العينة أو الليبل أو أي معلومة
    يمكن أن تكشف الإجابة للنموذج بشكل مباشر.
    """
    remove = []

    leakage_columns = {
        "id",
        "index",
        "unnamed0",
        "sourcefile",
        "sourcetarget",
        "scenario",
        "scenarioid",

        # أعمدة الليبل الأصلية والموحّدة:
        "marker",
        "class",
        "label",
        "target",
        "event",
        "category",
        "type",
    }

    for column in X.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())

        if normalized in leakage_columns:
            remove.append(column)

    return X.drop(columns=remove, errors="ignore"), remove