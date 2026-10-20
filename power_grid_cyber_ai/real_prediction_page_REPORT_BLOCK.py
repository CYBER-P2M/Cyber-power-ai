
# ======================================================================
# PASTE AT THE VERY END of dashboard/pages/real_prediction_page.py
# (after the final st.warning(...)).  Uses existing variables:
# ROOT, uploaded, row_number, feature_order, X_ready, predicted, probabilities, risk_score
# ======================================================================
import sys as _sys

if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from src import llm_report as _llm_report  # noqa: E402

st.divider()
st.subheader("التقرير التفسيري")

if predicted == "Normal":
    st.info("التنبؤ Normal: لا حاجة لتقرير تفسيري.")
else:
    _cache = st.session_state.setdefault("llm_reports", {})
    _key = (uploaded.name, int(row_number))

    if st.button("إنشاء التقرير"):
        _record = {name: float(value) for name, value in zip(feature_order, X_ready[0])}
        _prediction = {
            "predicted_class": predicted,
            "probabilities": probabilities,
            "risk_score": risk_score,
            "index": int(row_number),
            "source": uploaded.name,
        }
        with st.spinner("جاري إنشاء التقرير... (أول مرة قد يتم تحميل النموذج اللغوي)"):
            try:
                _cache[_key] = _llm_report.generate_report(_record, _prediction)
            except _llm_report.BaselineMissing as exc:
                st.error(f"ملف baseline غير موجود: {exc}")
                st.code("python tools\\build_baseline.py")
            except Exception as exc:  # the report must never break the page
                st.error(f"تعذر إنشاء التقرير: {exc}")

    _res = _cache.get(_key)
    if _res:
        if _res.get("lang_warning"):
            st.warning("التقرير يُنشأ بالإنجليزية لأن النموذج اللغوي الصغير ضعيف بالعربية.")
        st.markdown(_res["markdown"])
        _c1, _c2 = st.columns(2)
        _c1.image(_res["charts"][0])
        _c2.image(_res["charts"][1])
        st.download_button(
            "Download report (.md)",
            data=_res["markdown"].encode("utf-8"),
            file_name=f"report_{int(row_number)}_{predicted}.md",
            mime="text/markdown",
        )
