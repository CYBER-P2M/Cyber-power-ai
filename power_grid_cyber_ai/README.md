# PowerGrid Cyber AI

نسخة تشغيلية أكاديمية كاملة لتصنيف بيانات MSU/ORNL إلى Normal وFault وAttack. تعمل على بيانات تاريخية/محاكاة فقط، ولا تتصل بشبكة كهرباء حقيقية.

## التشغيل السريع

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# ضع CSV داخل data/raw/
python src/train_models.py --target marker
python src/evaluate_models.py
python src/explainability.py
streamlit run dashboard/app.py
```

إذا كان عمود التسمية لا يحمل اسماً معروفاً استخدم: `python src/train_models.py --target NAME`.

## تنزيل البيانات

استخدم `python tools/download_dataset.py --list` لعرض روابط المصدر ثم مرر الرابط المباشر عبر `--url`. لا يوجد رابط تخميني ثابت؛ تحقق من المصدر الرسمي قبل التنزيل.


C:\Users\LEO\Desktop\المشروع بدون الصوره\file\power_grid_cyber_ai\power_grid_cyber_ai

