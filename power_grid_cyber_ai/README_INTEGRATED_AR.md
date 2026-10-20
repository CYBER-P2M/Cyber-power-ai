# PowerGrid Cyber AI - Integrated Package

هذا الإصدار يجمع قسم Substation Vision وتجهيز Dataset المحطات.

## التشغيل السريع

```bash
pip install -r requirements-vision.txt
python -m computer_vision.seed_registry
python tools/prepare_substation_dataset.py
python tools/validate_substation_dataset.py
streamlit run dashboard/app.py
```

ضع الداتاست المفكوكة في `data/vision/dataset_raw/` وانسخ `classes.txt` إلى `data/vision/classes.txt`.
بعد تدريب YOLO انسخ `best.pt` إلى `models/vision/substation_yolo.pt`.


Get-ChildItem -Path .\runs -Recurse -Filter best.pt | Select-Object FullName, Length, LastWriteTime | Format-List

Get-Item ".\models\vision\substation_yolo.pt" | Format-List FullName, Length