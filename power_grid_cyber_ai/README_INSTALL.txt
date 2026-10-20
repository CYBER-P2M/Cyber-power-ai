1. Copy src/, tools/, logs/, results/reports/ into your project (do not overwrite your own files).
2. Append src/config_additions.py to the END of your src/config.py (check for name clashes first).
3. Paste dashboard/pages/real_prediction_page_REPORT_BLOCK.py at the END of dashboard/pages/real_prediction_page.py.
4. Add requirements_additions.txt lines to requirements.txt, then: pip install -r requirements.txt
5. python tools/build_baseline.py
6. python tools/test_report.py
7. python -m streamlit run dashboard/app.py
