"""Runs the three 'Demonstration Checks' from the manual headlessly with Streamlit's AppTest.
Run from this folder:  python test_app.py   (writes lab_outputs/demo_checks.json)
"""
import json

from streamlit.testing.v1 import AppTest

results = {}

# check 1: empty topic asks for input
at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
at.button[0].click().run()
results["empty_topic_warning"] = [w.value for w in at.warning]
results["empty_topic_generated_text"] = [m.value for m in at.markdown]

# check 2: two held-out topics produce a tip
for topic in ("recursion", "SQL joins"):
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.text_input[0].set_value(topic)
    at.button[0].click().run()
    assert not at.exception, at.exception
    results[topic] = [m.value for m in at.markdown]

# whitespace-only input behaves like empty input
at = AppTest.from_file("app.py", default_timeout=120).run()
at.text_input[0].set_value("   ")
at.button[0].click().run()
results["whitespace_only_warning"] = [w.value for w in at.warning]

json.dump(results, open("lab_outputs/demo_checks.json", "w"), indent=2)
print(json.dumps(results, indent=2))
