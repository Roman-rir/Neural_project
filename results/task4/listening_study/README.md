# Five-listener evaluation — awaiting real ratings

Give each of five different people their assigned L01–L05 HTML file and the accompanying audio folder. Open the file in a browser, consent, listen, rate every clip 1–5, then download the ratings JSON. Keep coordinator_key.json and the qualitative examples hidden until everyone finishes. Distribute only locally permitted audio; this package has not been published or sent to anyone.

Store the five returned files in a ratings folder and run:

```powershell
python -m src.task4.listening summarize --output-dir results/task4/listening_study --ratings-dir PATH_TO_RETURNED_JSON_FILES
```

Use five distinct humans, one assigned code each; do not let one person complete multiple codes. The coordinator must confirm this externally—files alone cannot prove human identity. No real ratings exist yet. Empty/missing/incomplete responses are not results.
