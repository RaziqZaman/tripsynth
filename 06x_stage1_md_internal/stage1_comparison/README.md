# Stage 1 Comparison Bundle

This folder contains the current synthetic-vs-real Stage 1 comparison outputs.

Important: both MATSim runs exited with `java_heap_out_of_memory`, so the event files are empty and count-validation performance against observed AADT is not available yet. The CSVs and charts here therefore compare run status, observed count matching, and weighted real-vs-synthetic demand distributions.

After rerunning MATSim successfully and running `./06_postprocess_matsim.sh`, rerun:

```bash
.venv/bin/python 06_visualize-stage1-results.py
```

The script will then add assigned-vs-observed count validation charts automatically.
