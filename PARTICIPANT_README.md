# MathWorks BCI Challenge -- Official Test Set

## Overview

This test set contains **2,000 unlabeled EEG motor imagery trials**. Classify each trial into one of four classes using your trained model.

**Classes:** `left_hand`, `right_hand`, `feet`, `idle`

---

## Data Format

Each `.npz` file contains exactly two arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `signal` | `(2500, channels)` | Raw EEG, float64 |
| `sampling_rate` | scalar | Always 500 Hz |

- **Epoch:** 5.0 seconds (2500 samples at 500 Hz)
- **Channels:** Either **29** or **46** per trial. Your model must handle both.
- **No preprocessing applied.** Signals are raw.
- **No metadata.** Files contain only signal + sampling rate. No labels, no subject info.

### Loading

```python
import numpy as np

data = np.load("test_data/test_00001.npz")
signal = data["signal"]            # (2500, 29) or (2500, 46)
sr = int(data["sampling_rate"])    # 500
```

MATLAB:
```matlab
data = py.numpy.load('test_data/test_00001.npz');
signal = double(data{'signal'});
```

No `allow_pickle=True` needed.

---

## Submission

### The only hard requirement: `predictions.csv`

Everything else is flexible. Submit whatever you have -- complete or incomplete, polished or rough. We want to see your work. An incomplete submission is infinitely better than no submission.

---

### Step 1: Create your Google Drive folder

Name it: `TeamName_MathWorksBCI` (no spaces)

```
TeamName_MathWorksBCI/
│
├── Folder_A_Software/
│   ├── predictions.csv             # ⬅ REQUIRED — your test set predictions
│   ├── accuracy_report.csv         # ⬅ REQUIRED — your validation results
│   ├── SUMMARY.md                  # Brief technical summary (1 page)
│   └── inference_video.mp4         # or .txt with YouTube/Drive link
│
├── Folder_B_Hardware/              # Only if you deployed on edge
│   ├── edge_metrics.csv            # Latency, throughput, model size
│   ├── edge_video.mp4              # or .txt with link
│   └── EDGE_README.md              # Device specs + what you deployed
│
└── README.md                       # Team overview
```

**Don't have everything?** That's okay. At minimum, submit `predictions.csv` and `README.md`. Include whatever else you have ready.

**Note:** Shortlisted teams will be asked to submit their full codebase, model weights, and reproducibility scripts in a later phase. For now, just share your results.

### Step 2: Share your folder

1. Right-click folder in Google Drive → **Share** → **Get link**
2. Set to **"Anyone with the link"** → **Viewer**
3. Copy the link

### Step 3: Submit the Google Form

Form link: *(will be shared by organizers)*

The form asks for:
- Team name
- Team members (names + affiliations)
- Contact email
- Your Google Drive folder link

---

## File Format Specifications

### `predictions.csv` -- THIS IS THE ONE THAT MATTERS

```csv
test_id,predicted_label
test_00001,left_hand
test_00002,right_hand
test_00003,feet
test_00004,idle
...
test_02000,left_hand
```

| Rule | Detail |
|------|--------|
| Header | Exactly `test_id,predicted_label` |
| Rows | Exactly 2,000 |
| test_id | `test_00001` through `test_02000` |
| Labels | `left_hand`, `right_hand`, `feet`, or `idle` |
| Format | Lowercase, underscore-separated, no extra whitespace |
| Encoding | UTF-8 |

**Validate before submitting:**

```python
import csv

VALID_LABELS = {"left_hand", "right_hand", "feet", "idle"}
EXPECTED_IDS = {f"test_{i:05d}" for i in range(1, 2001)}

with open("predictions.csv", "r") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ["test_id", "predicted_label"], \
        f"Wrong headers: {reader.fieldnames}"
    seen = set()
    for row in reader:
        tid, label = row["test_id"], row["predicted_label"]
        assert tid in EXPECTED_IDS, f"Unknown test_id: {tid}"
        assert tid not in seen, f"Duplicate: {tid}"
        assert label in VALID_LABELS, f"Invalid label '{label}' for {tid}"
        seen.add(tid)
    assert len(seen) == 2000, f"Only {len(seen)} predictions, expected 2000"

print("Valid!")
```

If this script prints "Valid!", your predictions file is good to go.

---

### `accuracy_report.csv` -- Your validation results

```csv
metric,value
overall_accuracy,67.50
evaluation_method,5-fold cross-validation
left_hand_accuracy,72.00
right_hand_accuracy,65.00
feet_accuracy,70.00
idle_accuracy,63.00
```

At minimum include `overall_accuracy` and `evaluation_method`. Per-class rows are appreciated but not mandatory.

---

### `SUMMARY.md` -- Your approach (1 page)

Briefly cover: preprocessing, features, model architecture, training strategy, how you handled 29 vs 46 channels, and edge deployment (if applicable). Keep it short.

---

### `inference_video.mp4` -- Show it running

A screen recording (max 5 minutes) showing your pipeline running inference on the test set. Should show the command being run, data loading, and predictions.csv being generated.

If you deployed on edge, show it running on the edge device.

Can be an `.mp4`/`.mov` file in the folder, or a `.txt` file containing a YouTube/Google Drive link (set to "Anyone with the link can view").

---

### `README.md` -- Team overview (root folder)

```markdown
# Team: [Your Team Name]

## Members
- [Name] -- [College/University] -- [Role]
- [Name] -- [College/University] -- [Role]

## Contact
- Email: [team contact email]

## Approach (2-3 sentences)
[What preprocessing, what model, how you handled channel differences]

## Reported Testing Accuracy
- Overall: XX.XX%
- Method: [e.g., 5-fold cross-validation]

## Tools Used
- Language: [Python / MATLAB / both]
- Key libraries: [e.g., PyTorch, MNE, scikit-learn]

## Edge Deployment (if applicable)
- Device: [e.g., Raspberry Pi 4]
- Latency: [e.g., 42 ms/trial]
- Model size: [e.g., 2.4 MB]
```

---

### Edge Deployment Files (Folder_B_Hardware/) -- Only if applicable

If you deployed on an edge device, include:

- **edge_metrics.csv** -- Device name, latency per trial, throughput, model size, quantization method, framework
- **edge_video.mp4** -- Video showing the physical device classifying trials with latency on screen
- **EDGE_README.md** -- What device you used, what framework, what you achieved

---

## Quick Checklist

**Must have:**
- [ ] `predictions.csv` with 2,000 valid predictions
- [ ] `accuracy_report.csv` with your testing accuracy
- [ ] `README.md` with team overview

**Should have:**
- [ ] `SUMMARY.md` (1 page)
- [ ] `inference_video.mp4` or link
- [ ] Google Drive folder shared as "Anyone with the link can view"

**If you deployed on edge:**
- [ ] `Folder_B_Hardware/` with edge metrics, video, and readme

---

## Scoring

| Criterion | Description |
|-----------|-------------|
| Classification accuracy | Correct predictions out of 2,000 trials |
| Cohen's kappa | Accounts for chance agreement |
| Per-class F1 | Balanced performance across all four classes |
| Approach quality | Methodology, generalization strategy |
| Edge deployment | Latency, model size, quantization, on-device demo |
| Documentation | Clarity and reproducibility |

Random chance baseline: 25.0% (4 classes).

Shortlisted teams will be invited for a final presentation to the jury.

---

## Rules

1. **No reverse-mapping.** Do not attempt to identify which trial came from which subject/session to recover labels. Instant disqualification.
2. **No manual labeling.** All predictions must come from your automated pipeline.
3. **Single submission.** One submission per team. Finalize before submitting.
4. **Independent work.** Do not share predictions or models with other teams.
5. **Reproducibility.** Top teams' code must reproduce their submitted predictions.
6. **Dataset policy.** You may train on the public dataset (Qing Zhou, IEEE DataPort, DOI: 10.21227/f1c7-7x89). External data must be disclosed.

---

## FAQ

**Q: Why do some trials have 29 channels and others 46?**
A: Two different EEG cap configurations were used in the source dataset. Your model must handle both.

**Q: Are the signals preprocessed?**
A: No. Raw signals, no filtering applied.

**Q: Can I use pretrained models or transfer learning?**
A: Yes, but disclose it in your SUMMARY.md.

**Q: What if my model can't handle both channel counts?**
A: It must. A model that crashes on one configuration will score 0 on those trials.

**Q: Is the class distribution balanced?**
A: We do not disclose the class distribution. Design accordingly.

**Q: My submission is incomplete -- should I still submit?**
A: Yes. Submit whatever you have. An incomplete submission is better than no submission.

---

## Deadline

**September 6, 2026 (Sunday)**

Do not modify your Google Drive folder after the deadline. File timestamps are verified.

---

## Contact

For queries: **bangalore-sps@ieee.org**

Read this document fully before reaching out.
