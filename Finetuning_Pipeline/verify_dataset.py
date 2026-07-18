# verify_dataset.py
"""
Verify that findings + impression are combined correctly.
"""
from datasets import load_dataset

ds = load_dataset("itsanmolgupta/mimic-cxr-dataset", trust_remote_code=True)
split = list(ds.keys())[0]
print(f"Split: {split}")
print(f"Columns: {ds[split].column_names}")
print(f"Size: {len(ds[split])}")

print("\n" + "=" * 70)
for i in range(3):
    sample = ds[split][i]
    print(f"\n--- Sample {i} ---")

    findings = sample.get("findings", None)
    impression = sample.get("impression", None)

    print(f"  findings:   {str(findings)[:200] if findings else '<EMPTY>'}")
    print(f"  impression: {str(impression)[:200] if impression else '<EMPTY>'}")

    # Simulate combination
    parts = []
    if findings and str(findings).strip() and str(findings).lower() not in ("nan", "none"):
        parts.append(f"FINDINGS: {str(findings).strip()}")
    if impression and str(impression).strip() and str(impression).lower() not in ("nan", "none"):
        parts.append(f"IMPRESSION: {str(impression).strip()}")

    combined = " ".join(parts) if parts else "Chest radiograph. No report available."
    print(f"  COMBINED:   {combined[:300]}")

print("\n" + "=" * 70)

# Count empty values
n_empty_findings = 0
n_empty_impression = 0
n_both_empty = 0
check_n = min(1000, len(ds[split]))

for i in range(check_n):
    sample = ds[split][i]
    f = sample.get("findings", None)
    imp = sample.get("impression", None)

    f_empty = (not f) or str(f).strip() == "" or str(f).lower() in ("nan", "none")
    i_empty = (not imp) or str(imp).strip() == "" or str(imp).lower() in ("nan", "none")

    if f_empty:
        n_empty_findings += 1
    if i_empty:
        n_empty_impression += 1
    if f_empty and i_empty:
        n_both_empty += 1

print(f"\nOut of first {check_n} samples:")
print(f"  Empty findings:   {n_empty_findings} ({n_empty_findings/check_n*100:.1f}%)")
print(f"  Empty impression: {n_empty_impression} ({n_empty_impression/check_n*100:.1f}%)")
print(f"  Both empty:       {n_both_empty} ({n_both_empty/check_n*100:.1f}%)")