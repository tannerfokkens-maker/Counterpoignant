# save as fix_fugue_labels.py
import json

with open("data/sequences.json") as f:
    sequences = json.load(f)
with open("data/piece_ids.json") as f:
    piece_ids = json.load(f)

FORM_CHORALE = 24
FORM_FUGUE = 26

fixed = 0
for i, pid in enumerate(piece_ids):
    pl = pid.lower()
    if "wtc1f" in pl or "wtc2f" in pl:
        seq = sequences[i]
        # Form token is in the prefix (first ~7 tokens)
        # Find and replace FORM_CHORALE with FORM_FUGUE
        for j in range(min(10, len(seq))):
            if seq[j] == FORM_CHORALE:
                seq[j] = FORM_FUGUE
                fixed += 1
                break

print(f"Fixed {fixed} sequences")

with open("data/sequences.json", "w") as f:
    json.dump(sequences, f)