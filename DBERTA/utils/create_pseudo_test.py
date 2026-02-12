import json
import random

# Paths to your data files
train_path = 'data/train.jsonl'
dev_path = 'data/dev.jsonl'
pseudo_test_path = 'data/pseudo_test.jsonl'
new_train_path = 'data/new_train.jsonl'

# Percentage for pseudo-test set
pseudo_test_ratio = 0.2  # 20%

# Load data
examples = []
for path in [train_path, dev_path]:
    with open(path, 'r') as f:
        examples.extend([json.loads(line) for line in f])

# Shuffle and split
total = len(examples)
random.shuffle(examples)
split_idx = int(total * (1 - pseudo_test_ratio))
new_train = examples[:split_idx]
pseudo_test = examples[split_idx:]

# Write new files
with open(new_train_path, 'w') as f:
    for ex in new_train:
        f.write(json.dumps(ex) + '\n')

with open(pseudo_test_path, 'w') as f:
    for ex in pseudo_test:
        f.write(json.dumps(ex) + '\n')

print(f"New train set: {len(new_train)} examples\nPseudo-test set: {len(pseudo_test)} examples")
