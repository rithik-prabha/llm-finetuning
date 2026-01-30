import json
import random
from pathlib import Path
from logger.log import log

INPUT_JSON = "data.sample_data"     
OUTPUT_DIR = "data"
TRAIN_RATIO = 0.85
SEED = 42

random.seed(SEED)

train_dir = Path(OUTPUT_DIR)/"train"
eval_dir = Path(OUTPUT_DIR)/"eval"
train_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok = True)


with open(INPUT_JSON, " r", encoding="utf-8")as f:
    data = json.load(f)

assert isinstance(data, list), "json root must be list"

random.shuffle(data)

split_idx = int(len(data)*TRAIN_RATIO)
train_data = data[:split_idx]
eval_data = data[split_idx:]

with open(train_dir, "w", encoding="utf-8") as f:
    json.dump(train_data, f , indent=2, ensure_ascii=False)
with open (eval_data, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False )


log.info(f"Total samples  : {len(data)}")
log.info(f"Train samples  : {len(train_data)}")
log.info(f"Eval samples   : {len(eval_data)}")
log.info("Split completed successfully")


