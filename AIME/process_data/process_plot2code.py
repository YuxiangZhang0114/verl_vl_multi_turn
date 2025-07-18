from datasets import load_dataset
import io

ds = load_dataset("TencentARC/Plot2Code")

# print(ds)
# DatasetDict({
#     test: Dataset({
#         features: ['image', 'url', 'code', 'instruction'],
#         num_rows: 368
#     })
# })

header = """
```Python
# ============================================================
# 📌 Agent Interpreter Initialization Cell
#     🚫 No Network | 🚫 No Filesystem Writes | 🚫 No Interactive Input
# ============================================================

# ─── Core Libraries ──────────────────────────────────────────
import math
import random
import warnings
import datetime as dt
import statistics as stats
from contextlib import contextmanager
from time import perf_counter

# ─── Data Science & Numerical Computing ──────────────────────
import numpy as np
import pandas as pd

# ─── Visualization (Matplotlib & Seaborn) ────────────────────
import matplotlib.pyplot as plt
plt.style.use("ggplot")

try:
    import seaborn as sns
except ImportError:
    sns = None

# --- Display tweaks ------------------------------------------
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
warnings.filterwarnings("ignore", category=FutureWarning)

print("✅ Core libraries loaded (minimal setup)")


```
"""
def image_to_bytes(image):
    """Convert PIL image to bytes"""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def make_map_fn(split):
    def process_fn(example, idx):
        image = example.pop("image")
        url = example.pop("url")
        code = example.pop("code")
        instruction = example.pop("instruction")

        # Convert PIL image to bytes to ensure consistency with other datasets
        image_bytes = image_to_bytes(image)

        data = {
            "data_source": "plot2code",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in code generation. You will be given a scientific plot image and/or descriptive instructions. Please generate Python matplotlib code that can reproduce the plot based on this information."
                        "Your goal is to produce high-quality, executable code so that the rendered result matches the reference plot as closely as possible."
                        "Please strictly follow these requirements:\n"
                        "Necessary imports and data generation steps must be included in the interpreter. The header file is as follows:\n"
                        +header+
                        "\n\nYour output will be used to evaluate the scientific plotting code generation ability of multimodal large models. Please actively call tools to plot in multiple rounds, observe the returned results, and modify your code to improve generation quality."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{instruction}",
                },
            ],
            "images": [{"bytes": image_bytes}],
            "ability": "code_generation",
            "reward_model": {"ground_truth": code},
            "extra_info": {
                "split": split,
                "index": idx,
                "code": code,
                "instruction": instruction,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "code_interpreter": {
                        "create_kwargs": {"code": code},
                    },
                },
            },
        }
        return data

    return process_fn

# 只保留使用了matplotlib的测试集
test_dataset = ds["test"].filter(lambda x: "matplotlib" in x["code"])
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=1)
local_dir = "data/plot2code"
import os
test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
print(f"Test dataset saved to {local_dir}/test.parquet")