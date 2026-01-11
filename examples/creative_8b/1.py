import wandb

api = wandb.Api()

# 替换成你的两个 run ID
run1 = api.run("mzfslhm-ustc/verl_grpo_example_CW_2/m7cecd8o")
run2 = api.run("mzfslhm-ustc/verl_grpo_example_CW_2/x0hp8268")

config1 = run1.config
config2 = run2.config

all_keys = set(config1.keys()) | set(config2.keys())

print("=== 参数差异 ===")
for key in sorted(all_keys):
    val1 = config1.get(key, "N/A")
    val2 = config2.get(key, "N/A")
    if val1 != val2:
        print(f"\n{key}:")
        print(f"  run1: {val1}")
        print(f"  run2: {val2}")