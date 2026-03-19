import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("=== Loading SmolVLA ===")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda")
policy.eval()

print("\n=== Input format (from config) ===")
for name, feat in policy.config.input_features.items():
    print(f"  {name}: type={feat.type}, shape={feat.shape}")

print("\n=== Output format (from config) ===")
for name, feat in policy.config.output_features.items():
    print(f"  {name}: type={feat.type}, shape={feat.shape}")
print(f"  chunk_size (action steps per inference): {policy.config.chunk_size}")

# Get the tokenizer from the internal VLM
tokenizer = policy.model.vlm_with_expert.processor.tokenizer
print(f"\n=== Tokenizer: {tokenizer.__class__.__name__} ===")

# Tokenize the task
task_text = "pick up the red block"
max_length = policy.config.tokenizer_max_length
tokenized = tokenizer(
    [task_text],
    padding="max_length",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
)

print(f"\n=== Building dummy input batch ===")
batch = {
    "observation.state": torch.randn(1, 6).to("cuda"),
    "observation.images.camera1": torch.randn(1, 3, 256, 256).to("cuda"),
    "observation.images.camera2": torch.randn(1, 3, 256, 256).to("cuda"),
    "observation.images.camera3": torch.randn(1, 3, 256, 256).to("cuda"),
    "observation.language.tokens": tokenized["input_ids"].to("cuda"),
    "observation.language.attention_mask": tokenized["attention_mask"].bool().to("cuda"),
}
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n=== Running inference ===")
with torch.no_grad():
    action = policy.select_action(batch)

print(f"\n=== Output ===")
print(f"  action shape: {action.shape}")
print(f"  action dtype: {action.dtype}")
print(f"  action device: {action.device}")
print(f"  action[0] (first step): {action[0].cpu().numpy()}")
print(f"  action range: [{action.min().item():.4f}, {action.max().item():.4f}]")

print(f"\n=== SUMMARY ===")
print(f"  INPUT:")
print(f"    - observation.state: [1, 6] (6 joint positions)")
print(f"    - observation.images.camera*: [1, 3, 256, 256] (up to 3 RGB cameras)")
print(f"    - observation.language.tokens: [1, {max_length}] (tokenized task instruction)")
print(f"    - observation.language.attention_mask: [1, {max_length}]")
print(f"  OUTPUT:")
print(f"    - action: [{action.shape[0]}, {action.shape[1]}] = {action.shape[0]} time steps x {action.shape[1]} DOF")
print(f"    - Each value = target joint position (normalized)")
print(f"\n=== SO-ARM100 Joint Mapping ===")
print(f"  action[t][0] = shoulder_pan   (joint 1)")
print(f"  action[t][1] = shoulder_lift  (joint 2)")
print(f"  action[t][2] = elbow_flex     (joint 3)")
print(f"  action[t][3] = wrist_flex     (joint 4)")
print(f"  action[t][4] = wrist_roll     (joint 5)")
print(f"  action[t][5] = gripper        (joint 6)")
