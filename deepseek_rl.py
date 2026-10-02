import random
import re
from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def build_toy_math_dataset(n=1000):
    problems = []
    for _ in range(n):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        question = f"What is {a} + {b}? Answer with just the integer."
        answer = str(a + b)
        problems.append({"question": question, "answer": answer})
    return Dataset.from_list(problems)


def load_policy(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    policy.eval()
    return tokenizer, policy


def generate_trajectory(policy, tokenizer, question: str, device, max_new_tokens: int = 128) -> str:
    prompt = question + "\nLet's think step by step.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = policy.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_final_answer(text: str) -> str:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def compute_reward(question: str, gt_answer: str, trajectory: str) -> float:
    pred = extract_final_answer(trajectory)
    correct = pred == gt_answer
    reward = 1.0 if correct else 0.0
    steps = trajectory.count("\n")
    reward += 0.1 * min(steps, 10)
    return reward


@dataclass
class TrajectoryBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    rewards: torch.Tensor


def collect_group_trajectories(tokenizer, policy, batch, device, group_size: int = 4) -> TrajectoryBatch:
    questions = batch["question"]
    answers = batch["answer"]

    all_input_ids = []
    all_attention_mask = []
    all_rewards = []

    for q, a in zip(questions, answers):
        group_trajs = []
        group_rewards = []

        for _ in range(group_size):
            traj = generate_trajectory(policy, tokenizer, q, device)
            full_text = q + "\n" + traj
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            reward = compute_reward(q, a, traj)
            group_trajs.append(enc)
            group_rewards.append(reward)

        input_ids = torch.cat([t["input_ids"] for t in group_trajs], dim=0)
        attention_mask = torch.cat([t["attention_mask"] for t in group_trajs], dim=0)
        rewards = torch.tensor(group_rewards, dtype=torch.float32)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_rewards.append(rewards)

    return TrajectoryBatch(
        input_ids=torch.cat(all_input_ids, dim=0),
        attention_mask=torch.cat(all_attention_mask, dim=0),
        rewards=torch.cat(all_rewards, dim=0),
    )


def grpo_loss(logits: torch.Tensor, batch: TrajectoryBatch, group_size: int = 4) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    seq_log_probs = log_probs.gather(-1, batch.input_ids.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = (seq_log_probs * batch.attention_mask).sum(dim=-1)

    rewards = batch.rewards
    seq_log_probs_groups = seq_log_probs.view(-1, group_size)
    rewards_groups = rewards.view(-1, group_size)

    baseline = rewards_groups.mean(dim=1, keepdim=True)
    advantages = rewards_groups - baseline
    advantages = advantages.view(-1)

    return -(seq_log_probs * advantages).mean()


def evaluate_model(policy, tokenizer, ds, device, n_samples: int = 100):
    correct = 0
    total = min(len(ds), n_samples)
    for i in range(total):
        item = ds[i]
        traj = generate_trajectory(policy, tokenizer, item["question"], device)
        pred = extract_final_answer(traj)
        if pred == item["answer"]:
            correct += 1
    acc = correct / total
    print(f"Accuracy: {acc:.3f}")
    return acc


def main():
    device = configure_device()
    base_model = "deepseek-ai/DeepSeek-V3"
    tokenizer, policy = load_policy(base_model)

    train_ds = build_toy_math_dataset(500)
    eval_ds = build_toy_math_dataset(100)

    optimizer = Adam(policy.parameters(), lr=1e-5)
    epochs = 3
    batch_size = 4
    group_size = 4

    policy.train()
    for epoch in range(epochs):
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idxs = indices[i:i + batch_size]
            if len(batch_idxs) < batch_size:
                continue

            batch = {"question": train_ds.select(batch_idxs)["question"], "answer": train_ds.select(batch_idxs)["answer"]}
            traj_batch = collect_group_trajectories(tokenizer, policy, batch, device, group_size=group_size)

            traj_batch.input_ids = traj_batch.input_ids.to(device)
            traj_batch.attention_mask = traj_batch.attention_mask.to(device)
            traj_batch.rewards = traj_batch.rewards.to(device)

            outputs = policy(
                input_ids=traj_batch.input_ids,
                attention_mask=traj_batch.attention_mask,
            )
            logits = outputs.logits
            loss = grpo_loss(logits, traj_batch, group_size=group_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} complete.")

    evaluate_model(policy, tokenizer, eval_ds, device)


if __name__ == "__main__":
    main()
