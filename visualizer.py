import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv("results_320.csv", sep=",")

'''
# Convert goal_reached to int if needed
df["goal_reached"] = df["goal_reached"].astype(int)

# Make output directory
os.makedirs("plots", exist_ok=True)

print("Summary:")
print(df.describe())
print("\nSuccess Rate:", df["goal_reached"].mean())
'''
# Make sure goal_reached is numeric
df["goal_reached"] = df["goal_reached"].astype(int)

# 1️⃣ Describe statistics and save to CSV
summary = df.describe()
summary.to_csv("summary_stats.csv")  # You can now open this in Excel/Sheets
print("\nSuccess Rate:", df["goal_reached"].mean())

'''
# -----------------------------------------------------------
# Plot 1: Episode Score Distribution
# -----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["episode_score"], bins=20)
plt.xlabel("Episode Score")
plt.ylabel("Count")
plt.title("Episode Score Distribution")
plt.tight_layout()
plt.savefig("plots/score_distribution.png")
plt.close()

# -----------------------------------------------------------
# Plot 2: Episode Score vs Training Steps
# -----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["num_training_steps"], df["episode_score"])
plt.xlabel("Training Steps")
plt.ylabel("Episode Score")
plt.title("Score vs. Training Steps")
plt.tight_layout()
plt.savefig("plots/score_vs_steps.png")
plt.close()

# -----------------------------------------------------------
# Plot 3: Success Rate by Algorithm & Action Space
# -----------------------------------------------------------
grouped = df.groupby(["algorithm", "action_space"])["goal_reached"].mean()

plt.figure(figsize=(8,5))
grouped.plot(kind="bar")
plt.ylabel("Success Rate")
plt.title("Success Rate by Algorithm and Action Space")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/success_rate_by_group.png")
plt.close()
'''
# Plot 1: Score distribution
plt.figure(figsize=(8,5))
plt.hist(df["episode_score"], bins=20, color="green")
plt.xlabel("Episode Score")
plt.ylabel("Count")
plt.title("Episode Score Distribution")
plt.savefig("plots/score_distribution.png")
plt.close()

# Plot 2: Score vs steps
plt.figure(figsize=(8,5))
plt.scatter(df["num_training_steps"], df["episode_score"], color="green")
plt.xlabel("Training Steps")
plt.ylabel("Episode Score")
plt.title("Score vs Training Steps")
plt.savefig("plots/score_vs_steps.png")
plt.close()

# Plot 3: Success rate grouping
plt.figure(figsize=(8,5))
grouped = df.groupby(["algorithm", "action_space"])["goal_reached"].mean()
grouped.plot(kind="bar", color="green")
plt.ylabel("Success Rate")
plt.title("Success Rate by Algorithm and Action Space")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/success_rate_by_group.png")
plt.close()

