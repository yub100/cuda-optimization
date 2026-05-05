import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gemm_time.csv")
cpu_df = df[df["cpu_ms"] > 0]

plt.figure(figsize=(8, 5))

# plt.plot(cpu_df["M"], cpu_df["cpu_ms"], marker="o", label="CPU")
plt.plot(df["M"], df["shared_ms"], marker="o", label="Shared Memory")
plt.plot(df["M"], df["reg_ms"], marker="o", label="Register Tiling")

plt.xlabel("M = K = N")
plt.ylabel("Time (ms)")
plt.title("GEMM Time vs Matrix Size")
plt.yscale("log")
plt.grid(True, which="both")
plt.legend()

plt.savefig("gemm_time_compare_log.png", dpi=300, bbox_inches="tight")
plt.show()