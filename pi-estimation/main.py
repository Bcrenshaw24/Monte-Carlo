import triton
import math
import torch
import triton.language as tl
import seaborn as sns

sns.set_theme(style="darkgrid")


@triton.jit
def estimate_pi_kernel(
    inside_count,
    trials_per_block,
    BLOCK_SIZE: tl.constexpr
):

    block_id = tl.program_id(0)
    block_inside_accum = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    num_loops = trials_per_block // BLOCK_SIZE

    for i in range(num_loops):
      offsets = tl.arange(0, BLOCK_SIZE)

      seed_x = block_id * trials_per_block + i * BLOCK_SIZE + offsets
      seed_y = seed_x + 987654321

      x = tl.rand(seed_x, offsets)
      y = tl.rand(seed_y, offsets)


      is_inside = (x * x) + (y * y) < 1.0

      block_inside_accum += tl.where(is_inside, 1, 0)

    total_hits = tl.sum(block_inside_accum, axis=0)

    tl.atomic_add(inside_count, total_hits)

def run_simulation(trials_per_block):
    grid_size = 4096
    block_size = 512

    total_trials = grid_size * trials_per_block

    inside_count = torch.zeros(1, dtype=torch.int64, device="cuda")

    estimate_pi_kernel[(grid_size,)](
        inside_count,
        trials_per_block=trials_per_block,
        BLOCK_SIZE=block_size
    )

    torch.cuda.synchronize()

    total_hits = inside_count.item()
    pi_estimate = 4.0 * total_hits / total_trials

    diff = abs(pi_estimate - math.pi)
    error = (diff / math.pi) * 100

    return pi_estimate, error


approximates = []
x = []
for i in range(10, 1000, 10):
  estimate, error = run_simulation(i * 1000)
  approximates.append(estimate)
  x.append(i)

sns.lineplot(x=x, y=approximates)
