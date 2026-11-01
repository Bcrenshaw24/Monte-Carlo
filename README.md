# Monte-Carlo 

A repository deticated to the exploration and application of Monte Carlo and similar topics. Applications include computing, mathematics, phsics, and finance. 

## PI Estimation 

I implemented this to get a good understanding of how Monte Carlo works in a base sense and how it can be done in CUDA. I added a triton version to analyze the error as the number of trials increased. I also wanted to compare CUDA vs Triton in terms of accuracy. 

### Results 

Error = (|esitimate - baseline| / baseline) * 100 

#### CUDA 

- Threads per Block: 215
- Blocks per Grid: 4096 
- Trials per Thread: 10,000
- Number of Trials: 10,485,760,000

Results: 

- Time: 78,818,925 ns
- Error: 0.00012%

#### Triton

- Grid Size: 4096
- Block Size: 256
- Trials per Thread: 10,000
- Number of Trials: 10,485,760,000

Results: 

- Time: 494,546,786 ns
- Error: 0.00028%

While Triton is excellent for experimentation and deploying models such as LLMs, it did seem to lag behind raw CUDA in terms of performance and accuracy as CUDA was over 6x faster. Though this comparison isn't entirely fair as Triton executes in a slightly more sequential manner thats optimized for different workloads as stated earlier.

