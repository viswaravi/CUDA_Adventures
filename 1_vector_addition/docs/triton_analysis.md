## Compiler Decisions

Observed:

- tile size 128x128x32
- software pipelining enabled
- vectorized loads
- tensor cores used

Evidence:

- PTX
- SASS
- Nsight metrics

Possible Reasoning:

- maximize tensor core utilization
- hide memory latency
- improve L2 reuse