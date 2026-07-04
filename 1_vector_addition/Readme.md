# Profiling

- Replay mode: application, since uses llaarge o

```bash
python tools/profile_matrix_runner.py --matrix 1_vector_addition/vector_addition_experiments.yaml --tool ncu --binary ./build/1_vector_addition/vec_add --ncu-replay-mode application --ncu-profiler-set full
```