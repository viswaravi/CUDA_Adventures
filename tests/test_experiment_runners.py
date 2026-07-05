import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from backend_runners import build_backend_command, make_backend_runners  # noqa: E402
from experiment_schema import load_experiment_matrix, sanitize_id  # noqa: E402
from matrix_runner import (  # noqa: E402
    build_cli_override_args,
    build_report_stem,
    filter_cases,
    parse_args as parse_matrix_args,
)
from profile_paths import profile_case_dir  # noqa: E402
from profile_runner import build_profiler_options, parse_args as parse_profile_args  # noqa: E402


MATRIX = ROOT / "ops/vector_add/experiments.yaml"


class ExperimentRunnerTests(unittest.TestCase):
    def test_loads_normalized_vector_add_matrix(self) -> None:
        matrix = load_experiment_matrix(MATRIX)
        self.assertEqual(matrix.experiment, "vector_addition")
        self.assertTrue(any(case.id == "cuda_pinned_s4_vectorized_512m"
                            for case in matrix.cases))
        cuda_case = next(case for case in matrix.cases
                         if case.id == "cuda_pinned_scalar_64m")
        self.assertEqual(
            cuda_case.executable,
            "./build/ops/vector_add/ops_vector_add_cuda_runner",
        )

    def test_rejects_legacy_variant_field(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(
                "experiment: vector_addition\n"
                "defaults: {op: vector_add}\n"
                "sweeps:\n"
                "  - id: old\n"
                "    variant: pageable\n"
            )
            path = pathlib.Path(fh.name)
        try:
            with self.assertRaisesRegex(ValueError, "Legacy 'variant'"):
                load_experiment_matrix(path)
        finally:
            path.unlink(missing_ok=True)

    def test_cuda_command_contains_normalized_dtype_and_tuning(self) -> None:
        matrix = load_experiment_matrix(MATRIX)
        case = next(case for case in matrix.cases
                    if case.id == "cuda_pinned_s4_vectorized_512m")
        command = build_backend_command(
            case,
            make_backend_runners(cuda_binary="/bin/echo"),
        ).argv()
        self.assertNotIn("--variant", command)
        self.assertIn("--dtype", command)
        self.assertIn("int32", command)
        self.assertIn("--vector-width", command)
        self.assertIn("4", command)

    def test_triton_and_cublas_commands_are_normalized(self) -> None:
        matrix = load_experiment_matrix(MATRIX)
        runners = make_backend_runners(
            triton_binary="/bin/echo",
            cublas_binary="/bin/echo",
        )
        triton = next(case for case in matrix.cases if case.backend == "triton")
        cublas = next(case for case in matrix.cases if case.backend == "cublas")
        self.assertIn("--block-size", build_backend_command(triton, runners).argv())
        cublas_command = build_backend_command(cublas, runners).argv()
        self.assertIn("--alpha", cublas_command)
        self.assertIn("--dtype", cublas_command)

    def test_id_sanitization(self) -> None:
        self.assertEqual(sanitize_id("pinned-large/s4-512m"), "pinned_large_s4_512m")

    def test_migrated_cuda_ops_generate_normalized_commands(self) -> None:
        runners = make_backend_runners(cuda_binary="/bin/echo")

        reduce_case = load_experiment_matrix(
            ROOT / "ops/reduce/experiments.yaml").cases[0]
        reduce_command = build_backend_command(reduce_case, runners).argv()
        self.assertNotIn("--variant", reduce_command)
        self.assertIn("--kernel", reduce_command)
        self.assertIn("--n", reduce_command)
        self.assertIn("--dtype", reduce_command)
        self.assertIn("float32", reduce_command)
        self.assertIn("--validate", reduce_command)
        self.assertIn("true", reduce_command)
        self.assertIn("--warmup", reduce_command)
        self.assertIn("0", reduce_command)
        self.assertIn("--repeats", reduce_command)
        self.assertIn("1", reduce_command)

        prefix_sum_case = load_experiment_matrix(
            ROOT / "ops/prefix_sum/experiments.yaml").cases[0]
        prefix_sum_command = build_backend_command(prefix_sum_case, runners).argv()
        self.assertNotIn("--variant", prefix_sum_command)
        self.assertIn("--kernel", prefix_sum_command)
        self.assertIn("--n", prefix_sum_command)
        self.assertIn("--dtype", prefix_sum_command)
        self.assertIn("float32", prefix_sum_command)
        self.assertIn("--validate", prefix_sum_command)
        self.assertIn("true", prefix_sum_command)
        self.assertIn("--warmup", prefix_sum_command)
        self.assertIn("0", prefix_sum_command)
        self.assertIn("--repeats", prefix_sum_command)
        self.assertIn("1", prefix_sum_command)

        matmul_case = load_experiment_matrix(
            ROOT / "ops/matmul/experiments.yaml").cases[0]
        matmul_command = build_backend_command(matmul_case, runners).argv()
        self.assertIn("--m", matmul_command)
        self.assertIn("--k", matmul_command)

        conv_case = load_experiment_matrix(
            ROOT / "ops/conv2d/experiments.yaml").cases[1]
        conv_command = build_backend_command(conv_case, runners).argv()
        self.assertIn("--filter", conv_command)
        self.assertNotIn("--variant", conv_command)

    def test_profile_runner_uses_yaml_profiler_defaults(self) -> None:
        args = parse_profile_args([
            "--config", str(MATRIX),
            "--id", "cuda_pinned_scalar_64m",
            "--tool", "ncu",
        ])
        matrix = load_experiment_matrix(MATRIX)
        options = build_profiler_options(args, matrix.profilers["ncu"])
        self.assertIn("--replay-mode", options)
        self.assertIn("application", options)
        self.assertIn("--set", options)
        self.assertIn("detailed", options)

    def test_profile_path_splits_case_id_without_backend_group_duplication(self) -> None:
        path = profile_case_dir(
            pathlib.Path("profiling"),
            "vector_addition",
            "ncu",
            "cuda_pinned_scalar_64m",
        )
        self.assertEqual(
            path,
            pathlib.Path("profiling/vector_addition/ncu/cuda/pinned/scalar/64m"),
        )

    def test_matrix_runner_filters_and_builds_delegation_args(self) -> None:
        matrix = load_experiment_matrix(MATRIX)
        args = parse_matrix_args([
            "--matrix", str(MATRIX),
            "--tool", "ncu",
            "--id", "cuda-pinned-scalar-64m",
            "--profiler-bin", "/bin/echo",
            "--ncu-replay-mode", "kernel",
        ])
        entries = filter_cases(matrix.cases, args)
        self.assertEqual([case.id for case in entries], ["cuda_pinned_scalar_64m"])
        self.assertEqual(
            build_report_stem(entries[0], "dev.build"),
            "cuda_pinned_scalar_64m__builddevpbuild",
        )
        self.assertEqual(
            build_cli_override_args(args),
            ["--profiler-bin", "/bin/echo", "--ncu-replay-mode", "kernel"],
        )


if __name__ == "__main__":
    unittest.main()
