"""Path and protocol guards for the EXP-012 benchmark and full runs.

This module does not import TensorFlow, training scripts, or data-preparation
scripts. It creates directories only through create_run_directory() and
create_full_run_directory().
"""

import ast
import json
from pathlib import Path

TRAINING_SEED = 42
LEARNING_RATE = 0.001
BENCHMARK_EPOCHS = 2
FULL_EPOCHS = 15
SHUFFLE_BUFFER_SIZE = 256
ALLOWED_BATCH_SIZES = (16, 64)
STUDY_BATCH_SIZES = (16, 32, 64)
EXPECTED_HEAD_TRAINABLE_PARAMS = 164742
FULL_RUNS = (
    {"id": "lr1e-3_bs16", "learning_rate": 1e-3, "batch_size": 16},
    {"id": "lr1e-3_bs32", "learning_rate": 1e-3, "batch_size": 32},
    {"id": "lr1e-3_bs64", "learning_rate": 1e-3, "batch_size": 64},
    {"id": "lr1e-4_bs32", "learning_rate": 1e-4, "batch_size": 32},
    {"id": "lr1e-5_bs32", "learning_rate": 1e-5, "batch_size": 32},
    {"id": "lr1e-6_bs32", "learning_rate": 1e-6, "batch_size": 32},
)
EARLY_STOPPING_KWARGS = {
    "monitor": "val_accuracy",
    "mode": "max",
    "patience": 5,
    "min_delta": 0,
    "restore_best_weights": True,
    "verbose": 1,
}
CHECKPOINT_KWARGS = {
    "monitor": "val_accuracy",
    "mode": "max",
    "save_best_only": True,
    "verbose": 1,
}
SELECTION_RULE = (
    "First epoch whose val_accuracy is strictly greater than every earlier "
    "epoch. Equal values do not replace the checkpoint or the best epoch."
)

FORBIDDEN_RELATIVE_DIRS = (
    "results/hyperparameter_analysis",
    "results/efficientnet",
    "results/efficientnet_augmented",
    "results/efficientnet_garbage",
    "results/efficientnet_household",
    "results/seed_study",
    "results/cnn_baseline",
    "results/mobilenet",
    "data",
    "docs",
)

FORBIDDEN_IMPORT_PREFIXES = (
    "src.training",
    "src.preprocessing",
    "src.data",
    "src.evaluation",
    "src.seed_study.train",
    "src.seed_study.preflight",
    "src.seed_study.datasets",
)

FORBIDDEN_CALLS = frozenset(
    {
        "image_dataset_from_directory",
        "freeze_manifest",
        "rmtree",
        "unlink",
    }
)


def project_root():
    root = Path(__file__).resolve().parents[2]
    model = root / "src" / "models" / "efficientnet_model.py"
    manifest = root / "results" / "seed_study" / "manifest" / "split_info.json"
    if not model.is_file() or not manifest.is_file():
        raise RuntimeError(f"Project markers are missing under {root}")
    return root


def benchmark_root():
    root = (project_root() / "results" / "exp012_comment15" / "_benchmark").resolve()
    if root.parts[-3:] != ("results", "exp012_comment15", "_benchmark"):
        raise RuntimeError(f"Benchmark root escaped its expected location: {root}")
    return root


def full_runs_root():
    root = (project_root() / "results" / "exp012_comment15" / "full_runs").resolve()
    if root.parts[-3:] != ("results", "exp012_comment15", "full_runs"):
        raise RuntimeError(f"Full-run root escaped its expected location: {root}")
    return root


def full_run_config(config_id):
    matches = [item for item in FULL_RUNS if item["id"] == config_id]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown full-run configuration: {config_id}")
    return matches[0]


def full_run_directory(config_id):
    config = full_run_config(config_id)
    return full_runs_root() / config["id"]


def run_directory(batch_size):
    if batch_size not in ALLOWED_BATCH_SIZES:
        raise RuntimeError(
            f"Benchmark batch size must be one of {ALLOWED_BATCH_SIZES}, "
            f"received {batch_size}"
        )
    return benchmark_root() / f"bs{batch_size}_lr1e-3"


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _reject_protected_path(resolved):
    for relative in FORBIDDEN_RELATIVE_DIRS:
        forbidden = (project_root() / relative).resolve()
        if resolved == forbidden or _is_within(resolved, forbidden):
            raise RuntimeError(f"Refusing to write inside protected path {forbidden}")
        if _is_within(forbidden, resolved):
            raise RuntimeError(f"Refusing an output that contains protected path {forbidden}")


def assert_within_benchmark(path):
    resolved = Path(path).resolve()
    root = benchmark_root()
    if not _is_within(resolved, root):
        raise RuntimeError(f"Refusing to write outside the benchmark root: {resolved}")
    _reject_protected_path(resolved)
    return resolved


def assert_within_full_runs(path):
    resolved = Path(path).resolve()
    root = full_runs_root()
    if not _is_within(resolved, root):
        raise RuntimeError(f"Refusing to write outside the full-run root: {resolved}")
    if _is_within(resolved, benchmark_root()) or _is_within(benchmark_root(), resolved):
        raise RuntimeError(f"Refusing to mix benchmark and full-run outputs: {resolved}")
    _reject_protected_path(resolved)
    return resolved


def assert_allowed_output(path):
    resolved = Path(path).resolve()
    if _is_within(resolved, benchmark_root()):
        return assert_within_benchmark(resolved)
    if _is_within(resolved, full_runs_root()):
        return assert_within_full_runs(resolved)
    raise RuntimeError(f"Refusing to write outside EXP-012 output directories: {resolved}")


def create_run_directory(batch_size):
    """Create one new benchmark directory. Refuse if it already exists."""
    run_dir = run_directory(batch_size)
    assert_within_benchmark(run_dir)
    if run_dir.parent != benchmark_root():
        raise RuntimeError(f"Run directory is not a direct child of the benchmark root: {run_dir}")
    if run_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing benchmark directory: {run_dir}"
        )
    benchmark_root().mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing benchmark directory: {run_dir}"
        )
    run_dir.mkdir(exist_ok=False)
    return run_dir


def create_full_run_directory(config_id):
    """Create one new full-run directory. Refuse if it already exists."""
    run_dir = full_run_directory(config_id)
    assert_within_full_runs(run_dir)
    if run_dir.parent != full_runs_root():
        raise RuntimeError(f"Run directory is not a direct child of the full-run root: {run_dir}")
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing full-run directory: {run_dir}")
    full_runs_root().mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing full-run directory: {run_dir}")
    run_dir.mkdir(exist_ok=False)
    return run_dir


def write_json(path, payload):
    destination = assert_allowed_output(path)
    if not destination.parent.is_dir():
        raise RuntimeError(f"Refusing to create an unexpected directory for {destination}")
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def package_directory():
    return Path(__file__).resolve().parent


def imported_modules(tree):
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def called_names(tree):
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.append(function.id)
        elif isinstance(function, ast.Attribute):
            names.append(function.attr)
            owner = function.value
            if isinstance(owner, ast.Name) and owner.id == "os" and function.attr in {"remove", "system"}:
                names.append(f"os.{function.attr}")
    return names


def fit_epoch_literals(tree):
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_fit = (
            isinstance(function, ast.Attribute) and function.attr == "fit"
        ) or (isinstance(function, ast.Name) and function.id == "fit")
        if not is_fit:
            continue
        for keyword in node.keywords:
            if keyword.arg != "epochs" or not isinstance(keyword.value, ast.Constant):
                continue
            if keyword.value.value != BENCHMARK_EPOCHS:
                literals.append(keyword.value.value)
    return literals


def assigned_trainable_true(tree):
    hits = []
    for node in ast.walk(tree):
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if not isinstance(value, ast.Constant) or value.value is not True:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr == "trainable":
                hits.append(node.lineno)
    return hits


def static_safety_report():
    """Read the package source. Do not import runtime modules and do not train."""
    errors = []
    files = sorted(package_directory().glob("*.py"))
    if not files:
        errors.append("package directory has no Python files")
    if not (package_directory() / "full_run.py").is_file():
        errors.append("full_run.py is missing")
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for module in imported_modules(tree):
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                errors.append(f"{path.name} imports {module}")
        for name in called_names(tree):
            if name in FORBIDDEN_CALLS or name in {"os.remove", "os.system"}:
                errors.append(f"{path.name} calls {name}")
        for line in assigned_trainable_true(tree):
            errors.append(f"{path.name}:{line} sets trainable = True")
        for epoch_budget in fit_epoch_literals(tree):
            errors.append(f"{path.name} calls fit with epochs={epoch_budget}")
        if path.name == "full_run.py":
            errors.extend(full_run_fit_errors(tree))
            errors.extend(dry_run_training_errors(tree))
    if BENCHMARK_EPOCHS != 2:
        errors.append("benchmark epoch budget is not 2")
    if FULL_EPOCHS != 15:
        errors.append("full-run epoch budget is not 15")
    if tuple((item["learning_rate"], item["batch_size"]) for item in FULL_RUNS) != (
        (1e-3, 16),
        (1e-3, 32),
        (1e-3, 64),
        (1e-4, 32),
        (1e-5, 32),
        (1e-6, 32),
    ):
        errors.append("full-run configurations do not match the approved six")
    if len({item["id"] for item in FULL_RUNS}) != 6:
        errors.append("full-run configuration ids are not unique")
    if EARLY_STOPPING_KWARGS != {
        "monitor": "val_accuracy",
        "mode": "max",
        "patience": 5,
        "min_delta": 0,
        "restore_best_weights": True,
        "verbose": 1,
    }:
        errors.append("EarlyStopping settings differ from the approved configuration")
    if CHECKPOINT_KWARGS != {
        "monitor": "val_accuracy",
        "mode": "max",
        "save_best_only": True,
        "verbose": 1,
    }:
        errors.append("ModelCheckpoint settings differ from the approved configuration")
    if ALLOWED_BATCH_SIZES != (16, 64):
        errors.append("benchmark batch sizes are not 16 and 64")
    if LEARNING_RATE != 0.001:
        errors.append("benchmark learning rate is not 0.001")
    if SHUFFLE_BUFFER_SIZE != 256:
        errors.append("shuffle buffer is not 256")
    try:
        root = benchmark_root()
    except RuntimeError as exc:
        errors.append(str(exc))
        root = None
    if root is not None:
        for relative in FORBIDDEN_RELATIVE_DIRS:
            forbidden = project_root() / relative
            if _is_within(root, forbidden) or _is_within(forbidden, root):
                errors.append(f"benchmark root collides with {relative}")
        for batch_size in ALLOWED_BATCH_SIZES:
            try:
                assert_within_benchmark(run_directory(batch_size))
            except RuntimeError as exc:
                errors.append(str(exc))
        protected = project_root() / "results" / "efficientnet" / "efficientnet.keras"
        try:
            assert_within_benchmark(protected)
        except RuntimeError:
            pass
        else:
            errors.append("path guard accepted a historical model file")
        manifest = project_root() / "results" / "seed_study" / "manifest" / "split_info.json"
        try:
            assert_within_benchmark(manifest)
        except RuntimeError:
            pass
        else:
            errors.append("path guard accepted the frozen manifest")
        try:
            study_root = full_runs_root()
        except RuntimeError as exc:
            errors.append(str(exc))
            study_root = None
        if study_root is not None:
            if _is_within(study_root, root) or _is_within(root, study_root):
                errors.append("full-run root collides with the benchmark root")
            for relative in FORBIDDEN_RELATIVE_DIRS:
                forbidden = project_root() / relative
                if _is_within(study_root, forbidden) or _is_within(forbidden, study_root):
                    errors.append(f"full-run root collides with {relative}")
            for config in FULL_RUNS:
                try:
                    assert_within_full_runs(full_run_directory(config["id"]))
                except RuntimeError as exc:
                    errors.append(str(exc))
            try:
                assert_within_full_runs(protected)
            except RuntimeError:
                pass
            else:
                errors.append("full-run path guard accepted a historical model file")
            try:
                assert_within_full_runs(manifest)
            except RuntimeError:
                pass
            else:
                errors.append("full-run path guard accepted the frozen manifest")
    return errors


def first_strict_maximum(values):
    """Return the 1-based epoch and value of the first strict maximum."""
    if not values:
        raise ValueError("Cannot select a best epoch from an empty history")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index + 1, best_value


def termination_reason(completed_epochs, stopped_epoch_attribute, epoch_budget):
    """Interpret Keras EarlyStopping without treating its initial 0 as a stop."""
    if completed_epochs == epoch_budget and stopped_epoch_attribute == 0:
        return "epoch_budget"
    if (
        stopped_epoch_attribute > 0
        and completed_epochs == stopped_epoch_attribute + 1
        and completed_epochs < epoch_budget
    ):
        return "early_stopping"
    raise RuntimeError(
        "Termination reason is ambiguous: "
        f"completed_epochs={completed_epochs}, "
        f"stopped_epoch={stopped_epoch_attribute}, "
        f"budget={epoch_budget}"
    )


def full_run_fit_errors(tree):
    errors = []
    fits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "fit":
            fits.append(node)
    if len(fits) != 1:
        errors.append(f"full_run.py has {len(fits)} fit calls")
    for node in fits:
        epochs = next((keyword.value for keyword in node.keywords if keyword.arg == "epochs"), None)
        if not isinstance(epochs, ast.Name) or epochs.id != "FULL_EPOCHS":
            errors.append("full_run.py fit does not use epochs=FULL_EPOCHS")
    return errors


def dry_run_training_errors(tree):
    errors = []
    dry_run = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "dry_run"),
        None,
    )
    if dry_run is None:
        return ["full_run.py has no dry_run function"]
    for node in ast.walk(dry_run):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
        if name in {"fit", "run_config", "create_full_run_directory", "mkdir"}:
            errors.append(f"dry_run calls {name}")
    return errors
