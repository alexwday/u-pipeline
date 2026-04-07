"""Single-command setup wizard for u-pipeline (phase 1).

Run from the project root with any Python 3.12+:

    python3 scripts/setup.py

Phase 1 lives here and uses *only* the standard library so it can run
in a fresh interpreter before the venv exists. It:

    1. Creates .venv at the project root.
    2. Installs all subprojects in editable mode.
    3. Tries to install rbc_security from the work-internal index;
       failures are tolerated so personal machines can continue.
    4. Copies .env.example -> .env where missing.
    5. Opens each .env in the user's editor and waits for them to
       save the credentials.
    6. Re-execs scripts/setup_phase2.py via the venv interpreter.

Phase 2 (validation, schema, seed data, debug server launch) lives in
scripts/setup_phase2.py because it needs psycopg2, dotenv, and the
ingestion/retriever packages, none of which exist until phase 1 has
finished installing them.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
PHASE2_MODULE = "scripts.setup_phase2"

SUBPROJECTS = ["u-ingestion", "u-retriever", "u-debug"]
ENV_SUBPROJECTS = ["u-ingestion", "u-retriever"]
OPTIONAL_PACKAGES = ["rbc_security"]


def _banner(text: str) -> None:
    """Print a section banner. Params: text. Returns: None."""
    width = 64
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def _check_python_version() -> None:
    """Verify Python >= 3.12. Returns: None."""
    if sys.version_info < (3, 12):
        print(f"Python 3.12+ required, found {sys.version}")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} OK")


def _venv_python() -> Path:
    """Return path to the venv Python interpreter. Returns: Path."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_env() -> dict:
    """Build a clean environment dict for venv subprocess calls.

    Strips PYTHONHOME and PYTHONPATH so the venv interpreter does
    not accidentally import modules (certifi, urllib3, ssl) from
    the system Python that launched setup.py. This is the most
    common cause of "SSL: CERTIFICATE_VERIFY_FAILED" during pip
    install on machines whose system Python has a stale or
    corporate-CA-customised certifi. Returns: dict.
    """
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    return env


def _create_venv() -> None:
    """Create .venv at project root if missing. Returns: None."""
    if VENV_DIR.exists():
        print(f"Venv exists at {VENV_DIR}")
        return
    print("Creating virtual environment...")
    subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        check=True,
    )
    print(f"Created {VENV_DIR}")


def _pip_install(args: list[str], check: bool) -> int:
    """Run `python -m pip install ...` in the venv with a clean env.

    Always invokes pip via `python -m pip` (not the .venv/bin/pip
    shebang script) so the venv interpreter is in charge of module
    resolution from the very first import. Always passes a scrubbed
    environment via _venv_env() to avoid leaking PYTHONPATH from the
    system interpreter.

    Params:
        args: pip arguments after `install` (e.g. ["--upgrade", "pip"])
        check: raise on non-zero exit if True

    Returns:
        process return code
    """
    py = str(_venv_python())
    cmd = [py, "-m", "pip", "install"] + args
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=_venv_env(),
        check=check,
    )
    return result.returncode


def _install_dependencies() -> None:
    """Install all subprojects in editable mode. Returns: None."""
    print("Upgrading pip in venv...")
    _pip_install(["--upgrade", "pip", "setuptools", "wheel"], check=True)
    args: list[str] = []
    for sub in SUBPROJECTS:
        args.extend(["-e", f"./{sub}[dev]"])
    print("Installing subprojects...")
    _pip_install(args, check=True)
    print("Subprojects installed")


def _install_optional_packages() -> None:
    """Try installing work-internal optional packages.

    Failures are expected on non-work environments and only produce a
    warning — they never abort the setup. Returns: None.
    """
    for package in OPTIONAL_PACKAGES:
        print(f"Attempting optional install: {package}")
        rc = _pip_install([package], check=False)
        if rc == 0:
            print(f"  {package}: installed")
        else:
            print(
                f"  {package}: not available in this environment "
                "(skipping — only present on the work index)"
            )


def _copy_env_files() -> list[Path]:
    """Copy .env.example -> .env where missing.

    Returns:
        list of .env paths that exist after copying
    """
    env_files: list[Path] = []
    for sub in ENV_SUBPROJECTS:
        env_file = PROJECT_ROOT / sub / ".env"
        example = PROJECT_ROOT / sub / ".env.example"
        if env_file.exists():
            print(f"  {sub}/.env already exists")
        elif example.exists():
            shutil.copy2(example, env_file)
            print(f"  Created {sub}/.env from .env.example")
        else:
            print(f"  WARNING: {sub}/.env.example not found")
            continue
        env_files.append(env_file)
    return env_files


def _editor_command(file_path: Path) -> list[str]:
    """Build the editor command for opening a file.

    Honours $VISUAL, then $EDITOR. Falls back to a sensible
    platform-specific default. On macOS we intentionally do NOT
    use `open -W` because it waits for the whole application to
    quit (not just the document window), which hangs forever if
    TextEdit has any other document open. The "Press Enter to
    confirm" prompt in _edit_env_file handles the wait instead.

    Params:
        file_path: file to open

    Returns:
        argv-style list ready for subprocess.run
    """
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        return shlex.split(editor) + [str(file_path)]
    if sys.platform == "darwin":
        return ["open", "-t", str(file_path)]
    if sys.platform.startswith("linux"):
        return ["nano", str(file_path)]
    return ["notepad", str(file_path)]


def _edit_env_file(env_file: Path) -> None:
    """Open a .env file in the user's editor and wait for confirmation.

    Params:
        env_file: path to the .env file
    """
    rel = env_file.relative_to(PROJECT_ROOT)
    print(f"\nNext: {rel}")
    print(
        "  Fill in DB connection, AUTH_MODE + credentials, "
        "LLM_ENDPOINT, and any model overrides."
    )
    input(f"  Press Enter to open {rel} in your editor...")
    cmd = _editor_command(env_file)
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(
            "  Editor command not found. Edit the file manually "
            f"at: {env_file}"
        )
    input(
        f"  Save your changes, then press Enter to confirm "
        f"{rel} is ready..."
    )


def _edit_all_env_files(env_files: list[Path]) -> None:
    """Walk the user through editing each .env file. Params: env_files."""
    if not env_files:
        return
    _banner("Configure .env files")
    print(
        "Each subproject has its own .env. They share most keys "
        "(DB, AUTH, LLM) so you can paste the same values into both."
    )
    for env_file in env_files:
        _edit_env_file(env_file)


def main() -> None:
    """Run phase 1 of the setup wizard. Returns: None."""
    _banner("U-Pipeline Setup")
    _check_python_version()

    _banner("Phase 1: Environment")
    _create_venv()
    _install_dependencies()
    _install_optional_packages()

    _banner("Phase 1.5: .env files")
    print("Copying .env files where missing:")
    env_files = _copy_env_files()
    _edit_all_env_files(env_files)

    print("\nRe-launching inside venv for phase 2...")
    # Launch via `-m` so PROJECT_ROOT is on sys.path and phase 2
    # can `from scripts import seed_data`. Running the file path
    # directly would put scripts/ (not PROJECT_ROOT) on sys.path[0].
    result = subprocess.run(
        [str(_venv_python()), "-m", PHASE2_MODULE],
        cwd=str(PROJECT_ROOT),
        env=_venv_env(),
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
