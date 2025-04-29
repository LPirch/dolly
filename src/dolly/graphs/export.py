import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger as log

from subprocess import run, PIPE


def export_cpg(sample, pyg_dir: Path, joern_memory: str, joern_cores: int):
    cpg_path = Path(sample["cpg_path"])
    out_file = pyg_dir.absolute() / cpg_path.with_suffix(".json").name
    if not out_file.exists():
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory() as tmp_dir, \
                files("dolly.graphs.joern_scripts") / "export.sc" as export_script, \
                files("dolly.graphs.joern_scripts") / "joern_export.sh" as run_script:
            cmd = [
                str(run_script),
                str(joern_memory),
                str(joern_cores),
                str(export_script),
                str(cpg_path),
                str(out_file)
            ]
            env = os.environ.copy()
            env["TERM"] = "dumb"
            result = run(cmd, stdout=PIPE, stderr=PIPE, cwd=tmp_dir, text=True, env=env)
            if len(result.stderr) > 0:
                log.error(result.stderr)
    return {"pyg_path": str(out_file.absolute())}


if __name__ == "__main__":
    export_cpg(
        Path("data/big-clone-bench/cpg/7667.cpg"),
        Path("tmp/pyg"),
        "2g",
        4,
    )
