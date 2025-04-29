from pathlib import Path
from importlib.resources import files
import subprocess
from tempfile import NamedTemporaryFile


def parse_cpg(sample, cpg_dir: Path, joern_memory: str, joern_cores: int):
    idx = sample["idx"]
    code = sample["func"]
    code = _wrap_in_dummy_class(code)
    out_file = cpg_dir / f"{idx}.cpg"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if not out_file.exists():
        with NamedTemporaryFile(suffix=".java") as f:
            f.write(code.encode())
            f.flush()
            with files('dolly.graphs.joern_scripts') / 'joern_parse.sh' as joern_parse_script:
                subprocess.run(
                    ['bash', str(joern_parse_script), str(f.name), str(out_file), str(joern_memory), str(joern_cores)],
                    capture_output=True,
                    text=True
                )
    return {"cpg_path": str(out_file.absolute())}


def _wrap_in_dummy_class(code: str) -> str:
    if "class" not in code.splitlines()[0]:
        code = "class Dummy {\n" + code + "\n}"
    return code
