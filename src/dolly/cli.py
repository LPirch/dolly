import typer

from dolly.dataset import app as dataset_app
from dolly.train import app as train_app
from dolly.evaluate import app as evaluate_app
from dolly.models import init_hf_cache

app = typer.Typer(pretty_exceptions_show_locals=False)

app.command("init-hf-cache")(init_hf_cache)
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")

if __name__ == "__main__":
    app()
