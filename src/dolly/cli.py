import typer

from dolly.dataset import load_hf


app = typer.Typer()

app.command("init-dataset")(load_hf)


if __name__ == "__main__":
    app()
