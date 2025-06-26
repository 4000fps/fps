import logging


def setup_logging() -> None:
    """Setup the logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s %(name)s : %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log", mode="a")],
    )
