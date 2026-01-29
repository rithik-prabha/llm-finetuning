import datetime
import logging
import os
from pathlib import Path

from uvicorn.config import LOGGING_CONFIG


class Directory:
    LOGS: str = "logs"

config = LOGGING_CONFIG
config['loggers']['chat_bot'] = {
    "handlers": ["default"],
    "level": "DEBUG",
    "propagate": True
}

# Get notify-all logger
log = logging.getLogger(__name__)


def init() -> None:
    Path(f"{os.getcwd()}/{Directory.LOGS}").mkdir(parents=True, exist_ok=True)
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"{Directory.LOGS}/{datetime_now}.log"
    logging.basicConfig(
        filename=file_name,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        force=True  # will override any earlier config

    )
    log.info('Logger File Output Initialized')








