from pathlib import Path

import appdirs


def db_storage_path():
    app_name = get_current_package_name()
    app_author = "CrewAI"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_current_package_name():
    return __package__
