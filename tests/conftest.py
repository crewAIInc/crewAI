# conftest.py
from dotenv import load_dotenv

import os

os.environ["OTEL_SDK_DISABLED"] = "true"


load_result = load_dotenv(override=True)
