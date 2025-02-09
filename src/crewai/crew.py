import asyncio
import json
import uuid
import warnings
from concurrent.futures import Future
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    InstanceOf,
    Json,
    PrivateAttr,
    field_validator,
    model_validator,
)

# Rest of crew.py content...
