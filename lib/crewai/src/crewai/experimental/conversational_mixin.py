"""Deprecated compatibility alias for :mod:`crewai.flow.conversational_mixin`."""

import sys

from crewai.flow import conversational_mixin as _real


sys.modules[__name__] = _real
