# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .xlerobot_2wheels import XLerobot2Wheels
from .xlerobot_2wheels_client import XLerobot2WheelsClient
from .config_xlerobot_2wheels import (
    XLerobot2WheelsConfig,
    XLerobot2WheelsClientConfig,
    XLerobot2WheelsHostConfig,
)

__all__ = [
    "XLerobot2Wheels",
    "XLerobot2WheelsClient", 
    "XLerobot2WheelsHost",
    "XLerobot2WheelsConfig",
    "XLerobot2WheelsClientConfig",
    "XLerobot2WheelsHostConfig",
]


def __getattr__(name: str):
    if name == "XLerobot2WheelsHost":
        from .xlerobot_2wheels_host import XLerobot2WheelsHost

        return XLerobot2WheelsHost
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
