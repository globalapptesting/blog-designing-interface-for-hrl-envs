from typing import TypeVar, Any

EnvState = TypeVar("EnvState")
EnvConfig = TypeVar("EnvConfig")
EnvCommonInfo = TypeVar("EnvCommonInfo", bound=dict[str, Any])
