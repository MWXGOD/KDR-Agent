import json
from typing import Dict, Union, Type, Tuple

class JsonChecker:
    def __init__(self, schema: Dict[str, Union[Type, Tuple[Type, ...]]]):
        """
        schema: dict, key=字段名, value=允许的类型（单个类型或类型元组）
        例如: {"span": str, "type": str, "confident": (int, float, str)}
        """
        self.schema = schema

    def check(self, data: Union[str, list]) -> bool:
        # 如果是字符串，尝试转成 list
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return False
        elif not isinstance(data, list):
            return False

        # 条件检查
        return (
            isinstance(data, list)
            and all(self._check_item(item) for item in data)
        )

    def _check_item(self, item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        for key, expected_type in self.schema.items():
            if key not in item or not isinstance(item[key], expected_type):
                return False
        return True