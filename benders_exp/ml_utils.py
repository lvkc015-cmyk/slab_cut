from __future__ import annotations

# 它完全忽略了输入的特征值到底是什么，直接利用列表推导式，
#生成一个长度和输入样本数相同的列表，里面全部填满初始化时设定的那个固定值 self.value
class ConstantClassifier:
    """Pickle-stable constant predictor for degenerate single-class heads."""

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def predict(self, x: list[list[float]]) -> list[int]:
        return [self.value for _ in x]
