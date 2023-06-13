import warnings


class TableMatchingEvaluator:
    def __init__(self):
        self.total: int = 0
        self.correct: int = 0

    def get_score(self) -> float:
        if self.total == 0:
            warnings.warn("No data is given.")
            raise ZeroDivisionError

        return self.correct / self.total

    def __call__(self, gt_source_table: str, dt_source_table: str) -> float:
        self.total += 1
        if gt_source_table == dt_source_table:
            self.correct += 1
        return self.correct / self.total
