import warnings
from typing import Dict, List, Tuple, Union
import ast
from utils.utils import shared_items


class CellMatchingEvaluator:
    def __init__(self):
        self.n_dt_cells_tp_recall = 0.
        self.n_gt_cells = 0.

        self.n_dt_cells_tp_prec = 0.
        self.n_dt_cells_tp_fp_prec = 0.

    def get_score(self) -> Dict[str, float]:
        if self.n_gt_cells == 0 or self.n_dt_cells_tp_fp_prec == 0:
            warnings.warn("No data is given.")
            raise ZeroDivisionError

        return {
            "cell_recall": self.n_dt_cells_tp_recall / self.n_gt_cells,
            "cell_precision": self.n_dt_cells_tp_prec / self.n_dt_cells_tp_fp_prec
        }

    @staticmethod
    def preprocess_cell_mappings(cell_mappings: Dict[Tuple[int, int], Tuple[int, int]]) -> List[str]:
        return [f"{k}:{v}".replace(' ', '') for k, v in cell_mappings.items()]

    @staticmethod
    def convert_string_to_cell_mappings(cell_mappings: List[str]) -> Dict[Tuple[int, int], Tuple[int, int]]:
        return {ast.literal_eval(x.split(":")[0]): ast.literal_eval(x.split(":")[1]) for x in cell_mappings}

    def update_metrics(
            self,
            gt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]],
            dt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        gt_cell_mappings: List[str] = self.preprocess_cell_mappings(gt_cell_mappings)
        dt_cell_mappings: List[str] = self.preprocess_cell_mappings(dt_cell_mappings)

        shared_cell_mappings: List[str] = shared_items(gt_cell_mappings, dt_cell_mappings)  # true positives
        assert len(shared_cell_mappings) <= len(gt_cell_mappings)

        tp: int = len(shared_cell_mappings)
        tp_fp: int = len(dt_cell_mappings)
        tp_fn: int = len(gt_cell_mappings)
        assert tp_fn >= tp, f"{tp_fn} < {tp}"
        assert tp_fp >= tp, f"{tp_fp} < {tp}"

        self.n_dt_cells_tp_recall += tp
        self.n_dt_cells_tp_fp_prec += tp_fp

        self.n_dt_cells_tp_prec += tp
        self.n_gt_cells += tp_fn

        return self.convert_string_to_cell_mappings(shared_cell_mappings)

    def __call__(
            self,
            gt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]],
            dt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> Dict[str, Union[float, Dict[Tuple[int, int], Tuple[int, int]]]]:
        shared_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]] = self.update_metrics(
            gt_cell_mappings=gt_cell_mappings,
            dt_cell_mappings=dt_cell_mappings,
        )
        return {
            "cell_recall": self.n_dt_cells_tp_recall / self.n_gt_cells,
            "cell_precision": self.n_dt_cells_tp_prec / self.n_dt_cells_tp_fp_prec,
            "shared_cell_mappings": shared_cell_mappings
        }
