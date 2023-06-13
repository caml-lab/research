import os
import re
import json
from typing import Dict, List, Tuple, Union
from bs4 import BeautifulSoup
import pandas as pd
import ast
from utils.utils import get_gpt_response


class CellMatcher:
    def __init__(
            self,
            dir_table_match: str,
            temperature: float = 0.,
            no_cell_indices: bool = False
    ):
        assert os.path.exists(dir_table_match), f"{dir_table_match} does not exist."

        self.dir_table_match: str = dir_table_match
        self.temperature: float = temperature
        self.no_cell_indices: bool = no_cell_indices

    @staticmethod
    def extract_dictionary_from_string(string: str):
        """Extract a dictionary whose keys and values are a tuple of two integers from a string."""
        pattern = r'\{\s*\(\s*\d+\s*,\s*\d+\s*\)\s*:\s*\(\s*\d+\s*,\s*\d+\s*\)\s*(,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*:\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,*\s*)*\}'

        st = time()
        match = re.search(pattern, string, re.DOTALL)
        print(f"Time taken to search for the pattern: {time() - st:.2f} seconds.")

        if match:
            dict_str = match.group()
            try:
                return ast.literal_eval(dict_str)
            except SyntaxError:
                print(f"Syntax error in the following string:\n\n{dict_str}\n\n")
                raise SyntaxError
        else:
            print(f"No match found in the given string:\n\n{string}\n\n")
            return {}

    def get_gpt_4_prediction(
            self,
            target_table: str,
            source_table: str,
            model: str = "gpt-4-0314",
            format: str = "html"
    ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], str]:
        messages: List[Dict[str, str]] = list()
        messages.append({"role": "system", "content": "You are a helpful assistant."})

        if format == "html":
            messages.append({
                "role": "user",
                "content": "Compare the following target and source tables and identify cells that contain floating point numbers with the same meaning present in both tables. "
                           "Return the matched cells in a Python dictionary with the following format:\n"
                           "{(target_table_row_index, target_table_column_index): (source_table_row_index, source_table_column_index), ...}}\n"
                           "Use 0-based indexing, including headers, rowspan, and colspan attributes. Locate as many matching cell pairs as possible. "
                           "If no matches are found, return an empty dictionary ({})."
            })
        elif format in ["markdown", "csv"]:
            if self.no_cell_indices:
                messages.append({
                    "role": "user",
                    "content": "Compare the following target and source tables and identify cells that contain floating point numbers with the same meaning present in both tables. "
                               "Return the matched cells in a Python dictionary with the following format:\n"
                               "{(target_table_row_index, target_table_column_index): (source_table_row_index, source_table_column_index), ...}}\n"
                               "Use 0-based indexing. Locate as many matching cell pairs as possible. "
                               "If no matches are found, return an empty dictionary ({})."
                })

            else:
                messages.append({
                    "role": "user",
                    "content": "Compare the following target and source tables and identify cells that contain floating point numbers with the same meaning present in both tables. "
                               "Return the matched cells in a Python dictionary with the following format:\n"
                               "{(target_table_row_index, target_table_column_index): (source_table_row_index, source_table_column_index), ...}}\n"
                               "Use the row and column indices provided on the leftmost column and the topmost row of the tables, respectively. "
                               "These indices are numerical and serve as identifiers to specify the location of each cell within the table. "
                               "The row index is listed vertically along the left side of the table, while the column index is listed horizontally at the top. "
                               "If no matches are found, return an empty dictionary ({})."
                })
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

        messages.append(
            {"role": "user",
             "content": f"The target table and its caption:\n{target_table}\n"
                        f"The source table and its caption:\n{source_table}\n"}
        )

        response: str = get_gpt_response(messages, model=model, temperature=self.temperature)
        dict_cells: dict = self.extract_dictionary_from_string(string=response)
        return dict_cells, response

    def match(
            self,
            target_table: str,
            source_table: str,
            format: str = "html"
    ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], str]:
        return self.get_gpt_4_prediction(
            target_table=target_table,
            source_table=source_table,
            format=format
        )

    def convert_html_to_markdown(self, table: str) -> str:
        soup = BeautifulSoup(
            table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
        )
        df = pd.read_html(str(soup))[0]

        if self.no_cell_indices:
            df.reset_index(drop=True, inplace=True)
            markdown_table = df.to_markdown(index=False)
            markdown_table_no_header = "\n".join(markdown_table.split("\n")[2:])  # Remove first two lines which contain the header
            return markdown_table_no_header
        else:
            return df.to_markdown()

    def convert_html_to_csv(self, table: str) -> str:
        soup = BeautifulSoup(
            table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
        )
        df = pd.read_html(str(soup))[0]

        if self.no_cell_indices:
            df.reset_index(drop=True, inplace=True)
            csv_table_no_header = df.to_csv(index=False, header=False)
            return csv_table_no_header
        else:
            return df.to_csv()

    def __call__(
            self,
            target_arxiv_id: str,
            source_arxiv_id: str,
            format: str = "html"
    ) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], str]:
        # load a target table and a source table
        fp_table_match: str = f"{self.dir_table_match}/{target_arxiv_id}/{source_arxiv_id}.json"
        table_match: Dict[str, str] = json.load(open(fp_table_match, "r"))

        target_table = table_match["target_table"]
        source_table = table_match["source_table"]  # "gt" source table

        if format == "markdown":
            target_table: str = self.convert_html_to_markdown(target_table)
            source_table: str = self.convert_html_to_markdown(source_table)
        elif format == "csv":
            target_table: str = self.convert_html_to_csv(target_table)
            source_table: str = self.convert_html_to_csv(source_table)

        target_table = target_table + '\n' + table_match["target_caption"]
        source_table = source_table + '\n' + table_match["source_caption"]  # "gt" source table

        # get response from GPT-4
        dt_cell_matches, response = self.match(
            target_table=target_table,
            source_table=source_table,
            format=format
        )

        return dt_cell_matches, response


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob
    from time import time
    from tqdm import tqdm
    from utils.cell_matching_evaluator import CellMatchingEvaluator
    from utils.utils import save_tables

    parser = ArgumentParser()
    parser.add_argument(
        "--dir_root",
        type=str,
        default="/Users/noel/projects/caml/research/arxiveri"
    )
    parser.add_argument(
        "--dir_table_match",
        type=str,
        default="dataset/ground_truth/table_match"
    )
    parser.add_argument(
        "--dir_cell_match",
        type=str,
        default="dataset/ground_truth/cell_match"
    )
    parser.add_argument(
        "--dir_ckpt",
        type=str,
        default="results/cell_match"
    )
    parser.add_argument("--format", type=str, default="html", choices=["html", "csv", "markdown"])
    parser.add_argument("--no_cell_indices", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    args.dir_table_match = f"{args.dir_root}/{args.dir_table_match}"
    args.dir_cell_match = f"{args.dir_root}/{args.dir_cell_match}"
    args.dir_ckpt = f"{args.dir_root}/{args.dir_ckpt}"

    # STEP 0: load the list of cell matches
    list_fp_cell_matches: List[str] = sorted(glob(f"{args.dir_cell_match}/*/*.json"))
    assert len(list_fp_cell_matches) > 0

    # filter fp_cell_matches that have no cell matches
    new_list_fp_cell_matches = []
    for fp_cell_match in list_fp_cell_matches:
        gt_cell_matches = json.load(open(fp_cell_match, "r"))
        if len(gt_cell_matches) != 0:
            new_list_fp_cell_matches.append(fp_cell_match)
    list_fp_cell_matches = new_list_fp_cell_matches

    # STEP 1: set the checkpoint directory with the temperature
    dir_ckpt: str = f"{args.dir_ckpt}/temperature_{int(args.temperature * 100):03d}"
    if args.no_cell_indices:
        dir_ckpt = f"{dir_ckpt}_{args.format}_no_cell_indices"
    else:
        dir_ckpt = f"{dir_ckpt}_{args.format}"

    # count the number of directories that start with "run_" in the checkpoint directory
    list_run_dir: List[str] = sorted(glob(f"{dir_ckpt}/run_*"))
    n_previous_runs: int = len(list_run_dir)

    continue_flag: bool = False
    if n_previous_runs > 0:
        # check if the latest run contains all the gpt-4 responses
        list_fp_previous_gpt_4_responses: List[str] = sorted(glob(
            f"{dir_ckpt}/run_{n_previous_runs - 1:02d}/gpt_4_responses/*/*.json"
        ))

        if len(list_fp_previous_gpt_4_responses) > 0:
            remaining_list_fp_cell_matches = []
            # compare the last two words of a file path in the lists and remove the matched ones from
            # list_fp_cell_matches
            for fp_cell_match in list_fp_cell_matches:
                flag: bool = True

                for fp_response in list_fp_previous_gpt_4_responses:
                    fp_response = '/'.join(fp_response.split("/")[-2:])

                    if '/'.join(fp_cell_match.split("/")[-2:]) == fp_response:
                        print('/'.join(fp_cell_match.split("/")[-2:]), fp_response)
                        flag = False
                        break
                    
                if flag:
                    remaining_list_fp_cell_matches.append(fp_cell_match)

            if len(remaining_list_fp_cell_matches) == 0:
                # if the latest run contains all the gpt-4 responses, create a new directory
                dir_ckpt = f"{dir_ckpt}/run_{n_previous_runs:02d}"
            else:
                # if the latest run does not contain all the gpt-4 responses, use the latest run
                dir_ckpt = f"{dir_ckpt}/run_{n_previous_runs - 1:02d}"
                list_fp_cell_matches = remaining_list_fp_cell_matches
                print(f"Running for the previous run... ({len(list_fp_cell_matches)} remaining)")
                continue_flag: bool = True
        else:
            dir_ckpt = f"{dir_ckpt}/run_{n_previous_runs:02d}"

    else:
        dir_ckpt = f"{dir_ckpt}/run_00"
    
    print(f"Checkpoint directory: {dir_ckpt}")

    cell_matcher = CellMatcher(
        dir_table_match=args.dir_table_match,
        temperature=args.temperature,
        no_cell_indices=args.no_cell_indices,
    )

    cell_matching_evaluator: callable = CellMatchingEvaluator()

    pbar = tqdm(list_fp_cell_matches)
    for i, fp_cell_match in enumerate(pbar):
        target_arxiv_id: str = fp_cell_match.split("/")[-2]
        source_arxiv_id: str = fp_cell_match.split("/")[-1].replace(".json", "")

        gt_cell_matches = json.load(open(fp_cell_match, "r"))
        assert len(gt_cell_matches) > 0, f"{fp_cell_match} has no cell matches"

        # dt
        start = time()
        dt_cell_mappings, response = cell_matcher(
            target_arxiv_id=target_arxiv_id,
            source_arxiv_id=source_arxiv_id,
            format=args.format
        )
        end = time()

        # gt
        assert len(gt_cell_matches) % 2 == 0, f"{gt_cell_matches} has an odd number of cells"
        gt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]] = dict()
        target_gt_cell_locations = [
            (cell["rowIndex"], cell["colIndex"]) for i, cell in enumerate(gt_cell_matches) if i % 2 == 0
        ]
        source_gt_cell_locations = [
            (cell["rowIndex"], cell["colIndex"]) for i, cell in enumerate(gt_cell_matches) if i % 2 == 1
        ]

        for target_gt_cell_location, source_gt_cell_location in zip(target_gt_cell_locations, source_gt_cell_locations):
            gt_cell_mappings[target_gt_cell_location] = source_gt_cell_location

        metric: Dict[str, Union[float, List[str]]] = cell_matching_evaluator(
            gt_cell_mappings=gt_cell_mappings, dt_cell_mappings=dt_cell_mappings
        )
        shared_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]] = metric["shared_cell_mappings"]

        pbar.set_description(
            f"(t:{args.temperature}) cell recall: {metric['cell_recall']:.3f} | "
            f"cell precision: {metric['cell_precision']:.3f} | "
            f"forward time: {end - start:.3f} sec."
        )

        fp_response: str = f"{dir_ckpt}/gpt_4_responses/{target_arxiv_id}/{source_arxiv_id}.json"
        os.makedirs(os.path.dirname(fp_response), exist_ok=True)
        json.dump(response, open(fp_response, "w"))

        fp_shared_cell_mappings: str = f"{dir_ckpt}/shared_cell_mappings/{target_arxiv_id}/{source_arxiv_id}.json"
        os.makedirs(os.path.dirname(fp_shared_cell_mappings), exist_ok=True)
        json.dump({str(k): v for k, v in shared_cell_mappings.items()}, open(fp_shared_cell_mappings, "w"))

        fp_gt_cell_mappings: str = f"{dir_ckpt}/shared_cell_mappings/{target_arxiv_id}/{source_arxiv_id}_gt.json"
        json.dump({str(k): v for k, v in gt_cell_mappings.items()}, open(fp_gt_cell_mappings, "w"))

        fp_table_match = f"{args.dir_table_match}/{target_arxiv_id}/{source_arxiv_id}.json"
        table_match = json.load(open(fp_table_match, "r"))

        save_tables(
            target_table=table_match["target_table"],
            source_table=table_match["source_table"],
            gt_cell_mappings=gt_cell_mappings,
            dt_cell_mappings=dt_cell_mappings,
            shared_cell_mappings=shared_cell_mappings,
            fp=f"{dir_ckpt}/shared_cell_mappings/{target_arxiv_id}/{source_arxiv_id}.html"
        )

    if continue_flag:
        fp_metric: str = f"{dir_ckpt}/metric_continued.json"
    else:
        fp_metric: str = f"{dir_ckpt}/metric.json"
    os.makedirs(os.path.dirname(fp_metric), exist_ok=True)
    json.dump(cell_matching_evaluator.get_score(), open(fp_metric, "w"))
