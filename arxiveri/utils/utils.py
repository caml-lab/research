import os
from glob import glob
from copy import deepcopy
import re
from time import sleep
from typing import Dict, List, Tuple

import colorsys
import pandas as pd
from bs4 import BeautifulSoup
import openai
import imgkit
import tiktoken


def shared_items(list1: list, list2: list):
    shared = []

    list1 = deepcopy(list1)
    list2 = deepcopy(list2)
    for item in list1:
        if item in list2:
            shared.append(item)
            list2.remove(item)
    return shared


def preprocess_cell_values(iter_cell_values: iter) -> iter:
    if isinstance(iter_cell_values, dict):
        for k in list(iter_cell_values.keys()):
            cell_value = iter_cell_values[k]
            try:
                new_cell_value: float = float(cell_value)
            except ValueError:
                new_cell_value: str = cell_value
            iter_cell_values[k] = new_cell_value

        return iter_cell_values

    else:
        list_new_cell_values: list = list()
        for cell_value in iter_cell_values:
            try:
                new_cell_value: float = float(cell_value)
            except ValueError:
                new_cell_value: str = cell_value
            list_new_cell_values.append(new_cell_value)

        return list_new_cell_values


def highlight_cells(
        html_table: str,
        dt_cell_locations: List[Tuple[int, int]],
        gt_cell_locations: List[Tuple[int, int]]
) -> str:
    soup = BeautifulSoup(html_table, 'html.parser')

    for row, col in dt_cell_locations:
        try:
            cell = soup.find_all('tr')[row].find_all('td')[col]
        except IndexError:
            # in case the verifier gives an index out of a table width or height
            continue
        cell['style'] = 'background-color: red;'

    for row, col in gt_cell_locations:
        cell = soup.find_all('tr')[row].find_all('td')[col]
        if 'style' in cell.attrs and 'background-color: red;' in cell['style']:
            cell['style'] = 'background-color: green;'
        else:
            cell['style'] = 'background-color: blue;'

    return str(soup)


def save_table_as_png(html_table, file_name):
    options = {
        'format': 'png',
        'encoding': "UTF-8",
        'quality': 100,
        'quiet': ''
    }

    imgkit.from_string(html_table, file_name, options=options)


def get_gpt_response(messages: List[Dict[str, str]], model="gpt-4-0314", temperature: float = 0., num_trial: int = 0) -> str:
    assert model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0314"], ValueError(model)

    try:
        # Make your OpenAI API request here
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )['choices'][0]['message']['content']
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        sleep(1.1 ** num_trial)
        print(f"[# trial: {num_trial}] OpenAI API returned an API Error: {e}")
        response = get_gpt_response(messages, model, temperature, num_trial + 1)

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        sleep(1.1 ** num_trial)
        print(f"[# trial: {num_trial}] Failed to connect to OpenAI API: {e}")
        response = get_gpt_response(messages, model, temperature, num_trial + 1)

    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        sleep(1.1 ** num_trial)
        print(f"[# trial: {num_trial}] OpenAI API request exceeded rate limit: {e}")
        response = get_gpt_response(messages, model, temperature, num_trial + 1)

    except openai.error.Timeout as e:
        sleep(1.1 ** num_trial)
        print(f"[# trial: {num_trial}] OpenAI API request timed out: {e}")
        response = get_gpt_response(messages, model, temperature, num_trial + 1)

    except openai.error.InvalidRequestError as e:
        response = 'openai.error.InvalidRequestError'
        raise e

    return response


def generate_fp(fp: str):
    if os.path.exists(fp):
        base_name, extension = os.path.splitext(fp)

        # count the number of files with the same base name in the directory
        index = len(glob(f"{base_name}*"))
        return f"{base_name}_{index}{extension}"
    else:
        return fp


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-4-0314"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_table_size(html_table: str) -> Dict[str, int]:
    df = pd.read_html(html_table)[0]

    n_rows = len(df)
    n_columns = len(df.columns)

    return {"n_rows": n_rows, "n_columns": n_columns}


def save_tables(
        target_table: str,
        source_table: str,
        gt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]],
        dt_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]],
        shared_cell_mappings: Dict[Tuple[int, int], Tuple[int, int]],
        fp: str
):
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = i / n
            lightness = (i % 2) * 0.4 + 0.4
            saturation = 1
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
        return colors

    def highlight_cells(data, cells):
        other = ''

        mask = pd.DataFrame(other, index=data.index, columns=data.columns)
        for cell in cells:
            attr = f'background-color: {cell["color"]}'
            try:
                mask.iat[cell['rowIndex'], cell['colIndex']] = attr
            except IndexError:
                # if the cell location is out of range, pass
                continue
        return mask

    soup1 = BeautifulSoup(
        target_table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
    )
    soup2 = BeautifulSoup(
        source_table.replace("</thead>", '').replace("<tbody>", '').replace("<thead>", "<tbody>"), 'html.parser'
    )

    df1 = pd.read_html(str(soup1))[0]
    df2 = pd.read_html(str(soup2))[0]

    gt_target_cell_mappings = list(gt_cell_mappings.keys())
    gt_source_cell_mappings = list(gt_cell_mappings.values())

    dt_target_cell_mappings = list(dt_cell_mappings.keys())
    dt_source_cell_mappings = list(dt_cell_mappings.values())

    shared_target_cell_mappings = list(shared_cell_mappings.keys())
    shared_source_cell_mappings = list(shared_cell_mappings.values())

    # among target cells, highlight the shared cells in green, ones only in gt in orange, and ones only in dt in red
    # gt_target_cells_to_highlight = [
    #     {"rowIndex": gt_target_cell_mappings[i][0], "colIndex": gt_target_cell_mappings[i][1], "color": "orange"}
    #     for i in range(len(gt_target_cell_mappings)) if gt_target_cell_mappings[i] not in shared_target_cell_mappings
    # ]

    list_colours = generate_colors(len(dt_target_cell_mappings))
    # remove a green colour
    try:
        list_colours.remove('#00ff00')
    except ValueError:
        pass

    dt_target_cells_to_highlight = [
        {"rowIndex": dt_target_cell_mappings[i][0], "colIndex": dt_target_cell_mappings[i][1], "color": list_colours[i]}
        for i in range(len(dt_target_cell_mappings)) if dt_target_cell_mappings[i] not in shared_target_cell_mappings
    ]
    shared_target_cells_to_highlight = [
        {"rowIndex": shared_target_cell_mappings[i][0], "colIndex": shared_target_cell_mappings[i][1], "color": "#00ff00"}
        for i in range(len(shared_target_cell_mappings))
    ]

    df1_styled = df1.style.apply(
        highlight_cells,
        # cells=gt_target_cells_to_highlight + dt_target_cells_to_highlight + shared_target_cells_to_highlight,
        cells=dt_target_cells_to_highlight + shared_target_cells_to_highlight,
        axis=None
    )

    # do the same for source cells
    # gt_source_cells_to_highlight = [
    #     {"rowIndex": gt_source_cell_mappings[i][0], "colIndex": gt_source_cell_mappings[i][1], "color": "orange"}
    #     for i in range(len(gt_source_cell_mappings)) if gt_source_cell_mappings[i] not in shared_source_cell_mappings
    # ]

    list_colours = generate_colors(len(dt_target_cell_mappings))
    # remove a green colour
    try:
        list_colours.remove('#00ff00')
    except ValueError:
        pass

    dt_source_cells_to_highlight = [
        {"rowIndex": dt_source_cell_mappings[i][0], "colIndex": dt_source_cell_mappings[i][1], "color": list_colours[i]}
        for i in range(len(dt_source_cell_mappings)) if dt_source_cell_mappings[i] not in shared_source_cell_mappings
    ]
    shared_source_cells_to_highlight = [
        {"rowIndex": shared_source_cell_mappings[i][0], "colIndex": shared_source_cell_mappings[i][1], "color": '#00ff00'}
        for i in range(len(shared_source_cell_mappings))
    ]

    df2_styled = df2.style.apply(
        highlight_cells,
        cells=dt_source_cells_to_highlight + shared_source_cells_to_highlight,
        # cells=gt_source_cells_to_highlight + dt_source_cells_to_highlight + shared_source_cells_to_highlight,
        axis=None
    )

    html = "<html><head><style>table {margin-bottom: 30px;}</style></head><body>"
    html += df1_styled.render()
    html += df2_styled.render()
    html += "</body></html>"

    with open(fp, "w") as f:
        f.write(html)


def extract_floats_from_table(table: str) -> List[float]:
    def str_to_number(s):
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            raise ValueError(f"Invalid input string: {s}")

    # convert a string table to a soup
    soup = BeautifulSoup(table, "html.parser")

    # Extract numbers from the table
    list_floats = []
    for row in soup.find_all("tr"):
        for cell in row.find_all(["td", "th"]):
            cell_numbers = re.findall(r"\d+(?:\.\d+)?", cell.text)
            list_floats.extend([str_to_number(n) for n in cell_numbers if isinstance(str_to_number(n), float)])
    return list_floats


def find_shared_floats(list1, list2) -> List[float]:
    list1_no_decimal = []  # set([int(str(x).replace(".", "")) for x in list1])
    list2_no_decimal = []  # set([int(str(x).replace(".", "")) for x in list2])
    for x in list1:
        x_str = str(x)
        if 'e' in x_str:
            x_int = int(x_str.split('e')[0])
        else:
            x_int = int(x_str.replace(".", ""))
        list1_no_decimal.append(x_int)
    list1_no_decimal = set(list1_no_decimal)

    for x in list2:
        x_str = str(x)
        if 'e' in x_str:
            x_int = int(x_str.split('e')[0])
        else:
            x_int = int(x_str.replace(".", ""))
        list2_no_decimal.append(x_int)
    list2_no_decimal = set(list2_no_decimal)

    # list1_no_decimal = set([int(str(x).replace(".", "")) for x in list1])
    # list2_no_decimal = set([int(str(x).replace(".", "")) for x in list2])

    shared_numbers = list1_no_decimal.intersection(list2_no_decimal)

    shared_floats = []
    for num_str in shared_numbers:
        num_float = float("0." + str(num_str))
        shared_floats.append(num_float)

    return sorted(shared_floats)


if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd

    # Load or create your dataset as a DataFrame
    # This example uses the built-in Iris dataset
    df = px.data.iris()

    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(df, color="species_id",
                                  dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
                                  labels={"species_id": "Species", "sepal_width": "Sepal Width",
                                          "sepal_length": "Sepal Length",
                                          "petal_width": "Petal Width", "petal_length": "Petal Length"},
                                  color_continuous_scale=px.colors.diverging.Tealrose)

    # Show the plot
    fig.show()
