import os
import polars as pl
from collections import defaultdict


class RadarDataIndex:
    def __init__(self, data_root):
        self.depth_data_root = os.path.join(data_root, "depthdata")
        self.radar_data_root = os.path.join(data_root, "radardata")
        self.rgbdata_data_root = os.path.join(data_root, "rgbdata")

        self.depth_data_table = self._construct_table(
            self.depth_data_root, parse_depth_data
        )
        self.radar_data_table = self._construct_table(
            self.radar_data_root, parse_radar_data
        )
        self.rgbdata_data_table = self._construct_table(
            self.rgbdata_data_root, parse_rgbdata_data
        )

        self.global_table = self.depth_data_table.join(
            self.radar_data_table, on=["id", "index"], how="inner", suffix="_radar"
        ).join(self.depth_data_table, on=["id", "index"], how="inner", suffix="_depth")

    @property
    def ids(self):
        return self.global_table.get_column("id").unique().sort().to_list()

    def _construct_table(self, root, parse_func):
        root = os.path.abspath(root)
        table = defaultdict(list)
        for file_name in os.listdir(root):
            if file_name.endswith(".npy") or file_name.endswith(".npz"):
                try:
                    parsed_data = parse_func(file_name, root)
                    for key, value in parsed_data.items():
                        table[key].append(value)
                except ValueError as e:
                    print(f"Error parsing {file_name}: {e}")
        return pl.DataFrame(table)

    def get_data_by_id(self, data_id):
        selected_table = self.global_table.filter(pl.col("id") == data_id)
        selected_table = selected_table.sort("index")

        depth_data_files = selected_table["file_name_depth"].to_list()
        radar_data_files = selected_table["file_name_radar"].to_list()
        rgbdata_data_files = selected_table["file_name"].to_list()

        return depth_data_files, radar_data_files, rgbdata_data_files


def parse_depth_data(file_name, file_root):
    """
    2025_05_12_15_48-mastlab-yao-0001
    """

    base_name = file_name.split(".")[0]
    # Parse the base name to extract the relevant information
    parts = base_name.split("-")
    if len(parts) != 4:
        raise ValueError(
            "Base name format is incorrect. Expected format: YYYY_MM_DD_HH_MM-<other_info>-<other_info>-<other_info>"
        )

    return dict(
        id=parts[0],
        location=parts[1],
        subject=parts[2],
        index=int(parts[3]),
        file_name=os.path.join(file_root, file_name),  # Use the full path
    )


def parse_radar_data(file_name, file_root):
    """
    2025_05_12_15_48-mastlab-yao-100-2D-2-2-0001
    """

    base_name = file_name.split(".")[0]
    # Parse the base name to extract the relevant information
    parts = base_name.split("-")
    if len(parts) != 8:
        raise ValueError(
            "Base name format is incorrect. Expected format: YYYY_MM_DD_HH_MM-<other_info>-<other_info>-<other_info>-<other_info>-<other_info>-<other_info>"
        )

    return dict(
        id=parts[0],
        location=parts[1],
        subject=parts[2],
        freqency=int(parts[3]),
        dimension=parts[4],
        angle=int(parts[5]),
        distance=int(parts[6]),
        index=int(parts[7]),
        file_name=os.path.join(file_root, file_name),  # Use the full path
    )


def parse_rgbdata_data(file_name, file_root):
    """
    2025_05_12_15_48-mastlab-yao-0001
    """
    base_name = file_name.split(".")[0]

    # Parse the base name to extract the relevant information
    parts = base_name.split("-")
    if len(parts) != 4:
        raise ValueError(
            "Base name format is incorrect. Expected format: YYYY_MM_DD_HH_MM-<other_info>-<other_info>-<other_info>"
        )

    return dict(
        id=parts[0],
        location=parts[1],
        subject=parts[2],
        index=int(parts[3]),
        file_name=os.path.join(file_root, file_name),  # Use the full path
    )


if __name__ == "__main__":
    data_root = "dataset/radar-data"
    radar_data_index = RadarDataIndex(data_root)
    print(radar_data_index.global_table)
    _, _, depth_data_files = radar_data_index.get_data_by_id("2025_05_12_15_48")
    print(depth_data_files)
    print(radar_data_index.ids)

