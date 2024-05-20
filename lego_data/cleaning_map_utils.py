import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def generate_mapping_country_name(
    df1,
    df2,
):
    name_mapping = {}
    for country_name in df1["country"]:
        try:
            chosen_data = df2[df2["iso_a3"].str.startswith(country_name)][
                "iso_a3"
            ].unique()[0]
            name_mapping[country_name] = chosen_data
        except Exception:
            continue

    return name_mapping


def read_world_df():
    world_df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    return world_df
