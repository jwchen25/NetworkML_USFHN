import numpy as np
import pandas as pd
import hnelib.plot
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


PLOT_VARS = {
    'figures': {
        'page-width': 17,
    },
    'colors': {
        'color_1': '#45A1F8',
        'color_2': '#FF6437', 
        'gender': {
            'All': '#5E5E5E',
            'Male': '#4B917D',
            'Female': '#FF6437',
            'Unknown': '#509BF5',
        },
        'dark_gray': '#5E5E5E',
        'categories': {
            'Humanities': '#9BC53D',
            'Social Sciences': '#5BC0EB',
            'STEM': '#E55934',
        },
        'department-level': {
            'black': '#0e6101',
            'red': '#4c4c4c', 
            'green': '#a30005'
        },
        'academia': '#45A1F8',
        'Academia': '#45A1F8',
        'umbrellas': {
            'Applied Sciences': '#1b9e77',
            'Education': '#d95f02',
            'Engineering': '#7570b3',
            'Mathematics and Computing': '#e7298a',
            'Mathematics & Computing': '#e7298a',
            'Humanities': '#66a61e',
            'Medicine and Health': '#e6ab02',
            'Medicine & Health': '#e6ab02',
            'Natural Sciences': '#a6761d',
            'Social Sciences': '#666666',
            # academia's not an umbrella but it's nice to be able to call it by this
            'Academia': '#45A1F8',
            # extra umbrellas
            'Public Administration and Policy': '#1C4D7C',
            'Public Administration & Policy': '#1C4D7C',
            'Journalism, Media, Communication': '#A62A17',
        },
        'ranks': {
            'Assistant Professor': '#e41a1c',
            'Associate Professor': '#377eb8',
            'Professor': '#4daf4a',
        },
    },
    'arrowprops': {
        'arrowstyle': '->',
        'connectionstyle': 'arc3',
        'color': '#5E5E5E', # dark gray
        'lw': .5,
    },
    'abbreviations': {
        'Academia': 'Academia',
        'Applied Sciences': 'App Sc.',
        'Education': 'Edu.',
        'Engineering': 'Eng.',
        'Mathematics and Computing': 'Comp. & Math.',
        'Humanities': 'Hum.',
        'Medicine and Health': 'Med. & Hlth.',
        'Natural Sciences': 'Nat. Sc.',
        'Social Sciences': 'Soc. Sc.',
    },
    'text': {
        'annotations': {
            'labelsize': 12,
        },
        'axis': {
            'fontsize': 16,
        },
    },
    'artists': {
        'title': {
            'functions': [
                (matplotlib.axes.Axes, 'set_title')
            ],
            'kwargs': {
                'pad': 10,
                'fontsize': 30,
                'size': 30,
            },
        },
        'axis_labels': {
            'functions': [
                (matplotlib.axes.Axes, 'set_xlabel'),
                (matplotlib.axes.Axes, 'set_ylabel'),
            ],
            'kwargs': {
                'size': 24,
            },
        },
        'legend': {
            'functions': [
                (matplotlib.axes.Axes, 'legend'),
            ],
            'kwargs': {
                'prop': {
                    'size': 20,
                },
            },
        },
    },
    'subplots_set_methods': [
        {
            'method': 'tick_params',
            'kwargs': {
                'labelsize': 16,
            },
        },
    ],
}


def filter_to_academia_and_domain(df):
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    return df


def annotate_color(df, add_umbrella=False):
    """
    expects df to have columns:
    - TaxonomyLevel
    - TaxonomyValue
    """

    cols = list(df.columns)

    new_dfs = []
    for level, rows in df.groupby('TaxonomyLevel'):
        rows = rows.copy()

        if level == 'Academia':
            rows['Umbrella'] = 'Academia'
        elif level == 'Umbrella':
            rows['Umbrella'] = rows['TaxonomyValue']

        rows['Color'] = rows['Umbrella'].apply(PLOT_VARS['colors']['umbrellas'].get)
        new_dfs.append(rows)

    new_cols = ['Color']

    if add_umbrella:
        new_cols.append('Umbrella')

    new_df = pd.concat(new_dfs)
    new_df = new_df[
        cols + new_cols
    ]

    return new_df


def add_gridlines_and_annotate_rank_direction(
    ax,
    rank_type='production',
    x_gridlines_to_break=[],
    annotation_arrow_x=.01,
    annotation_text_x=.12,
    break_height=.0075,
    fontsize=hnelib.plot.FONTSIZES['annotation'],
):
    if not x_gridlines_to_break:
        x_gridlines_to_break = [.2, .4] if rank_type == 'production' else [.2]

    text = 'more faculty produced' if rank_type == 'production' else "more prestigious"

    hnelib.plot.add_gridlines(
        ax,
        xs=[x for x in ax.get_xticks() if x not in x_gridlines_to_break],
        ys=ax.get_yticks(),
    )

    annotation_y = break_height / 2

    ax.annotate(
        "",
        xy=(annotation_arrow_x, annotation_y),
        xytext=(annotation_text_x, annotation_y),
        arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
    )

    ax.annotate(
        text,
        xy=(annotation_text_x, annotation_y),
        ha='left',
        va='center',
        fontsize=fontsize,
    )

    for x in x_gridlines_to_break:
        ax.plot(
            [x, x],
            [break_height, ax.get_ylim()[1]],
            lw=.5,
            alpha=.5,
            color=PLOT_VARS['colors']['dark_gray'],
        )


def filter_by_taxonomy(df, level=None, value=None):
    df = df.copy()
    if level:
        df = df[
            df['TaxonomyLevel'] == level
        ]

    if value:
        df = df[
            df['TaxonomyValue'] == value
        ]

    return df


def plot_basic_geodesics(
    ax=None,
    df=pd.DataFrame(),
    level='Academia',
    value='Academia',
    color=PLOT_VARS['colors']['umbrellas']['Academia'],
    annotate_rank_direction=True,
):
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    df = filter_by_taxonomy(df, level, value)

    rolling_rows = []
    # window = .025
    window = .05
    for i in np.arange(0, 1.01, .01):
        rows = df[
            (df['Percentile'] < i + window)
            &
            (df['Percentile'] > i - window)
        ].copy()

        rolling_rows.append({
            'Percentile': i,
            'MeanFractionalPathLength': rows['FractionalPathLength'].mean(),
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
        })

    df = pd.DataFrame(rolling_rows)

    df = df.sort_values(by='Percentile')

    ax.plot(
        df['Percentile'],
        df['MeanFractionalPathLength'],
        lw=1,
        color=color,
        zorder=2,
    )

    ticks = [0, .2, .4, .6, .8, 1]
    lim = [0, 1]

    ax.set_xlim(lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))

    ax.set_ylim(lim)
    ax.set_yticks(ticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))

    ax.set_xlabel('university prestige')
    ax.set_ylabel(r"$\frac{\mathrm{mean\ geodesic\ distance}}{\mathrm{diameter}}$")

    if annotate_rank_direction:
        add_gridlines_and_annotate_rank_direction(
            ax,
            rank_type='prestige',
            break_height=.075,
            x_gridlines_to_break=[.2, .4],
        )

    hnelib.plot.finalize(ax)

    return df


def plot_multiple_geodesics():
    df = pd.read_csv('/home/jwchen/code/NetworkML_USFHN/scripts/geodesics.csv')

    df = filter_to_academia_and_domain(df)
    df = annotate_color(df)

    fig, ax = plt.subplots(figsize=(3.50394, 2.5), tight_layout=True)

    dfs = []
    for level, rows in df.groupby('TaxonomyLevel'):
        rows = rows.copy()

        for value, _rows in rows.groupby('TaxonomyValue'):
            _df = plot_basic_geodesics(
                ax,
                df=_rows,
                level=level,
                value=value,
                color=_rows.iloc[0]['Color'],
                annotate_rank_direction=False,
            )
            dfs.append(_df)

    add_gridlines_and_annotate_rank_direction(
        ax,
        rank_type='prestige',
        break_height=.075,
    )

    ax.set_xticklabels([0, 20, 40, 60, 80, 100])

    # legend = plot_utils.add_umbrella_legend(
    #     ax,
    #     get_umbrella_legend_handles_kwargs={
    #         'style': 'none',
    #         'include_academia': True,
    #     },
    #     legend_kwargs={
    #         'fontsize': hnelib.plot.FONTSIZES['legend'],
    #         'loc': 'center left',
    #         'bbox_to_anchor': (.9, .5),
    #         'bbox_transform': ax.transAxes,
    #     },
    # )

    hnelib.plot.finalize(ax)

    data_df = pd.concat(dfs)
    data_df = annotate_color(data_df)
    data_df['Line'] = data_df['TaxonomyValue']

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    return pd.DataFrame(plot_data)