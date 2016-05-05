import sqlite3
from bokeh.io import curdoc
from bokeh.models import (
    GMapPlot,
    GMapOptions,
    ColumnDataSource,
    DataRange1d,
    Select,
    Slider,
    HBox,
    VBoxForm
)
from bokeh.models.glyphs import Patches
from bokeh.colors import RGB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def data_source(category='ROBBERY', year='2003'):
    """Returns ColumnDataSource for input crime category
    """
    query = """SELECT X, Y FROM crimes
               WHERE Category = ? AND Year = ?"""
    df = pd.read_sql_query(query, conn, params=(category, str(year)))
    cm = plt.cm.get_cmap('Spectral_r')
    H, x_edges, y_edges = np.histogram2d(df.X, df.Y, bins=100)
    H = (H - H.min()) / (H.max() - H.min())
    nx, ny = H.shape
    x = []
    y = []
    color = []
    for i in range(nx):
        for j in range(ny):
            if H[i,j] > 0.0:
                x.append([x_edges[i], x_edges[i+1], x_edges[i+1], x_edges[i]])
                y.append([y_edges[j], y_edges[j], y_edges[j+1], y_edges[j+1]])
                rgba = np.asarray(cm(H[i,j])) * 255
                c_hex = RGB(*rgba).to_hex()
                color.append(c_hex)
    return ColumnDataSource(dict(x=x, y=y, color=color))

def make_plot(source):
    map_options = GMapOptions(
        lat=37.7678631,
        lng=-122.4333172,
        map_type='roadmap',
        zoom=13
    )
    plot = GMapPlot(
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        map_options=map_options,
        plot_width=800,
        plot_height=800,
        min_border_left=0,
        min_border_right=0,
        min_border_top=0,
        min_border_bottom=0,
        toolbar_location=None
    )
    patches = Patches(xs='x', ys='y', fill_color='color', fill_alpha=0.7,
        line_color=None)
    plot.add_glyph(source, patches)
    return plot

def update(attrname, old, new):
    crime = crime_select.value
    year = year_slider.value
    src = data_source(crime, year)
    source.data.update(src.data)

conn = sqlite3.connect('data/sf_crimes.sqlite')
c = conn.cursor()

# Make the select tool with default value set to 'ROBBERY'
options = list(map(lambda tup: tup[0],
    c.execute('SELECT DISTINCT Category FROM crimes').fetchall()))
for opt in options: print(opt)
crime_select = Select(value='ROBBERY', title='Crime type', options=options)

# Make the slider tool which chooses the year
year_slider = Slider(title='Year', start=2003, end=2015, step=1, value=2003)

# Make the plot
source = data_source()
plot = make_plot(source)

crime_select.on_change('value', update)
year_slider.on_change('value', update)

# Add to document
inputs = HBox(VBoxForm(year_slider, crime_select))
curdoc().add_root(HBox(inputs, plot))
