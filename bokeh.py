import sqlite3
from bokeh.plotting import figure
from bokeh.io import curdoc, vform
from bokeh.models import (
    ColumnDataSource,
    Select,
    Slider
)
from bokeh.models.glyphs import Patches
from bokeh.colors import RGB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile

def data_source(category='ROBBERY', year='2003'):
    """Returns ColumnDataSource for input crime category
    """
    query = """SELECT X, Y FROM crimes
               WHERE Category = ? AND Year = ?"""
    df = pd.read_sql_query(query, conn, params=(category, str(year)))
    cm = plt.cm.get_cmap('Spectral_r')
    H, x_edges, y_edges = np.histogram2d(df.X, df.Y, bins=50)
    H = (H - H.min()) / (H.max() - H.min())
    nx, ny = H.shape
    x = []
    y = []
    color = []
    for i in range(nx):
        for j in range(ny):
            if H[i,j] > 0.0:
                x.append([x_edges[i], x_edges[i+1], x_edges[i+1], 
                    x_edges[i]])
                y.append([y_edges[j], y_edges[j], y_edges[j+1], 
                    y_edges[j+1]])
                rgba = np.asarray(cm(H[i,j])) * 255
                c_hex = RGB(*rgba).to_hex()
                color.append(c_hex)
    return ColumnDataSource(dict(x=x, y=y, color=color))

def patch_dict(reader):
    p_dict = {}
    for sr in reader.shapeRecords():
        name = sr.record[1]
        x = []
        y = []
        points = sr.shape.points
        if len(sr.shape.parts) == 1:
            x.append([point[0] for point in points])
            y.append([point[1] for point in points])
        else:
            parts = sr.shape.parts
            parts.append(len(points))
            parts_idx = zip(parts[:-1], parts[1:])
            for start, stop in parts_idx:
                x.append([point[0] for point in points[start:stop]])
                y.append([point[1] for point in points[start:stop]])
        p_dict[name] = (x, y)
    return p_dict

def plot_patches(p_dict, fig, line_color, line_width=1):
    for patch in p_dict:
        x, y = p_dict[patch]
        fig.patches(x, y, color='#e8e8e8', line_width=line_width,
                    line_color=line_color, fill_alpha=0.5)
    return None

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
dat = shapefile.Reader('map/districts')

shp_dict = patch_dict(dat)
p = figure(width=800, height=800, background_fill='#686868')
plot_patches(shp_dict, p, 'white', 2)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.toolbar_location = None
p.logo = None
p.xaxis.visible = None
p.yaxis.visible = None

source = data_source()
p.patches(xs='x', ys='y', fill_color='color', fill_alpha=0.7,
        line_color=None, source=source)
#p.add_glyph(source, patches)

crime_select.on_change('value', update)
year_slider.on_change('value', update)

# Add to document
curdoc().add_root(vform(p, year_slider, crime_select))
