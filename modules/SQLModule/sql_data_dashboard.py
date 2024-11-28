# sql_data_dashboard.py
#
# Description: 
#  This module runs an interactive Dash dashboard for displaying and exploring
#   the data inside the SQL database. 
#
#  

import sqlalchemy as sa
from sqlalchemy.orm import  sessionmaker        
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os 

from SQLModule import db


def create_layout(df : pd.DataFrame, app) -> dbc.Container: 

    app.layout = dbc.Container([
            dbc.Row([
                html.H1("COMAP Manchester Pipeline Data Dashboard", className="text-center"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(children="Observation Type"),
                    #dcc.Dropdown(id='obs-type-selector', 
                    #    multi=True),
                ], width=6),
                dbc.Col([
                    html.Div(children="Observation Group"),
                    #dcc.Dropdown(id='obs-group-selector', 
                   #     multi=True),
                ], width=6), 
            ],justify='center'),
            html.Br(),
            dbc.Row([
                html.Div(children="Date Range"),
                #dcc.DatePickerRange(id='date-picker',
                #                    start_date=df['date'].min(),
                #                    end_date=df['date'].max())
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                #    dcc.Graph(id='pie-chart'),
                #    dcc.Checklist(options=['Highlight Groupings'], 
                #                inline=True,
                 #               id='pie-chart-options')
                ], width=6),
                dbc.Col([
                #    dcc.Graph(id='processed-data-histogram'),
                #    html.Button("Export CSV", id='btn-csv', style={'display': 'inline-block'}),
                    #html.Textbox(id='csv-filename', value='data.csv')
                ], width=6)
            ])
    ], fluid=True)

    return app.layout 

def create_app(df : pd.DataFrame) -> dash.Dash:

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  
    app.layout = create_layout(df, app)

    return app

def load_data(database_path: str) -> pd.DataFrame:

    db.connect(database_path)
    query = "SELECT obsid, level1_path, level2_path, source, source_group, utc_start FROM comap_data"
    df = pd.read_sql(query, db.database)
    df['date'] = pd.to_datetime(df['utc_start'], format='%Y-%m-%d-%H:%M:%S')
    df.sort_values('obsid', inplace=True)
    df['processed'] = df['level2_path'].notnull() 

    return df

import plotly.graph_objects as go

def create_processed_bar(df):
    total_rows = len(df[(df['source_group'] != 'SkyDip')])
    processed_rows = len(df[(df['processed'] == True) & (df['source_group'] != 'SkyDip')])
    
    fig = go.Figure()
    
    # Add total rows bar
    fig.add_trace(go.Bar(
        y=[total_rows],
        name='Total Rows',
        marker_color='lightgray'
    ))
    
    # Add processed rows bar
    fig.add_trace(go.Bar(
        y=[processed_rows],
        name='Processed Rows',
        marker_color='blue'
    ))
    
    fig.update_layout(
        barmode='overlay',
        title='Data Processing Status',
        yaxis_title='Number of Rows',
        showlegend=True
    )
    
    return fig

import plotly.graph_objects as go
import pandas as pd

def create_source_pie(df, start_date=None, end_date=None, date_column='date', processed=False):
    # Filter by date range if provided
    if start_date and end_date:
        mask = (df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))
        df = df[mask]

    if processed:
        df = df[df['processed'] == True]
    
    # Count observations per source group
    source_counts = df['source_group'].value_counts()
    
    # Create labels with counts
    labels = [f"{source} ({count})" for source, count in source_counts.items()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=source_counts.values,
        textposition='inside'
    )])
    
    # Add title with date range if applicable
    title = "Distribution of Source Groups"
    if start_date and end_date:
        title += f"<br>{start_date} to {end_date}"
    
    fig.update_layout(
        title=title,
        showlegend=True
    )
    
    return fig

# Example usage:
# fig = create_source_pie(df, '2024-01-01', '2024-03-31')
# fig.show()

# Example usage:
# fig = create_processed_bar(df)
# fig.show()
def main(): 
    figure_path = 'figures/DataAcquisition'
    os.makedirs(figure_path, exist_ok=True)
    database_path = 'databases/COMAP_manchester.db'
    df = load_data(database_path)


    fig = create_source_pie(df)
    fig.write_image(f"{figure_path}/source_pie_chart.png", format='png', engine='kaleido')

    fig = create_source_pie(df,processed=True)
    fig.write_image(f"{figure_path}/processed_source_pie_chart.png", format='png', engine='kaleido')

    fig = create_processed_bar(df)
    fig.write_image(f"{figure_path}/processed_bar_chart.png", format='png', engine='kaleido')

    #app = create_app(df)
    #app.run_server(debug=True)


if __name__ == "__main__":
    main() 