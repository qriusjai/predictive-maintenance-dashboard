import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# Load the bearing data
# In a real application, you might load this from a database or cloud storage
try:
    # Check if running in an environment where the original df is available (like the notebook)
    # If not, load from CSV. Assumes bearing_training_data.csv is available.
    if 'df' in globals():
        data_df = df.copy()
        print("Using global 'df' DataFrame.")
    else:
        # Assume the CSV is in the same directory as the app.py script
        data_df = pd.read_csv('bearing_training_data.csv')
        print("Loaded data from bearing_training_data.csv.")

    # Ensure necessary columns exist
    required_cols = ['equipment_id', 'rms', 'kurtosis', 'peak_value', 'anomaly_score', 'anomaly_detected', 'true_anomaly', 'status']
    for col in required_cols:
        if col not in data_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure true_anomaly is boolean
    data_df['true_anomaly'] = data_df['status'] == 'faulty'

    # Recalculate threshold based on the anomaly scores of normal data
    # This might be calculated during model training in a production setup
    normal_anomaly_scores = data_df[data_df['true_anomaly'] == False]['anomaly_score']
    if not normal_anomaly_scores.empty:
        threshold = np.percentile(normal_anomaly_scores, 85)
    else:
        threshold = 0.5 # Default threshold if no normal data

    # Use overall metrics from the previously run code if available, otherwise use placeholders
    # In a real production setup, these would come from saved model evaluation results
    best_precision = globals().get('best_precision', 0.0)
    best_recall = globals().get('best_recall', 0.0)
    best_f1 = globals().get('best_f1', 0.0)
    best_contamination = globals().get('best_contamination', 0.0)


except Exception as e:
    print(f"Error loading data or calculating initial metrics: {e}")
    print("Creating a dummy DataFrame and using default metrics.")
    # Create a dummy DataFrame as a fallback
    data = {
        'equipment_id': [f'EQUIP-{i:03d}' for i in range(1, 37)] * 300,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=300*36, freq='10s')),
        'rms': np.random.rand(300*36) * 10 + np.sin(np.linspace(0, 100, 300*36)) * 2 + np.random.randn(300*36),
        'kurtosis': np.random.rand(300*36) * 5 + 2 + np.cos(np.linspace(0, 100, 300*36)) * 1 + np.random.randn(300*36)*0.5,
        'peak_value': np.random.rand(300*36) * 20 + np.sin(np.linspace(0, 100, 300*36)) * 4 + np.random.randn(300*36)*2,
        'anomaly_score': np.random.rand(300*36),
        'anomaly_detected': np.random.choice([True, False], 300*36, p=[0.03, 0.97]),
        'status': ['normal'] * int(300*36*0.4) + ['faulty'] * int(300*36*0.6) + ['normal'] * (300*36 - int(300*36*0.4) - int(300*36*0.6)),
    }
    data_df = pd.DataFrame(data)
    data_df['true_anomaly'] = data_df['status'] == 'faulty'
    # Adjust dummy anomaly_detected to align somewhat with 'faulty' status for demonstration
    data_df.loc[data_df['true_anomaly'], 'anomaly_detected'] = np.random.choice([True, False], size=data_df['true_anomaly'].sum(), p=[0.7, 0.3])
    data_df.loc[~data_df['true_anomaly'], 'anomaly_detected'] = np.random.choice([True, False], size=(~data_df['true_anomaly']).sum(), p=[0.05, 0.95])
    # Recalculate threshold for dummy data
    normal_anomaly_scores = data_df[data_df['true_anomaly'] == False]['anomaly_score']
    if not normal_anomaly_scores.empty:
        threshold = np.percentile(normal_anomaly_scores, 85)
    else:
        threshold = 0.5 # Default threshold if no normal data

    best_precision = 0.793 # Example metric
    best_recall = 0.463   # Example metric
    best_f1 = 0.584       # Example metric
    best_contamination = 0.33 # Example metric


# Initialize the Dash app
# For deployment, __name__ is standard. server is needed for Gunicorn.
app = dash.Dash(__name__)
server = app.server # This is needed for Gunicorn or other WSGI servers

# App layout
app.layout = html.Div([
    html.H1("Predictive Maintenance Dashboard"),

    html.Label("Select Equipment ID:"),
    dcc.Dropdown(
        id='equipment-dropdown',
        options=[{'label': i, 'value': i} for i in data_df['equipment_id'].unique()],
        value=data_df['equipment_id'].unique()[0] if len(data_df['equipment_id'].unique()) > 0 else None,
        clearable=False,
        style={'width': '50%'}
    ),

    html.Hr(), # Horizontal line for separation

    # Div to display summary text
    html.Div(id='equipment-summary', style={'padding': '10px 0'}),

    html.Hr(), # Horizontal line for separation

    # Div to display plots
    html.Div([
        dcc.Graph(id='rms-trend-graph'),
        dcc.Graph(id='kurtosis-trend-graph'),
        dcc.Graph(id='peak-trend-graph'),
        dcc.Graph(id='anomaly-score-graph'),
        dcc.Graph(id='feature-space-graph') # Add feature space plot
    ])
])

# Define callback to update graphs and summary
@app.callback(
    [Output('equipment-summary', 'children'),
     Output('rms-trend-graph', 'figure'),
     Output('kurtosis-trend-graph', 'figure'),
     Output('peak-trend-graph', 'figure'),
     Output('anomaly-score-graph', 'figure'),
     Output('feature-space-graph', 'figure')],
    [Input('equipment-dropdown', 'value')]
)
def update_dashboard(selected_equipment_id):
    if selected_equipment_id is None:
        # Return empty figures and default message if no equipment is selected
        empty_summary = html.Div("Please select an equipment ID.")
        empty_fig = go.Figure()
        return empty_summary, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    filtered_df = data_df[data_df['equipment_id'] == selected_equipment_id].reset_index(drop=True)

    if len(filtered_df) == 0:
        # Return empty figures and error message if no data is found
        error_summary = html.Div(f"No data found for equipment ID: {selected_equipment_id}")
        empty_fig = go.Figure()
        return error_summary, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Generate summary text
    total_readings = len(filtered_df)
    anomalies_detected_count = filtered_df['anomaly_detected'].sum()
    true_anomalies_count = filtered_df['true_anomaly'].sum() if 'true_anomaly' in filtered_df.columns else "N/A"

    summary_parts = [
        html.H3(f"Summary for Equipment ID: {selected_equipment_id}"),
        html.P(f"Total Readings: {total_readings:,}"),
        html.P(f"True Anomalies (Ground Truth): {true_anomalies_count:,}" if isinstance(true_anomalies_count, int) else f"True Anomalies (Ground Truth): {true_anomalies_count}"),
        html.P(f"Anomalies Detected (Model): {anomalies_detected_count:,}"),
    ]

    if anomalies_detected_count > 0:
        summary_parts.append(html.H4("Anomaly Detection Details:"))
        first_detection_idx = filtered_df[filtered_df['anomaly_detected']].index.min()
        summary_parts.append(html.P(f"First Anomaly Detected at Sample Index: {first_detection_idx:,}"))

        if true_anomalies_count > 0 and isinstance(true_anomalies_count, int):
             first_fault_idx = filtered_df[filtered_df['true_anomaly']].index.min()
             summary_parts.append(html.P(f"First True Fault at Sample Index: {first_fault_idx:,}"))
             if first_detection_idx is not None and first_fault_idx is not None and first_detection_idx < first_fault_idx:
                 samples_early = first_fault_idx - first_detection_idx
                 # Assuming 100 samples per hour
                 hours_early = samples_early / 100
                 summary_parts.append(html.P(f"Early Warning Time: {hours_early:.1f} hours before fault"))
             elif first_detection_idx is not None and first_fault_idx is not None and first_detection_idx == first_fault_idx:
                  summary_parts.append(html.P(f"Anomaly Detected at the same time as first true fault"))
             elif first_detection_idx is not None and first_fault_idx is not None and first_detection_idx > first_fault_idx:
                  summary_parts.append(html.P(f"Anomaly Detected after first true fault"))
             else:
                 summary_parts.append(html.P(f"Could not determine early warning time."))

        elif true_anomalies_count == 0:
             summary_parts.append(html.P(f"No true faults recorded for this equipment."))

    else:
        summary_parts.append(html.P("No anomalies detected by the model for this equipment."))

    # Use overall metrics if available
    if best_precision > 0 or best_recall > 0 or best_f1 > 0:
         summary_parts.append(html.H4(f"Overall Model Performance (Contamination={best_contamination:.2f}):"))
         summary_parts.append(html.P(f"Precision: {best_precision:.1%}"))
         summary_parts.append(html.P(f"Recall: {best_recall:.1%}"))
         summary_parts.append(html.P(f"F1-Score: {best_f1:.3f}"))


    summary_text_div = html.Div(summary_parts)


    # Create time index
    time_index = list(range(len(filtered_df)))

    # Plot 1: RMS Trend
    rms_fig = go.Figure()
    rms_fig.add_trace(go.Scattergl(x=time_index, y=filtered_df['rms'], mode='lines', name='RMS Vibration'))
    rms_fig.add_hline(y=8.0, line_dash="dash", line_color="orange", annotation_text="Warning (8 mm/s)", annotation_position="bottom right")
    rms_fig.add_hline(y=15.0, line_dash="dash", line_color="red", annotation_text="Critical (15 mm/s)", annotation_position="bottom right")
    anomaly_points_rms = filtered_df[filtered_df['anomaly_detected']]
    rms_fig.add_trace(go.Scattergl(x=anomaly_points_rms.index, y=anomaly_points_rms['rms'],
                                  mode='markers', name='Anomaly Detected', marker=dict(color='red', size=10)))
    rms_fig.update_layout(title='RMS Vibration Trend', xaxis_title='Sample Index (Time)', yaxis_title='RMS (mm/s)', height=300) # Set height

    # Plot 2: Kurtosis Trend
    kurtosis_fig = go.Figure()
    kurtosis_fig.add_trace(go.Scattergl(x=time_index, y=filtered_df['kurtosis'], mode='lines', name='Kurtosis', line=dict(color='green')))
    kurtosis_fig.add_hline(y=4.0, line_dash="dash", line_color="orange", annotation_text="Normal Limit", annotation_position="bottom right")
    anomaly_points_kurtosis = filtered_df[filtered_df['anomaly_detected']]
    kurtosis_fig.add_trace(go.Scattergl(x=anomaly_points_kurtosis.index, y=anomaly_points_kurtosis['kurtosis'],
                                        mode='markers', name='Anomaly Detected', marker=dict(color='red', size=10)))
    kurtosis_fig.update_layout(title='Kurtosis Trend', xaxis_title='Sample Index (Time)', yaxis_title='Kurtosis', height=300) # Set height


    # Plot 3: Peak Value Trend
    peak_fig = go.Figure()
    peak_fig.add_trace(go.Scattergl(x=time_index, y=filtered_df['peak_value'], mode='lines', name='Peak Value', line=dict(color='orange')))
    anomaly_points_peak = filtered_df[filtered_df['anomaly_detected']]
    peak_fig.add_trace(go.Scattergl(x=anomaly_points_peak.index, y=anomaly_points_peak['peak_value'],
                                   mode='markers', name='Anomaly Detected', marker=dict(color='red', size=10)))
    peak_fig.update_layout(title='Peak Value Trend', xaxis_title='Sample Index (Time)', yaxis_title='Peak Value', height=300) # Set height


    # Plot 4: Anomaly Score Trend
    anomaly_score_fig = go.Figure()
    anomaly_score_fig.add_trace(go.Scattergl(x=time_index, y=filtered_df['anomaly_score'], mode='lines', name='Anomaly Score', line=dict(color='purple')))
    anomaly_score_fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f'Threshold ({threshold:.3f})', annotation_position="bottom right")
    anomaly_points_score = filtered_df[filtered_df['anomaly_detected']]
    anomaly_score_fig.add_trace(go.Scattergl(x=anomaly_points_score.index, y=anomaly_points_score['anomaly_score'],
                                            mode='markers', name='Anomaly Detected', marker=dict(color='red', size=10)))
    anomaly_score_fig.update_layout(title='Isolation Forest Anomaly Score Trend', xaxis_title='Sample Index (Time)', yaxis_title='Anomaly Score', yrange=[0, 1], height=300) # Set height


    # Plot 5: Feature Space (RMS vs Kurtosis)
    feature_space_fig = go.Figure()
    normal_data = filtered_df[filtered_df['anomaly_detected'] == False]
    anomaly_data = filtered_df[filtered_df['anomaly_detected'] == True]

    feature_space_fig.add_trace(go.Scattergl(x=normal_data['rms'], y=normal_data['kurtosis'],
                                           mode='markers', name='Normal', marker=dict(color='green', opacity=0.5)))
    feature_space_fig.add_trace(go.Scattergl(x=anomaly_data['rms'], y=anomaly_data['kurtosis'],
                                           mode='markers', name='Anomaly Detected', marker=dict(color='red', opacity=0.8, size=8)))

    feature_space_fig.update_layout(title='Feature Space: RMS vs Kurtosis', xaxis_title='RMS Vibration (mm/s)', yaxis_title='Kurtosis', height=400) # Set height


    return summary_text_div, rms_fig, kurtosis_fig, peak_fig, anomaly_score_fig, feature_space_fig


if __name__ == '__main__':
    # Use debug=True for development
    # In production, set debug=False
    app.run_server(debug=True)
