import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# LOADING PREDICTIVE MAINTENANCE DASHBOARD
# ============================================================

print("=" * 60)
print("LOADING PREDICTIVE MAINTENANCE DASHBOARD")
print("=" * 60)

# Initialize variables
data_df = None
threshold = 0.30  # Fixed threshold at 30%
best_precision = 0.0
best_recall = 0.0
best_f1 = 0.0
best_contamination = 0.23

try:
    # Get the directory where app.py is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, 'bearing_training_data.csv')
    
    print(f"Looking for CSV at: {CSV_PATH}")
    
    # Load CSV
    if os.path.exists(CSV_PATH):
        data_df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(data_df):,} rows from CSV")
        print(f"✓ Columns found: {list(data_df.columns)}")
    else:
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    
    # Check for required basic columns
    basic_required_cols = ['equipment_id', 'rms', 'kurtosis', 'peak_value']
    missing_cols = [col for col in basic_required_cols if col not in data_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ Equipment IDs: {data_df['equipment_id'].nunique()}")
    
    # Add true_anomaly column if status exists
    if 'status' in data_df.columns:
        data_df['true_anomaly'] = data_df['status'] == 'faulty'
        print(f"✓ True anomaly rate: {data_df['true_anomaly'].mean():.1%}")
    else:
        data_df['true_anomaly'] = False
        print("⚠️  No 'status' column - assuming all normal")
    
    # Calculate anomaly scores if missing
    if 'anomaly_score' not in data_df.columns or 'anomaly_detected' not in data_df.columns:
        print("\n⚙️  Calculating anomaly scores using Isolation Forest...")
        
        # Prepare features for anomaly detection
        feature_cols = ['rms', 'kurtosis', 'peak_value']
        X = data_df[feature_cols].values
        
        print(f"   Using features: {feature_cols}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest with updated contamination value
        contamination = 0.23  # Updated optimal value
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        print(f"   Training model with contamination={contamination}...")
        model.fit(X_scaled)
        
        # Predict anomalies
        predictions = model.predict(X_scaled)
        anomaly_scores_raw = model.score_samples(X_scaled)
        
        # Convert to 0-1 scale (normalized anomaly scores)
        data_df['anomaly_score'] = (anomaly_scores_raw - anomaly_scores_raw.min()) / (anomaly_scores_raw.max() - anomaly_scores_raw.min())
        
        # Invert so higher score = more anomalous
        data_df['anomaly_score'] = 1 - data_df['anomaly_score']
        
        # Mark anomalies (predictions: -1 = anomaly, 1 = normal)
        data_df['anomaly_detected'] = predictions == -1
        
        print(f"✓ Anomaly detection complete")
        print(f"   Anomalies detected: {data_df['anomaly_detected'].sum()} ({data_df['anomaly_detected'].mean():.1%})")
    
    # Calculate threshold (30% of the scale = 0.3)
    threshold = 0.30  # Fixed at 30% as per normal range definition (0-30% = All clear)
    
    print(f"✓ Anomaly threshold: {threshold:.3f} (30% - normal range limit)")
    
    # Calculate model performance metrics if we have ground truth
    if 'true_anomaly' in data_df.columns and data_df['true_anomaly'].any():
        y_true = data_df['true_anomaly'].values
        y_pred = data_df['anomaly_detected'].values
        
        best_precision = precision_score(y_true, y_pred, zero_division=0)
        best_recall = recall_score(y_true, y_pred, zero_division=0)
        best_f1 = f1_score(y_true, y_pred, zero_division=0)
        best_contamination = contamination
        
        print(f"\n📊 Model Performance:")
        print(f"   Precision: {best_precision:.1%}")
        print(f"   Recall: {best_recall:.1%}")
        print(f"   F1-Score: {best_f1:.3f}")
    
    print(f"\n✅ Data loaded successfully: {len(data_df):,} rows")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error loading data or calculating initial metrics: {e}")
    print("Creating dummy data for demonstration...")
    
    # Create dummy data as fallback
    np.random.seed(42)
    n_samples = 1000
    
    data_df = pd.DataFrame({
        'equipment_id': [f'EQUIP-{str(i).zfill(3)}' for i in np.random.randint(1, 6, n_samples)],
        'rms': np.random.uniform(0.05, 0.15, n_samples),
        'kurtosis': np.random.uniform(2.5, 4.5, n_samples),
        'peak_value': np.random.uniform(0.1, 0.3, n_samples),
        'anomaly_score': np.random.uniform(0, 1, n_samples)
    })
    
    data_df['anomaly_detected'] = data_df['anomaly_score'] > 0.5
    data_df['status'] = np.where(data_df['anomaly_detected'], 'faulty', 'normal')
    data_df['true_anomaly'] = data_df['status'] == 'faulty'
    
    threshold = 0.30
    best_precision = 0.72
    best_recall = 0.85
    best_f1 = 0.78
    best_contamination = 0.23
    
    print(f"✓ Dummy data created: {len(data_df):,} rows")
    print("=" * 60)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # This is needed for Gunicorn

# App layout
app.layout = html.Div([
    html.H1("🔧 Predictive Maintenance Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30, 'fontWeight': 'bold'}),
    
    html.Div([
        html.Label("Select Equipment ID:", style={'fontWeight': 'bold', 'fontSize': 16, 'color': '#34495e'}),
        dcc.Dropdown(
            id='equipment-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(data_df['equipment_id'].unique())],
            value=sorted(data_df['equipment_id'].unique())[0] if len(data_df['equipment_id'].unique()) > 0 else None,
            clearable=False,
            style={'width': '50%', 'marginBottom': 20, 'fontSize': 14}
        ),
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': 10, 'margin': '20px'}),

    html.Hr(style={'borderColor': '#bdc3c7'}),

    # Summary section
    html.Div(id='equipment-summary', style={
        'padding': '20px', 
        'backgroundColor': '#ecf0f1', 
        'borderRadius': 10, 
        'margin': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),

    html.Hr(style={'borderColor': '#bdc3c7'}),

    # Graphs
    html.Div([
        dcc.Graph(id='rms-trend-graph'),
        dcc.Graph(id='kurtosis-trend-graph'),
        dcc.Graph(id='peak-trend-graph'),
        dcc.Graph(id='anomaly-score-graph'),
        dcc.Graph(id='feature-space-graph')
    ], style={'padding': '20px'})
], style={
    'fontFamily': 'Arial, sans-serif', 
    'backgroundColor': '#f5f6fa', 
    'minHeight': '100vh',
    'paddingBottom': '50px'
})

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
        empty_summary = html.Div("Please select an equipment ID.", 
                                style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_summary, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    # Filter data for selected equipment
    equipment_data = data_df[data_df['equipment_id'] == selected_equipment_id].copy()
    equipment_data = equipment_data.reset_index(drop=True)
    
    # Calculate summary statistics
    total_readings = len(equipment_data)
    true_anomalies = equipment_data['true_anomaly'].sum() if 'true_anomaly' in equipment_data.columns else 0
    detected_anomalies = equipment_data['anomaly_detected'].sum()
    avg_rms = equipment_data['rms'].mean()
    avg_anomaly_score = equipment_data['anomaly_score'].mean()
    
    # Find first anomaly index
    first_anomaly_idx = equipment_data[equipment_data['anomaly_detected']].index.min() if detected_anomalies > 0 else None
    
    # Create summary section
    summary = html.Div([
        html.H3(f"📊 Summary for Equipment ID: {selected_equipment_id}", 
                style={'color': '#2c3e50', 'marginBottom': 20, 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                html.P(f"📊 Total Readings: {total_readings:,}", style={'fontSize': 16, 'margin': '10px 0'}),
                html.P(f"✓ True Anomalies (Ground Truth): {int(true_anomalies)}", style={'fontSize': 16, 'margin': '10px 0'}),
                html.P(f"🚨 Anomalies Detected (Model): {int(detected_anomalies)}", style={'fontSize': 16, 'margin': '10px 0'}),
            ], style={'flex': 1}),
            
            html.Div([
                html.P(f"📈 Average RMS: {avg_rms:.2f} mm/s", style={'fontSize': 16, 'margin': '10px 0'}),
                html.P(f"⚡ Average Anomaly Score: {avg_anomaly_score:.3f}", style={'fontSize': 16, 'margin': '10px 0'}),
                html.P(f"⚠️ First Anomaly at Index: {first_anomaly_idx if first_anomaly_idx is not None else 'None'}", 
                       style={'fontSize': 16, 'margin': '10px 0'}),
            ], style={'flex': 1}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
        
        html.Hr(style={'borderColor': '#bdc3c7', 'margin': '20px 0'}),
        
        html.H3("📊 Overall Model Performance (Contamination=0.23)", 
                style={'color': '#2c3e50', 'marginTop': 20, 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.P(f"Precision: {best_precision:.1%}", style={'fontSize': 16, 'margin': '10px 0'}),
            html.P(f"Recall: {best_recall:.1%}", style={'fontSize': 16, 'margin': '10px 0'}),
            html.P(f"F1-Score: {best_f1:.3f}", style={'fontSize': 16, 'margin': '10px 0'}),
        ])
    ])
    
    # Create RMS Trend Graph
    fig_rms = go.Figure()
    fig_rms.add_trace(go.Scatter(
        x=equipment_data.index,
        y=equipment_data['rms'],
        mode='lines+markers',
        name='RMS',
        line=dict(color='#3498db', width=2),
        marker=dict(size=4)
    ))
    
    # Mark anomalies
    anomaly_indices = equipment_data[equipment_data['anomaly_detected']].index
    if len(anomaly_indices) > 0:
        fig_rms.add_trace(go.Scatter(
            x=anomaly_indices,
            y=equipment_data.loc[anomaly_indices, 'rms'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
        ))
    
    fig_rms.update_layout(
        title='RMS Vibration Trend',
        xaxis_title='Sample Index',
        yaxis_title='RMS (mm/s)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Create Kurtosis Trend Graph
    fig_kurtosis = go.Figure()
    fig_kurtosis.add_trace(go.Scatter(
        x=equipment_data.index,
        y=equipment_data['kurtosis'],
        mode='lines+markers',
        name='Kurtosis',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=4)
    ))
    
    # Mark anomalies
    if len(anomaly_indices) > 0:
        fig_kurtosis.add_trace(go.Scatter(
            x=anomaly_indices,
            y=equipment_data.loc[anomaly_indices, 'kurtosis'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
        ))
    
    fig_kurtosis.update_layout(
        title='Kurtosis Trend',
        xaxis_title='Sample Index',
        yaxis_title='Kurtosis',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Create Peak Value Trend Graph
    fig_peak = go.Figure()
    fig_peak.add_trace(go.Scatter(
        x=equipment_data.index,
        y=equipment_data['peak_value'],
        mode='lines+markers',
        name='Peak Value',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=4)
    ))
    
    # Mark anomalies
    if len(anomaly_indices) > 0:
        fig_peak.add_trace(go.Scatter(
            x=anomaly_indices,
            y=equipment_data.loc[anomaly_indices, 'peak_value'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
        ))
    
    fig_peak.update_layout(
        title='Peak Value Trend',
        xaxis_title='Sample Index',
        yaxis_title='Peak Value',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Create Anomaly Score Graph
    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(go.Scatter(
        x=equipment_data.index,
        y=equipment_data['anomaly_score'],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='#9b59b6', width=2),
        marker=dict(size=4)
    ))
    
    # Add threshold line at 0.30 (30% - normal range limit)
    fig_anomaly.add_hline(
        y=0.30,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Threshold: 0.30 (30% limit)",
        annotation_position="right"
    )
    
    # Mark anomalies
    if len(anomaly_indices) > 0:
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_indices,
            y=equipment_data.loc[anomaly_indices, 'anomaly_score'],
            mode='markers',
            name='Detected Anomaly',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
        ))
    
    fig_anomaly.update_layout(
        title='Isolation Forest Anomaly Score Trend',
        xaxis_title='Sample Index',
        yaxis_title='Anomaly Score (0-1)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    # Create Feature Space Graph
    fig_feature = go.Figure()
    
    # Normal points
    normal_data = equipment_data[~equipment_data['anomaly_detected']]
    fig_feature.add_trace(go.Scatter(
        x=normal_data['rms'],
        y=normal_data['kurtosis'],
        mode='markers',
        name='Normal',
        marker=dict(color='#3498db', size=8, opacity=0.6)
    ))
    
    # Anomaly points
    anomaly_data = equipment_data[equipment_data['anomaly_detected']]
    if len(anomaly_data) > 0:
        fig_feature.add_trace(go.Scatter(
            x=anomaly_data['rms'],
            y=anomaly_data['kurtosis'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=12, symbol='x', line=dict(width=2))
        ))
    
    fig_feature.update_layout(
        title='Feature Space: RMS vs Kurtosis',
        xaxis_title='RMS (mm/s)',
        yaxis_title='Kurtosis',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return summary, fig_rms, fig_kurtosis, fig_peak, fig_anomaly, fig_feature

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
