"""
Visualization utilities for Excel RAG analysis.
Generates charts and plots based on data analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import io
import base64

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Handles data visualization for Excel analysis results."""
    
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure plotly
        import plotly.io as pio
        pio.templates.default = "plotly_white"
    
    def create_summary_charts(self, data_summary: Dict, sheet_name: str = "Sheet") -> Dict[str, Any]:
        """Create summary charts for Excel data overview."""
        try:
            charts = {}
            
            # Data overview chart
            if "overall_summary" in data_summary:
                overall = data_summary["overall_summary"]
                
                # Sheet information
                if "sheets" in data_summary:
                    sheet_names = list(data_summary["sheets"].keys())
                    sheet_rows = [data_summary["sheets"][name]["summary"]["shape"][0] 
                                for name in sheet_names]
                    sheet_cols = [data_summary["sheets"][name]["summary"]["shape"][1] 
                                for name in sheet_names]
                    
                    # Create subplot for sheet overview
                    fig = sp.make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Rows per Sheet", "Columns per Sheet"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    fig.add_trace(
                        go.Bar(x=sheet_names, y=sheet_rows, name="Rows", marker_color='lightblue'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=sheet_names, y=sheet_cols, name="Columns", marker_color='lightcoral'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f"Data Overview - {sheet_name}",
                        showlegend=False,
                        height=400
                    )
                    
                    charts["overview"] = fig
            
            # Missing data visualization
            if "sheets" in data_summary:
                missing_data_charts = {}
                
                for sheet_name, sheet_data in data_summary["sheets"].items():
                    summary = sheet_data.get("summary", {})
                    missing_values = summary.get("missing_values", {})
                    
                    if missing_values:
                        # Create missing data heatmap
                        df = pd.DataFrame(list(missing_values.items()), 
                                        columns=['Column', 'Missing_Count'])
                        
                        fig = px.bar(df, x='Column', y='Missing_Count',
                                   title=f"Missing Data - {sheet_name}",
                                   color='Missing_Count',
                                   color_continuous_scale='Reds')
                        
                        fig.update_layout(height=400)
                        missing_data_charts[sheet_name] = fig
                
                charts["missing_data"] = missing_data_charts
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating summary charts: {str(e)}")
            return {}
    
    def create_trend_charts(self, df: pd.DataFrame, date_column: str = None) -> Dict[str, Any]:
        """Create trend analysis charts."""
        try:
            charts = {}
            
            # Auto-detect date column if not specified
            if date_column is None:
                date_columns = df.select_dtypes(include=['datetime64']).columns
                if len(date_columns) > 0:
                    date_column = date_columns[0]
            
            if date_column and date_column in df.columns:
                # Sort by date
                df_sorted = df.sort_values(date_column)
                
                # Time series for numerical columns
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numerical_cols[:3]:  # Limit to 3 columns
                    fig = px.line(df_sorted, x=date_column, y=col,
                                title=f"Trend Analysis - {col} over Time")
                    
                    # Add trend line
                    z = np.polyfit(range(len(df_sorted)), df_sorted[col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(x=df_sorted[date_column], 
                                 y=p(range(len(df_sorted))),
                                 mode='lines', name='Trend Line',
                                 line=dict(dash='dash', color='red'))
                    )
                    
                    fig.update_layout(height=400)
                    charts[f"trend_{col}"] = fig
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating trend charts: {str(e)}")
            return {}
    
    def create_correlation_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation analysis charts."""
        try:
            charts = {}
            
            # Select numerical columns
            numerical_df = df.select_dtypes(include=[np.number])
            
            if len(numerical_df.columns) < 2:
                return charts
            
            # Correlation matrix heatmap
            corr_matrix = numerical_df.corr()
            
            fig = px.imshow(corr_matrix,
                           title="Correlation Matrix",
                           color_continuous_scale='RdBu',
                           aspect="auto")
            
            fig.update_layout(height=500)
            charts["correlation_matrix"] = fig
            
            # Scatter plot matrix for top correlations
            if len(numerical_df.columns) <= 6:
                fig = px.scatter_matrix(numerical_df,
                                      title="Scatter Plot Matrix")
                fig.update_layout(height=600)
                charts["scatter_matrix"] = fig
            
            # Top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Create top correlations chart
            if corr_pairs:
                top_corrs = corr_pairs[:10]
                fig = px.bar(
                    x=[f"{pair['var1']} vs {pair['var2']}" for pair in top_corrs],
                    y=[pair['correlation'] for pair in top_corrs],
                    title="Top Correlations",
                    color=[abs(pair['correlation']) for pair in top_corrs],
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=400)
                charts["top_correlations"] = fig
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating correlation charts: {str(e)}")
            return {}
    
    def create_anomaly_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create anomaly detection charts."""
        try:
            charts = {}
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols[:3]:  # Limit to 3 columns
                # Box plot for outlier detection
                fig = px.box(df, y=col, title=f"Outlier Detection - {col}")
                fig.update_layout(height=400)
                charts[f"boxplot_{col}"] = fig
                
                # Histogram with outlier highlighting
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                fig = px.histogram(df, x=col, title=f"Distribution with Outliers - {col}")
                
                if len(outliers) > 0:
                    fig.add_trace(
                        go.Scatter(x=outliers[col], y=[0]*len(outliers),
                                 mode='markers', name='Outliers',
                                 marker=dict(color='red', size=8))
                    )
                
                fig.update_layout(height=400)
                charts[f"histogram_{col}"] = fig
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating anomaly charts: {str(e)}")
            return {}
    
    def create_distribution_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create data distribution charts."""
        try:
            charts = {}
            
            # Numerical distributions
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols[:4]:  # Limit to 4 columns
                fig = px.histogram(df, x=col, title=f"Distribution - {col}")
                fig.update_layout(height=400)
                charts[f"distribution_{col}"] = fig
            
            # Categorical distributions
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols[:3]:  # Limit to 3 columns
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top Values - {col}")
                fig.update_layout(height=400)
                charts[f"categorical_{col}"] = fig
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating distribution charts: {str(e)}")
            return {}
    
    def create_custom_chart(self, chart_type: str, data: Any, **kwargs) -> Optional[go.Figure]:
        """Create a custom chart based on type and data."""
        try:
            if chart_type == "line":
                return px.line(data, **kwargs)
            elif chart_type == "bar":
                return px.bar(data, **kwargs)
            elif chart_type == "scatter":
                return px.scatter(data, **kwargs)
            elif chart_type == "histogram":
                return px.histogram(data, **kwargs)
            elif chart_type == "box":
                return px.box(data, **kwargs)
            elif chart_type == "heatmap":
                return px.imshow(data, **kwargs)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating custom chart: {str(e)}")
            return None
    
    def save_chart_as_image(self, fig: go.Figure, file_path: str, format: str = "png") -> bool:
        """Save a Plotly chart as an image file."""
        try:
            file_path = Path(file_path)
            
            if format.lower() == "png":
                fig.write_image(str(file_path.with_suffix('.png')))
            elif format.lower() == "jpg":
                fig.write_image(str(file_path.with_suffix('.jpg')))
            elif format.lower() == "svg":
                fig.write_image(str(file_path.with_suffix('.svg')))
            elif format.lower() == "pdf":
                fig.write_image(str(file_path.with_suffix('.pdf')))
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Chart saved as {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
            return False
    
    def chart_to_html(self, fig: go.Figure) -> str:
        """Convert a Plotly chart to HTML string."""
        try:
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        except Exception as e:
            logger.error(f"Error converting chart to HTML: {str(e)}")
            return f"<p>Error displaying chart: {str(e)}</p>"
    
    def create_dashboard(self, charts: Dict[str, Any], title: str = "Data Analysis Dashboard") -> str:
        """Create an HTML dashboard from multiple charts."""
        try:
            html_parts = [
                f"<html><head><title>{title}</title></head><body>",
                f"<h1>{title}</h1>"
            ]
            
            for chart_name, chart in charts.items():
                if hasattr(chart, 'to_html'):
                    html_parts.append(f"<h2>{chart_name.replace('_', ' ').title()}</h2>")
                    html_parts.append(chart.to_html(include_plotlyjs='cdn', full_html=False))
                    html_parts.append("<hr>")
            
            html_parts.append("</body></html>")
            
            return "\n".join(html_parts)
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def get_chart_recommendations(self, data_summary: Dict) -> List[str]:
        """Get recommendations for appropriate chart types based on data."""
        try:
            recommendations = []
            
            if "sheets" in data_summary:
                for sheet_name, sheet_data in data_summary["sheets"].items():
                    summary = sheet_data.get("summary", {})
                    
                    # Check data types
                    dtypes = summary.get("dtypes", {})
                    categorical_count = sum(1 for dtype in dtypes.values() 
                                         if str(dtype) in ['object', 'category'])
                    numerical_count = sum(1 for dtype in dtypes.values() 
                                       if str(dtype) in ['int64', 'float64'])
                    
                    if numerical_count > 1:
                        recommendations.append(f"Use correlation heatmaps for {sheet_name} to show relationships between numerical variables")
                    
                    if categorical_count > 0:
                        recommendations.append(f"Use bar charts for {sheet_name} to show frequency distributions of categorical variables")
                    
                    # Check for time series
                    if any("datetime" in str(dtype) for dtype in dtypes.values()):
                        recommendations.append(f"Use line charts for {sheet_name} to show trends over time")
                    
                    # Check data size
                    shape = summary.get("shape", (0, 0))
                    if shape[0] > 1000:
                        recommendations.append(f"Consider sampling data from {sheet_name} for better visualization performance")
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Error getting chart recommendations: {str(e)}")
            return ["Consider the data types and structure when choosing visualizations"] 