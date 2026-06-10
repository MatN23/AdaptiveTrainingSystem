# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
from datetime import datetime
from typing import List
from utils.data_processing import validate_data_comprehensive


def create_data_summary_report(data_paths: List[str], tokenizer, 
                              output_path: str = "data_summary_report.html"):
    """Create comprehensive HTML report of dataset analysis."""
    
    # FIXED: Use double curly braces {{ }} to escape braces in CSS
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }}
            .error {{ color: red; }}
            .warning {{ color: orange; }}
            .success {{ color: green; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Dataset Analysis Report</h1>
        <p>Generated on: {timestamp}</p>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for data_path in data_paths:
        logging.info(f"Analyzing {data_path}...")
        stats = validate_data_comprehensive(data_path, tokenizer)
        
        html_content += f"""
        <div class="section">
            <h2>Dataset: {os.path.basename(data_path)}</h2>
            
            <h3>File Information</h3>
            <div class="metric">Size: {stats['file_info'].get('size_mb', 0):.1f} MB</div>
            <div class="metric">Modified: {stats['file_info'].get('modified', 'Unknown')}</div>
            
            <h3>Conversation Statistics</h3>
            <div class="metric">Total Lines: {stats['conversation_stats'].get('total_lines', 0):,}</div>
            <div class="metric">Valid Conversations: {stats['conversation_stats'].get('valid_conversations', 0):,}</div>
            <div class="metric">Success Rate: {stats['quality_metrics'].get('success_rate', 0):.2%}</div>
            
            <h3>Token Statistics</h3>
            <div class="metric">Avg Tokens: {stats['token_stats'].get('avg_tokens', 0):.1f}</div>
            <div class="metric">Max Tokens: {stats['token_stats'].get('max_tokens', 0):,}</div>
            <div class="metric">Min Tokens: {stats['token_stats'].get('min_tokens', 0):,}</div>
            
            <h3>Role Distribution</h3>
            <table>
                <tr><th>Role</th><th>Count</th></tr>
        """
        
        role_dist = stats['conversation_stats'].get('role_distribution', {})
        for role, count in role_dist.items():
            html_content += f"<tr><td>{role}</td><td>{count:,}</td></tr>"
        
        html_content += """
            </table>
            
            <h3>Quality Issues (Sample)</h3>
            <ul>
        """
        
        issues = stats['quality_metrics'].get('quality_issues_sample', [])
        for issue in issues[:10]:  # Show first 10 issues
            html_content += f"<li class='error'>{issue}</li>"
        
        html_content += """
            </ul>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Data summary report saved: {output_path}")


def create_training_report(experiment_path: str, output_path: str = None):
    """Create a comprehensive training report."""
    import json
    from pathlib import Path
    
    experiment_dir = Path(experiment_path)
    if output_path is None:
        output_path = experiment_dir / "training_report.html"
    
    summary_file = experiment_dir / "training_summary.json"
    if not summary_file.exists():
        logging.error(f"Training summary not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    metrics_file = experiment_dir / "logs" / summary['experiment_name'] / "metrics.jsonl"
    metrics = []
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
    
    # FIXED: Use double curly braces {{ }} to escape braces in CSS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report - {summary['experiment_name']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Training Report</h1>
        <h2>Experiment: {summary['experiment_name']}</h2>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h3>Training Summary</h3>
            <div class="metric">Total Time: {summary['total_training_time_hours']:.2f} hours</div>
            <div class="metric">Total Epochs: {summary['total_epochs']}</div>
            <div class="metric">Total Steps: {summary['total_steps']}</div>
            <div class="metric">Best Eval Loss: {summary['final_metrics']['best_eval_loss']:.6f}</div>
        </div>
        
        <div class="section">
            <h3>Model Configuration</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    # Add key config parameters
    key_params = [
        'hidden_size', 'num_layers', 'num_heads', 'seq_length',
        'batch_size', 'learning_rate', 'num_epochs', 'precision'
    ]
    
    for param in key_params:
        if param in summary['model_config']:
            value = summary['model_config'][param]
            html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h3>Health Summary</h3>
    """
    
    health = summary.get('health_summary', {})
    if health:
        html_content += f"""
            <div class="metric">Status: {health.get('status', 'Unknown')}</div>
            <div class="metric">Avg Loss: {health.get('avg_loss', 0):.6f}</div>
            <div class="metric">Avg Grad Norm: {health.get('avg_grad_norm', 0):.4f}</div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Training report saved: {output_path}")