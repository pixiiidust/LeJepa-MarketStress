"""
OutputWriter: writes scores.csv, soxx_lejepa_chart.png, run_config.json, console summary.

Chart layout:
    Top panel:    SOXX close, crash onset vertical line, shaded drawdown window
    Bottom panel: LeJEPA score + threshold, RV20 z-score + threshold, VIX z-score + threshold,
                  first breach markers for each signal

Public interface:
    OutputWriter(output_dir).write(scores_df, verdict, config)
"""
