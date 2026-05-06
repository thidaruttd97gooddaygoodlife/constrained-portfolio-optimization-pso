import re

file_path = r'c:\Users\THITHI\ciproject\constrained-portfolio-optimization-pso\portfolio_pso.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the missing best_run in write_detailed_report
old_logic = r'''    constrained_test = constrained_artifacts.metrics_summary.query\("portfolio == @constrained_artifacts.experiment_name and sample == 'test'"\).iloc\[0\]
    equal_test = constrained_artifacts.metrics_summary.query\("portfolio == 'EqualWeight' and sample == 'test'"\).iloc\[0\]

    report = f"""# รายงานการวิจัยขั้นสูง'''

new_logic = r'''    constrained_test = constrained_artifacts.metrics_summary.query("portfolio == @constrained_artifacts.experiment_name and sample == 'test'").iloc[0]
    equal_test = constrained_artifacts.metrics_summary.query("portfolio == 'EqualWeight' and sample == 'test'").iloc[0]
    best_run = constrained_artifacts.run_history_df.sort_values("sharpe", ascending=False).iloc[0]

    report = f"""# รายงานการวิจัยขั้นสูง'''

content = re.sub(old_logic, new_logic, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed best_run in write_detailed_report.")
