import ast
with open(r'C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\9.00Trading_ML_Prediction_Widget_Demo_Live_backtrade\Examin.py', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'StrategyLossAnalyzer':
        print("Class found at line", node.lineno)
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        print("Methods defined:", methods)