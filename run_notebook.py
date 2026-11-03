"""Execute the notebook and capture outputs for README."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json, time

print("Carregando notebook...")
with open('trabalho_final.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Clear all outputs first
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

print(f"Executando {len([c for c in nb.cells if c.cell_type == 'code'])} células de código...")
print("Isso pode levar 10-20 minutos para SVM e RandomForest...")

ep = ExecutePreprocessor(
    timeout=1800,  # 30 minutos
    kernel_name='python3',
    allow_errors=False
)

t0 = time.time()
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    elapsed = time.time() - t0
    print(f"\n✅ Notebook executado com sucesso em {elapsed/60:.1f} minutos!")

    with open('trabalho_final.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook salvo com outputs.")

except Exception as e:
    print(f"\n❌ Erro na execução: {e}")
    # Save partially executed notebook
    with open('trabalho_final_partial.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook parcialmente executado salvo como trabalho_final_partial.ipynb")
    raise
