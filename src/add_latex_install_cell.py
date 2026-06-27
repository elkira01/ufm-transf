import json
from pathlib import Path

def add_install_cell():
    notebook_path = Path("ufm-training.ipynb")
    if not notebook_path.exists():
        notebook_path = Path("src/ufm-training.ipynb")
        
    if not notebook_path.exists():
        print("Error: Could not find ufm-training.ipynb in current directory or src/")
        return

    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Define the new cell to insert
    install_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "trusted": True
        },
        "outputs": [],
        "source": [
            "# Install pdflatex and dependencies on Kaggle (requires internet enabled in notebook settings)\n",
            "!apt-get update && apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super texlive-bibtex-extra\n"
        ]
    }

    # Find the Step 9 markdown cell
    target_idx = -1
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            joined = "".join(source)
            if "## 9) Compile LaTeX PDFs" in joined:
                target_idx = idx
                break

    if target_idx != -1:
        # Insert the cell before Step 9
        nb["cells"].insert(target_idx, install_cell)
        print(f"Successfully inserted installation cell at index {target_idx}.")
        
        # Save the notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("Notebook saved successfully.")
    else:
        print("Error: Could not find '## 9) Compile LaTeX PDFs' markdown cell.")

if __name__ == "__main__":
    add_install_cell()
