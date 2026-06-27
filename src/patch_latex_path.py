import json
from pathlib import Path

def patch_notebook():
    notebook_path = Path("ufm-training.ipynb")
    if not notebook_path.exists():
        notebook_path = Path("src/ufm-training.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Could not find ufm-training.ipynb in current directory or src/")
        return

    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update config cell
    config_patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        
        # Look for the config cell
        if any("PROJECT_ROOT =" in line for line in source) and any("source_latex = PROJECT_ROOT" in line for line in source):
            new_source = []
            for line in source:
                # Add print statement
                if "print('RESULTS_DIR:', RESULTS_DIR)" in line:
                    new_source.append(line)
                    new_source.append("print('TEMPLATE_LATEX_DIR:', TEMPLATE_LATEX_DIR)\n")
                # Insert configuration variable and update source_latex
                elif "source_latex = PROJECT_ROOT / 'latex'" in line:
                    new_source.append("TEMPLATE_LATEX_DIR = PROJECT_ROOT / 'latex'  # Adjust path if template is uploaded in a separate dataset\n")
                    new_source.append("source_latex = TEMPLATE_LATEX_DIR\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            config_patched = True
            print("Successfully patched configuration cell (Step 0).")
            break

    # 2. Update path verification cell
    verify_patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        
        # Look for the path verification cell
        if any("required_paths = [" in line for line in source) and any("PROJECT_ROOT / 'latex'" in line for line in source):
            new_source = []
            for line in source:
                if "PROJECT_ROOT / 'latex'" in line:
                    new_source.append("    TEMPLATE_LATEX_DIR,\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            verify_patched = True
            print("Successfully patched verification cell (Step 1).")
            break

    if config_patched or verify_patched:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("Notebook saved successfully.")
    else:
        print("No matches found. Notebook was not modified.")

if __name__ == "__main__":
    patch_notebook()
