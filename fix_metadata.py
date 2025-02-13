import argparse
import nbformat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set all cells to slide type "skip"')
    parser.add_argument('notebook', type=str, help='Path to the notebook')
    args = parser.parse_args()

    # Load your notebook
    notebook_path = args.notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Modify all cells
    for cell in nb.cells:
        if 'metadata' not in cell:
            cell['metadata'] = {}
        cell['metadata']['slideshow'] = {'slide_type': 'skip'}

    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("All cells have been set to slide type 'skip'")