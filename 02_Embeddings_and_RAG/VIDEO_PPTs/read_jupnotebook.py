import nbformat
import os

# Reload the uploaded notebook to get exact code
this_file_dir = os.path.dirname(__file__)
parent_dir = os.path.join(this_file_dir, os.pardir)  # os.pardir is a platform-independent way to refer to the parent directory
print(this_file_dir, "\n", parent_dir)
notebook_path = os.path.join(parent_dir,"Pythonic_RAG_Assignment.ipynb")
print(notebook_path)

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
except FileNotFoundError:
    print("Error: One or both of the specified files were not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Collect all non-trivial code snippets, including distance metrics
code_snippets = []

for cell in notebook.cells:
    if cell.cell_type == "code":
        code_text = cell.source.strip()
        # Exclude very small or trivial cells like print only
        if len(code_text) > 30:
            code_snippets.append(code_text)

# Show all to identify the distance metric snippet and others
print(f"Found {len(code_snippets)} code snippets:")

# Save snippets to a file
output_file = "extracted_code_snippets.py"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Code snippets extracted from Pythonic_RAG_Assignment.ipynb\n")
    f.write(f"# Total snippets: {len(code_snippets)}\n\n")
    
    for i, snippet in enumerate(code_snippets, 1):
        f.write(f"# --- Snippet {i} ---\n")
        f.write(snippet)
        f.write("\n\n" + "="*80 + "\n\n")

print(f"Code snippets saved to: {output_file}")

# Also display in terminal
for i, snippet in enumerate(code_snippets, 1):
    print(f"\n--- Snippet {i} ---")
    print(snippet)
    print("-" * 50)

'''
import os

def join_files(current_filename, parent_filename, output_filename):
    """
    Joins the content of a file in the current directory with a file in the parent directory.

    Args:
        current_filename (str): The name of the file in the current directory.
        parent_filename (str): The name of the file in the parent directory.
        output_filename (str): The name of the new file to store the combined content.
    """
    try:
        # Get the path to the current script's directory
        script_dir = os.path.dirname(__file__)

        # Get the path to the parent directory
        parent_dir = os.path.join(script_dir, os.pardir)  # os.pardir is a platform-independent way to refer to the parent directory

        # Construct the full paths to the files
        current_filepath = os.path.join(script_dir, current_filename)
        parent_filepath = os.path.join(parent_dir, parent_filename)
        output_filepath = os.path.join(script_dir, output_filename)

        # Read the content of the file in the current directory
        with open(current_filepath, 'r') as current_file:
            current_content = current_file.read()

        # Read the content of the file in the parent directory
        with open(parent_filepath, 'r') as parent_file:
            parent_content = parent_file.read()

        # Combine the content (you can customize how you want to join them)
        combined_content = current_content + "\n" + parent_content  # Example: Add a newline between the content

        # Write the combined content to the output file
        with open(output_filepath, 'w') as output_file:
            output_file.write(combined_content)

        print(f"Content of '{current_filename}' and '{parent_filename}' successfully joined and saved to '{output_filename}'")

    except FileNotFoundError:
        print("Error: One or both of the specified files were not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
current_file_name = "file_in_current_dir.txt"
parent_file_name = "file_in_parent_dir.txt"
output_file_name = "combined_file.txt"

# Create some dummy files for testing (optional)
with open(current_file_name, "w") as f:
    f.write("Content from the current directory.")
# Change to the parent directory to create the second file
os.chdir("..")
with open(parent_file_name, "w") as f:
    f.write("Content from the parent directory.")
# Change back to the script directory
os.chdir(os.path.dirname(__file__))

# Call the function to join the files
join_files(current_file_name, parent_file_name, output_file_name)
'''
