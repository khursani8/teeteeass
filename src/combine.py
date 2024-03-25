from pathlib import Path
from datetime import datetime

def flatten_list(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten_list(element))
        else:
            result.append(element)
    return result

now = datetime.now()
dbp_path = sorted(Path("./source/dbp_clean/").glob("*.list"))
single_path = Path("./source/single.list")
common_path = Path("./source/common.list")

paths = [single_path,common_path] + dbp_path

contents = []
for path in paths:
    with open(path, 'r') as file:
        content = file.read()
        contents.append(content)

contents = flatten_list(contents)

contentss = "\n".join(contents).split("\n")

# Create a dictionary to store unique text entries and their associated ARPABET representations
unique_entries = {}

# Iterate through each entry and update the dictionary with only the first occurrence of each text entry
for entry in contentss:
    if not entry:
        continue
    text, arpabet = entry.split('|')
    if text not in unique_entries:
        unique_entries[text] = arpabet

# Convert the unique dictionary entries back to a list of strings
unique_list = [f"{text.strip()}|{arpabet.strip()}" for text, arpabet in unique_entries.items()]

# Open a new file for writing the combined contents
with open('combine.rep', 'w+') as combined_file:
    combined_file.write(f"## Date:  {now.date()}\n")
    combined_file.write("## Generated by DSMO\n\n")
    for data in unique_list:
        combined_file.write(data.replace("|",'  ') + '\n')  # Adding a newline as a delimiter