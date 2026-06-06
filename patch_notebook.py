import json

with open('experiments_runner.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "'dataset.batch_size': bs_map[1024]," in line:
                source[i] = line.replace('bs_map[1024]', 'bs_map.get(1024, 4)')
            elif "'dataset.batch_size': bs_map[seq]," in line:
                source[i] = line.replace('bs_map[seq]', 'bs_map.get(seq, 1)')
            elif "'dataset.batch_size': bs_map[L_train]," in line:
                source[i] = line.replace('bs_map[L_train]', 'bs_map.get(L_train, 1)')
            
            # Wrap the actual run_training calls inside try-except so a single run failure doesn't crash the loop
            if 'train_res = run_training(' in line:
                indent = line[:len(line) - len(line.lstrip())]
                # Replace the line with try-except block
                source[i] = f"{indent}try:\n{indent}    {line.lstrip()}{indent}    print('Train success:', train_res['ok'])\n{indent}except Exception as e:\n{indent}    print(f'Exception during run: {{e}}')\n"
            
            # Remove the old print('Train success:', train_res['ok']) since it's now in the try block
            if "print('Train success:', train_res['ok'])" in line and 'try:' not in line:
                source[i] = ""

with open('experiments_runner.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
