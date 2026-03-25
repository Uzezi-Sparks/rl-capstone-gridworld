


with open('src/environment/gridworld.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines, start=1):
    if 'class ' in line:
        print(f'{i}: {line}')



