from pathlib import Path

output_file = Path('foundations_of_database.triplets')
lines = []

for line in Path('foundations_of_database copy.triplets.bak').read_text().split('\n'):
    if not line: continue
    
    pre, ind, rtype, post, position = line.split('\t')
    position = f'x-{position}'
    
    lines.append(f'{pre}\t{ind}\t{rtype}\t{post}\t{position}')
    
output_file.write_text('\n'.join(lines))