import PyPDF2
import re
from tqdm import tqdm
from collections import namedtuple

Chapter = namedtuple('Chapter', ['index', 'title', 'content'])

# TODO: print all title and sub-title

# pdf_file = open('./Web Resource space model.pdf', 'rb')
pdf_file = open('../outlines/data/history/', 'rb')
pdf_object = PyPDF2.PdfReader(pdf_file)
page_count = len(pdf_object.pages)

chapters = set()
chapter_index, sub_chapter, subsub_chapter = None, None, None

content_mapper = {}

for page_index, page in enumerate(pdf_object.pages):
    # skip preface and references
    if page_index < 12 or page_index > 227:
        continue

    text = page.extract_text().strip()

    if text.startswith('Chapter '):
        _, chapter_index, *title_words = text.split('\n')[0].strip().split(' ')
        appendix = {
            '3': ['for', 'Resource', 'Space', 'Model'],
            '4': ['Space', 'Model'],
            '5': ['Space', 'Model'],
            '7': ['Space'],
            '8': ['Space'],
        }
        chapter_title = ' '.join(title_words + appendix.get(chapter_index, []))
        sub_chapter, subsub_chapter = None, None
        text = '\n'.join(text.split('\n')[1 + (1 if chapter_index in appendix else 0):])

    elif page_index == 64:
        chapter_index, chapter_title = '2', 'A Semantic Overlay Integrating Normalization with Autonomy'
        sub_chapter, subsub_chapter = None, None

    if page_index == 31:
        sub_chapter, sub_title = '1.3', 'Application Scenarios of the Resource Space Model'
        subsub_chapter = None
    if page_index == 88:
        subsub_chapter, subsub_title = '2.4.4', 'Operations on the Union View of Resource Space and Semantic Link Network'

    for line in [line.strip().strip('-') for line in text.split('\n') if line]:
        if re.match('\d+\s+Chapter', line):
            line = ''

        index, *title_words = line.split(' ')
        if (
            re.match('[1-9](\.[1-9])+', index) and
            not title_words[-1].isnumeric() and
            # 5.2 formula definition
            title_words[0] != '|'
        ):
            appendix = {
                '2.2': 'Network',
                '2.3.1': 'Space Model',
                '2.3.2': 'Network and Correlation',
                '3.2': 'Spaces',
                '4.3.2': 'Calculus',
                '4.3.3': 'Algebra',
                '5.4.1': 'Complexity',
                '8.2': 'Peer-to-Peer',
                '9.4.2': 'Model',
            }

            if len(index.split('.')) == 2:
                sub_chapter, sub_title = index, ' '.join(title_words + appendix.get(index, '').split(' '))
                subsub_chapter, subsub_title = None, None
                line = ''
            elif len(index.split('.')) == 3:
                subsub_chapter, subsub_title = index, ' '.join(title_words + appendix.get(index, '').split(' '))
                line = ''
            else:
                raise Exception(f'Invalid index {index} and title {title_words}')

        if subsub_chapter:
            chapters.add(Chapter(subsub_chapter, subsub_title, ''))
            content_mapper[subsub_chapter] = content_mapper.get(subsub_chapter, '') + line
        elif sub_chapter:
            chapters.add(Chapter(sub_chapter, sub_title, ''))
            content_mapper[sub_chapter] = content_mapper.get(sub_chapter, '') + line
        elif chapter_index:
            chapters.add(Chapter(chapter_index, chapter_title, ''))
            content_mapper[chapter_index] = content_mapper.get(chapter_index, '') + line

# print(content_mapper)

mapper = {}
for page in pdf_object.pages[5:9]:
    text = page.extract_text().strip()
    for line in text.split('\n'):
        key, *value = line.split(' ')
        mapper[key] = ' '.join(value)

output = {}

for index, title, _ in sorted(chapters, key=lambda c: c.index):
    if (chapter_len := len(index.split('.'))) == 1:
        output[index] = {
            'title': title,
            'content': content_mapper[index],
            'sub_chapters': {},
        }
    elif chapter_len == 2:
        parent_index, child_index = index.split('.')
        output[parent_index]['sub_chapters'][index] = {
            'title': title,
            'content': content_mapper[index],
            'sub_chapters': {},
        }
    elif chapter_len == 3:
        parent_index, child_index, child_child_index = index.split('.')
        output[parent_index]['sub_chapters'][f'{parent_index}.{child_index}']['sub_chapters'][index] = {
            'title': title,
            'content': content_mapper[index],
        }

import json
from pathlib import Path

# TODO: 分段
Path('./output.json').write_text(json.dumps(output, indent=2))