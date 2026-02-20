"""
use script to make pdf's into markdowns, usable for index script.
"""
import os
import pathlib
import pymupdf4llm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../../data')

    for root, dirs, files in os.walk(r'{}'.format(data_dir+'/books')):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                markdown_name = 'markdown_' + file.split('.')[0] +'.md'
                md_text = pymupdf4llm.to_markdown(f'{path}')
                pathlib.Path(data_dir + '/markdowns/'+ markdown_name).write_bytes(md_text.encode())
                print(f'parsed file : {path}\n')

if __name__ == '__main__':
    main()
