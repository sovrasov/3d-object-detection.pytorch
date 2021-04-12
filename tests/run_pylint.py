import logging
import os
import re
import sys

from pylint.lint import Run

if __name__ == '__main__':
    ignored_patterns = [
        '3rdparty/',
        'setup.py',
        'configs/detection',
        'torchdet3d/models',
        'build',
        'deprecated',
        '.history',
        'torchdet3d/models',
    ]

    to_pylint = []
    wd = os.path.abspath('.')
    for root, dirnames, filenames in os.walk(wd):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, wd)
                if all(not re.match(pattern, rel_path) for pattern in ignored_patterns):
                    to_pylint.append(rel_path)

    msg_status = Run(to_pylint, exit=False).linter.msg_status
    if msg_status:
        logging.error(f'pylint failed with code {msg_status}')
        sys.exit(msg_status)
