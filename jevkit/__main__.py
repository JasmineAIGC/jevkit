# -*- coding: utf-8 -*-
"""python -m jevkit 等价于 jevkit 命令。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
