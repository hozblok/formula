[🇬🇧 English](tests.md) · 🇷🇺 **Русский**

## Запуск тестов

Этот проект поставляется с компилируемым расширением Python (`formula._formula`).
Расширение необходимо собрать до того, как тесты смогут импортировать пакет.
Проект использует **src-layout**: Python-пакет расположен в
`src/formula/`, а не в корне репозитория.

### Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install .[test]

python -m pytest -q
```

Для активной разработки предпочтительнее editable-установка, чтобы изменения в
Python-обёртке вступали в силу без переустановки:

```bash
python -m pip install -e .[test]
```

### Запуск подмножества тестов

```bash
# Один файл
python -m pytest -v tests/test_simple.py

# Только тесты отдельных функций, добавленные в последнем релизе
python -m pytest -v tests/functions/

# Тесты одной функции
python -m pytest -v tests/functions/one_arg/test_sin.py

# По ключевому слову по всему набору тестов
python -m pytest -v -k "sin or cos"
```

### Устранение неполадок

#### Компиляция завершается с ошибкой

Убедитесь, что в PATH доступен работающий инструментарий C++:

| ОС       | Инструментарий                                       |
|----------|------------------------------------------------------|
| Linux    | `g++` (или `clang++`); `python3-dev`                  |
| macOS    | Xcode Command Line Tools (`xcode-select --install`)  |
| Windows  | Visual Studio Build Tools (рабочая нагрузка C++)     |

#### Устаревшее расширение после получения новых изменений C++

Скомпилированный `.so`/`.pyd` от предыдущей editable-сборки может скрыть свежую
ошибку сборки. Перед переустановкой очистите оставшиеся артефакты:

```bash
rm -f src/formula/_formula*.so src/formula/_formula*.pyd
rm -rf src/formula/__pycache__ build *.egg-info
python -m pip install .[test]
```

#### macOS: `ld: library not found` или wheel не устанавливается на старых macOS

Задайте целевую версию развёртывания перед установкой, чтобы итоговый `.so` был
ABI-совместим с той версией macOS, на которой вы собираетесь запускать:

```bash
export MACOSX_DEPLOYMENT_TARGET=10.15   # Intel; используйте 11.0+ для Apple Silicon
python -m pip install .[test]
```

#### `python setup.py build_ext --inplace` устарел

Setuptools >= 58 выводит предупреждение об устаревании; будущие версии его
удалят. Используйте вместо этого `pip install .[test]` (или
`pip install -e .[test]` для editable-режима) — они делают то же самое через
поддерживаемые пути PEP 517/660.

### Структура проекта (src-layout)

```
formula/                  ← repo root
├── src/
│   ├── formula/          ← Python package (import target)
│   │   ├── __init__.py
│   │   ├── formula.py
│   │   └── _formula.so   ← built C++ extension (after build)
│   └── cpp/              ← C++ extension sources
│       ├── main.cpp
│       ├── csconstants.hpp
│       ├── cseval/
│       └── csformula/
├── tests/
│   ├── functions/        ← per-function tests
│   ├── test_operations.py
│   └── test_simple.py
├── boost_headers/
├── setup.py
└── pyproject.toml
```

### Что делает CI

Рабочие процессы тестирования GitHub Actions
([test-mac.yml](../.github/workflows/test-mac.yml),
[test-ubu.yml](../.github/workflows/test-ubu.yml),
[test-win.yml](../.github/workflows/test-win.yml)) запускают
`pip install .[test]` (не editable), а затем `python -m pytest -q`.
Тесты wheel в публикующем рабочем процессе используют `cibuildwheel`, который
устанавливает собранный wheel в свежий venv и запускает `pytest {project}/tests` —
см. [pyproject.toml](../pyproject.toml) `[tool.cibuildwheel]`.
