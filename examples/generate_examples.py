"""Генератор эталонных PDF для интеграционных тестов.

Скрипт создаёт три векторных PDF-файла в ``examples/``:

* ``simple.pdf`` — один лист: источник, предохранитель F1, кнопка SB1,
  реле K1, лампа HL1, провода, метки цепей L1/N и таблица BOM из 4 строк;
* ``multipage.pdf`` — два листа со сквозной меткой цепи L1 и разными
  компонентами (R1, R2);
* ``problem.pdf`` — три листа: растровая страница без векторного слоя,
  X-образное пересечение проводов без точки и таблица BOM с пустой
  ячейкой наименования.

Кириллица вставляется только встроенным шрифтом (DejaVu Sans). Если шрифт
не найден, используется латиница — PDF всё равно остаётся читаемым.
Запуск::

    python examples/generate_examples.py [--output-dir examples]

Файлы детерминированы: повторная генерация даёт тот же набор страниц.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

#: Кандидаты шрифта с поддержкой кириллицы.
FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
)

#: Имя встроенного шрифта в PDF.
FONT_NAME = "DejaVu"


def _font_file() -> Path | None:
    """Возвращает путь к шрифту с кириллицей или ``None``."""
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


class _Text:
    """Вставка текста с кириллицей и латинским fallback-ом."""

    def __init__(self, page: pymupdf.Page) -> None:
        self.page = page
        self.font = _font_file()
        self.fontname = FONT_NAME if self.font is not None else "helv"
        if self.font is not None:
            page.insert_font(fontname=FONT_NAME, fontfile=str(self.font))

    def write(self, x: float, y: float, text: str, size: float = 9.0) -> None:
        """Вставляет строку; при отсутствии шрифта заменяет кириллицу."""
        if self.font is None:
            text = text.encode("ascii", "replace").decode("ascii")
        self.page.insert_text(
            (x, y), text, fontname=self.fontname, fontsize=size
        )


def _new_document() -> pymupdf.Document:
    """Создаёт пустой PDF-документ."""
    return pymupdf.open()


def _save(document: pymupdf.Document, target: Path) -> Path:
    """Подмножество шрифтов (уменьшение размера) и сохранение PDF."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        document.subset_fonts()
    except Exception:  # noqa: BLE001 - subsetting не должен ломать генерацию
        pass
    document.save(str(target))
    document.close()
    return target


def _table(
    page: pymupdf.Page,
    text: _Text,
    *,
    x0: float,
    y0: float,
    width: float,
    col_widths: list[float],
    row_height: float,
    rows: list[list[str]],
) -> None:
    """Рисует таблицу с границами и вставляет текст ячеек.

    Пустая строка в ``rows`` оставляет ячейку без текста (битая таблица).
    """
    cols = len(col_widths)
    height = row_height * len(rows)
    x1 = x0 + width
    y1 = y0 + height
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1)
    x = x0
    for col_width in col_widths[:-1]:
        x += col_width
        page.draw_line((x, y0), (x, y1))
    for row_index in range(1, len(rows)):
        y = y0 + row_index * row_height
        page.draw_line((x0, y), (x1, y))
    x = x0
    for row_index, row in enumerate(rows):
        y_baseline = y0 + row_index * row_height + row_height * 0.65
        x = x0
        for col_index in range(cols):
            value = row[col_index] if col_index < len(row) else ""
            if value:
                text.write(x + 4.0, y_baseline, value)
            x += col_widths[col_index]
    del text  # текст вставлен через переданный объект


def _simple(page: pymupdf.Page, text: _Text) -> None:
    """Рисует однолистовую схему simple.pdf."""
    # Шины питания L1 (верх) и N (низ).
    page.draw_line((60, 100), (520, 100))
    page.draw_line((60, 300), (520, 300))
    text.write(60, 96, "L1")
    text.write(60, 296, "N")

    # Источник/батарея: длинная и короткая пластины.
    page.draw_line((70, 150), (70, 210))
    page.draw_line((84, 165), (84, 195))
    page.draw_line((70, 100), (70, 150))
    page.draw_line((84, 195), (84, 300))

    # F1 — предохранитель (прямоугольник 60×20, ratio 3).
    page.draw_rect(pymupdf.Rect(150, 150, 210, 170), color=(0, 0, 0), width=1)
    page.draw_line((150, 100), (150, 150))
    page.draw_line((150, 170), (150, 300))
    text.write(172, 145, "F1")

    # SB1 — кнопка: два контакта под углом + толкатель.
    page.draw_line((230, 150), (255, 150))
    page.draw_line((255, 150), (275, 135))
    page.draw_line((255, 140), (255, 128))
    page.draw_line((230, 100), (230, 150))
    page.draw_line((275, 135), (275, 300))
    text.write(232, 124, "SB1")

    # K1 — реле: прямоугольник + обмотка-дуга внутри.
    page.draw_rect(pymupdf.Rect(400, 150, 480, 210), color=(0, 0, 0), width=1)
    shape = page.new_shape()
    shape.draw_bezier(
        pymupdf.Point(415, 180),
        pymupdf.Point(425, 155),
        pymupdf.Point(455, 155),
        pymupdf.Point(465, 180),
    )
    shape.draw_bezier(
        pymupdf.Point(465, 180),
        pymupdf.Point(455, 205),
        pymupdf.Point(425, 205),
        pymupdf.Point(415, 180),
    )
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    page.draw_line((400, 100), (400, 150))
    page.draw_line((400, 210), (400, 300))
    text.write(428, 145, "K1")

    # HL1 — лампа: окружность радиусом 12 pt.
    page.draw_circle((300, 200), 12, color=(0, 0, 0), width=1)
    page.draw_line((300, 100), (300, 188))
    page.draw_line((300, 212), (300, 300))
    text.write(318, 205, "HL1")

    # Таблица BOM: 4 строки + заголовок.
    _table(
        page,
        text,
        x0=40,
        y0=600,
        width=515,
        col_widths=[90.0, 300.0, 125.0],
        row_height=24.0,
        rows=[
            ["Поз.", "Наименование", "Кол-во"],
            ["F1", "Предохранитель 5А", "1"],
            ["SB1", "Кнопка", "1"],
            ["K1", "Реле 24В", "1"],
            ["HL1", "Лампа 24В", "1"],
        ],
    )


def _multipage(document: pymupdf.Document) -> None:
    """Рисует двухлистовую схему multipage.pdf со сквозной меткой L1."""
    for page_index, (ref, rect) in enumerate(
        (
            ("R1", pymupdf.Rect(100, 55, 140, 65)),
            ("R2", pymupdf.Rect(100, 50, 140, 70)),
        )
    ):
        page = document.new_page(width=300, height=200)
        text = _Text(page)
        page.draw_line((20, 60), (100, 60))
        text.write(20, 56, "L1")
        page.draw_rect(rect, color=(0, 0, 0), width=1)
        text.write(145, 64, ref)
        # Номер листа — только как текст, не влияет на цепи.
        text.write(250, 190, f"Лист {page_index + 1}")


def _problem(document: pymupdf.Document) -> None:
    """Рисует трёхлистовую проблемную схему problem.pdf."""
    # Лист 1: только растр, без векторного слоя.
    raster_page = document.new_page(width=200, height=200)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16))
    pixmap.clear_with(230)
    raster_page.insert_image(
        pymupdf.Rect(20, 20, 180, 180), pixmap=pixmap
    )

    # Лист 2: X-образное пересечение проводов БЕЗ точки соединения.
    crossing_page = document.new_page(width=200, height=200)
    crossing_page.draw_line((20, 20), (120, 120))
    crossing_page.draw_line((20, 120), (120, 20))

    # Лист 3: таблица BOM с пустой ячейкой наименования.
    table_page = document.new_page(width=500, height=200)
    text = _Text(table_page)
    _table(
        table_page,
        text,
        x0=40,
        y0=40,
        width=420,
        col_widths=[80.0, 240.0, 100.0],
        row_height=30.0,
        rows=[
            ["Поз.", "Наименование", "Кол-во"],
            ["R1", "", "1"],
            ["R2", "Резистор 10 кОм", "1"],
        ],
    )


def generate_simple(path: str | Path) -> Path:
    """Создаёт ``simple.pdf`` и возвращает путь."""
    document = _new_document()
    page = document.new_page(width=595, height=842)
    _simple(page, _Text(page))
    return _save(document, Path(path))


def generate_multipage(path: str | Path) -> Path:
    """Создаёт ``multipage.pdf`` и возвращает путь."""
    document = _new_document()
    _multipage(document)
    return _save(document, Path(path))


def generate_problem(path: str | Path) -> Path:
    """Создаёт ``problem.pdf`` и возвращает путь."""
    document = _new_document()
    _problem(document)
    return _save(document, Path(path))


#: Имя файла → функция генерации.
GENERATORS = {
    "simple.pdf": generate_simple,
    "multipage.pdf": generate_multipage,
    "problem.pdf": generate_problem,
}


def generate_all(output_dir: str | Path) -> dict[str, Path]:
    """Генерирует все эталонные PDF в ``output_dir``.

    Args:
        output_dir: Каталог назначения.

    Returns:
        Отображение «имя файла → путь».
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        name: generator(directory / name)
        for name, generator in GENERATORS.items()
    }


def main(argv: list[str] | None = None) -> int:
    """Точка входа скрипта генерации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Каталог для PDF (по умолчанию — каталог examples/)",
    )
    args = parser.parse_args(argv)
    created = generate_all(args.output_dir)
    for name, path in created.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())