import sys

class FixedWidthBarChart:
    def __init__(self, iterable, total=None, width=20, height=10, title="BAR CHART"):
        self.iterable = iter(iterable)
        self.total_steps = total or self._try_len(iterable)
        self.width = width
        self.height = height
        self.title = title
        self.segment = self.total_steps / width
        self.values = [None] * width
        self.last_draw_lines = 0
        self.curr_step = 0
        self._buffered_value = None

        self.colors = {
            'box': '\033[36m',
            'bar': '\033[34m',
            'text': '\033[37m',
            'reset': '\033[0m'
        }

    def _try_len(self, it):
        try:
            return len(it)
        except TypeError:
            raise ValueError("Must provide total= when using a generator.")

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.iterable)
        return item

    def update(self, value):
        index = int(self.curr_step / self.segment)
        if 0 <= index < self.width:
            self.values[index] = value
        self.curr_step += 1
        self._draw()

    def _draw(self):
        valid_vals = [v for v in self.values if v is not None]
        if not valid_vals:
            return

        min_val = min(valid_vals)
        max_val = max(valid_vals)
        range_val = max_val - min_val if max_val != min_val else 1

        normalized = [
            int((v - min_val) / range_val * (self.height - 1)) if v is not None else -1
            for v in self.values
        ]

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for x, h in enumerate(normalized):
            if h < 0:
                continue
            for y in range(h + 1):
                grid[-1 - y][x] = '⣿'

        lines = []
        lines.append(f"{self.colors['box']} {self.title.center(self.width)} {self.colors['reset']}")
        lines.append(f"{self.colors['box']}┌{'─'*self.width}┐{self.colors['reset']}")
        for i, row in enumerate(grid):
            suffix = ''
            if i == 0:
                suffix = f"{self.colors['text']} {max(valid_vals):.1f}{self.colors['reset']}"
            elif i == self.height - 1:
                suffix = f"{self.colors['text']} {min(valid_vals):.1f}{self.colors['reset']}"
            lines.append(
                f"{self.colors['box']}│{self.colors['bar']}{''.join(row)}{self.colors['box']}│{self.colors['reset']}{suffix}"
            )
        lines.append(f"{self.colors['box']}└{'─'*self.width}┘{self.colors['reset']}")
        last_val = valid_vals[-1]
        lines.append(f"{self.colors['text']}Latest: {last_val:.2f}{self.colors['reset']}")

        sys.stdout.write('\033[F' * self.last_draw_lines)
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()
        self.last_draw_lines = len(lines)










import sys

class FixedWidthLinePlot:
    def __init__(self, iterable, total=None, width=40, height=10, title="LINE PLOT"):
        self.iterable = iter(iterable)
        self.total_steps = total or self._try_len(iterable)
        self.width = width
        self.height = height
        self.title = title
        self.segment = self.total_steps / width
        self.values = [None] * width
        self.last_draw_lines = 0
        self.curr_step = 0

        self.colors = {
            'box': '\033[36m',
            'line': '\033[34m',
            'text': '\033[37m',
            'reset': '\033[0m'
        }

    def _try_len(self, it):
        try:
            return len(it)
        except TypeError:
            raise ValueError("Must specify `total=` when using a generator.")

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.iterable)
        return item

    def update(self, value):
        index = int(self.curr_step / self.segment)
        if 0 <= index < self.width:
            self.values[index] = value
        self.curr_step += 1
        self._draw()

    def _draw(self):
        valid_vals = [v for v in self.values if v is not None]
        if len(valid_vals) < 2:
            return

        min_val = min(valid_vals)
        max_val = max(valid_vals)
        range_val = max_val - min_val if max_val != min_val else 1

        normalized = [
            int((v - min_val) / range_val * (self.height - 1)) if v is not None else -1
            for v in self.values
        ]

        # 构建空白网格
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # 绘制折线点
        for x in range(1, self.width):
            if self.values[x - 1] is None or self.values[x] is None:
                continue

            y0 = self.height - 1 - normalized[x - 1]
            y1 = self.height - 1 - normalized[x]

            if y0 == y1:
                grid[y0][x] = '─'
            elif y1 < y0:
                grid[y1][x] = '╱'
            else:
                grid[y1][x] = '╲'

        # 标出当前点
        for x in range(self.width):
            y = self.height - 1 - normalized[x]
            if 0 <= y < self.height and self.values[x] is not None:
                grid[y][x] = '●'

        # 构建文本输出
        lines = []
        lines.append(f"{self.colors['box']} {self.title.center(self.width)} {self.colors['reset']}")
        lines.append(f"{self.colors['box']}┌{'─'*self.width}┐{self.colors['reset']}")
        for i, row in enumerate(grid):
            suffix = ''
            if i == 0:
                suffix = f"{self.colors['text']} {max(valid_vals):.1f}{self.colors['reset']}"
            elif i == self.height - 1:
                suffix = f"{self.colors['text']} {min(valid_vals):.1f}{self.colors['reset']}"
            lines.append(
                f"{self.colors['box']}│{self.colors['line']}{''.join(row)}{self.colors['box']}│{self.colors['reset']}{suffix}"
            )
        lines.append(f"{self.colors['box']}└{'─'*self.width}┘{self.colors['reset']}")
        lines.append(f"{self.colors['text']}Latest: {valid_vals[-1]:.2f}{self.colors['reset']}")

        sys.stdout.write('\033[F' * self.last_draw_lines)
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()
        self.last_draw_lines = len(lines)
