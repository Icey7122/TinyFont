"""TinyFont viewer and renderer.

This module provides a simple Tkinter-based viewer for TinyFont data
stored in JSON or TYF formats. It includes utilities to rasterize
strokes to pixel images and a `FontViewer` class to display and
interactively preview fonts.

The file is intended to be run as a script to start the viewer.
"""

import json
import io
import struct
import os
import math

try:
    import tkinter as tk
    import tkinter.messagebox as messagebox
    import tkinter.filedialog as filedialog
except ImportError:
    import Tkinter as tk
    import tkMessageBox as messagebox
    import tkFileDialog as filedialog

try:
    # Enable per-monitor DPI awareness on Windows to improve HiDPI support.
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    # Silently ignore if ctypes or the API is unavailable.
    pass

from tinyfont import TyfParser

def get_circle_brush(thickness):
    """Generate integer offsets approximating a circular brush.

    Args:
        thickness (int): Stroke thickness in pixels.

    Returns:
        list[tuple[int, int]]: List of (dx, dy) offsets to apply when
            stamping the brush at a pixel position. Always returns at
            least `[(0, 0)]` for thickness <= 1 or if no offsets are
            computed.
    """
    if thickness <= 1:
        return [(0, 0)]

    offsets = []
    radius = thickness / 2.0
    r_sq = radius * radius
    bound = int(math.ceil(radius))
    for dx in range(-bound, bound + 1):
        for dy in range(-bound, bound + 1):
            # Use a small epsilon to include border pixels.
            if dx * dx + dy * dy < r_sq + 0.1:
                offsets.append((dx, dy))
    return offsets if offsets else [(0, 0)]

def get_line_pixels(p0, p1, brush, size):
    """Rasterize a line segment and apply a brush to produce pixel coords.

    The function uses an integer Bresenham-like algorithm to iterate
    over the integer coordinates between `p0` and `p1`. For each
    visited pixel the brush offsets are applied and valid pixels are
    collected.

    Args:
        p0 (tuple[int, int]): Start point (x0, y0).
        p1 (tuple[int, int]): End point (x1, y1).
        brush (list[tuple[int, int]]): Offsets produced by
            `get_circle_brush` to stamp around each visited pixel.
        size (int): Image dimension used to clip coordinates (0..size-1).

    Returns:
        set[tuple[int, int]]: Set of (x, y) pixel coordinates inside the
            clipping box.
    """
    pixels = set()
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        for ox, oy in brush:
            tx, ty = x0 + ox, y0 + oy
            if 0 <= tx < size and 0 <= ty < size:
                pixels.add((tx, ty))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pixels

class FontViewer:
    """Tkinter-based viewer for TinyFont data.

    The viewer supports loading fonts from JSON or TYF, rendering
    skeleton strokes and pixelized previews, and basic interactive
    controls (zoom, stroke width, wrapping).
    """

    def __init__(self, master):
        """Initialize UI components and default state.

        Args:
            master (tk.Tk): Root Tkinter window instance.
        """
        self.master = master
        master.title("TinyFont Tool")
        master.geometry("1200x900")

        self.font_data_json = {}
        self.font_data_tyf = None
        self.current_mode = None
        self.img_refs = []

        # Top control bar
        ctrl_frame = tk.Frame(master, bg="#ddd", pady=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(ctrl_frame, text="打开字体", command=self.open_file).pack(side=tk.LEFT, padx=10)
        self.lbl_file = tk.Label(ctrl_frame, text="未加载", width=15, anchor="w", bg="#eee")
        self.lbl_file.pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text="字号:", bg="#ddd").pack(side=tk.LEFT, padx=5)
        self.entry_base_size = tk.Entry(ctrl_frame, width=4)
        self.entry_base_size.insert(0, "16")
        self.entry_base_size.pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text=" 缩放:", bg="#ddd").pack(side=tk.LEFT)
        self.scale_zoom = tk.Scale(
            ctrl_frame,
            from_=1,
            to=40,
            orient=tk.HORIZONTAL,
            length=120,
            command=lambda x: self.draw(),
        )
        self.scale_zoom.set(8)
        self.scale_zoom.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text=" 线宽:", bg="#ddd").pack(side=tk.LEFT)
        self.entry_width = tk.Entry(ctrl_frame, width=3)
        self.entry_width.insert(0, "1")
        self.entry_width.pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text=" 单宽:", bg="#ddd").pack(side=tk.LEFT, padx=5)
        self.entry_wrap_cnt = tk.Entry(ctrl_frame, width=3)
        self.entry_wrap_cnt.insert(0, "16")
        self.entry_wrap_cnt.pack(side=tk.LEFT)

        self.show_skel = tk.IntVar(value=1)
        self.show_pixel = tk.IntVar(value=1)
        tk.Checkbutton(ctrl_frame, text="骨架", variable=self.show_skel, command=self.draw, bg="#ddd").pack(side=tk.LEFT)
        tk.Checkbutton(ctrl_frame, text="像素", variable=self.show_pixel, command=self.draw, bg="#ddd").pack(side=tk.LEFT)

        # Multi-line input area
        input_box = tk.Frame(master, bg="#eee", pady=5)
        input_box.pack(side=tk.TOP, fill=tk.X)

        tk.Label(input_box, text="预览文本:", bg="#eee").pack(side=tk.LEFT, padx=10)
        self.text_area = tk.Text(input_box, height=4, width=60, font=("Consolas", 11))
        self.text_area.insert(
            "1.0",
            u"""我可以吞下玻璃而不伤身体
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
the quick brown fox jumps over the lazy dog
!@#$%^&*(-_{}[]\|;:'".\,<>?/~`""",
        )
        self.text_area.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.text_area.bind("<Control-Return>", lambda e: self.draw())

        tk.Button(input_box, text="刷新渲染\n(Ctrl+Enter)", command=self.draw, bg="#4CAF50", fg="white", padx=10).pack(side=tk.LEFT, padx=10)

        # Scrollable canvas
        self.canvas_frame = tk.Frame(master, bg="#222")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="#1a1a1a",
            xscrollcommand=self.hbar.set,
            yscrollcommand=self.vbar.set,
            highlightthickness=0,
        )

        self.hbar.config(command=self.canvas.xview)
        self.vbar.config(command=self.canvas.yview)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_wheel)

    def on_mouse_wheel(self, event):
        """Scroll vertically in response to mouse wheel events."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_shift_wheel(self, event):
        """Scroll horizontally when Shift+wheel is used."""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_ctrl_wheel(self, event):
        """Adjust zoom level when Ctrl+wheel is used."""
        curr = self.scale_zoom.get()
        self.scale_zoom.set(curr + (1 if event.delta > 0 else -1))

    def open_file(self):
        """Open a font file and load font data.

        Supports JSON and TYF file formats. On success the viewer will
        switch mode and trigger a redraw. Errors are shown in a
        message box.
        """
        fname = filedialog.askopenfilename(filetypes=[("Font Files", "*.json *.tyf"), ("All", "*.*")])
        if not fname:
            return

        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext == ".json":
                with io.open(fname, "r", encoding="utf-8") as f:
                    self.font_data_json = json.load(f)
                self.current_mode = "JSON"
            else:
                p = TyfParser()
                if p.load(fname):
                    self.font_data_tyf = p
                    self.current_mode = "TYF"
                else:
                    return

            self.lbl_file.config(text=os.path.basename(fname), fg="green")
            self.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw(self):
        """Render the current text using the loaded font data.

        The method reads UI parameters (base size, zoom, line width,
        wrap limit), composes pixel and skeleton renderings per
        character, and places them onto the canvas. It also updates
        the canvas scroll region.
        """
        if not self.current_mode:
            return

        self.canvas.delete("all")
        self.img_refs = []

        try:
            b_size = int(self.entry_base_size.get())
            zoom = self.scale_zoom.get()
            l_width = int(self.entry_width.get())
            wrap_limit = int(self.entry_wrap_cnt.get())
        except Exception:
            return

        content = self.text_area.get("1.0", tk.END)
        lines = content.splitlines()

        brush = get_circle_brush(l_width)
        gap = int(b_size * zoom * 0.1)
        cw, ch = b_size * zoom + gap, int(b_size * zoom * 1.3)

        current_row = 0
        for line in lines:
            if not line:
                # Preserve empty manual lines.
                current_row += 1
                continue

            for i, char in enumerate(line):
                # Compute automatic wrap position within a manual line.
                line_row_offset, col = divmod(i, wrap_limit)

                # Final display row = manual row + wrapped offset.
                display_row = current_row + line_row_offset
                x, y = 20 + col * cw, 20 + display_row * ch

                if self.current_mode == "JSON":
                    strokes = self.font_data_json.get("U+%04X" % ord(char), self.font_data_json.get("U+%X" % ord(char), []))
                else:
                    strokes = self.font_data_tyf.get_strokes(ord(char))

                if not strokes:
                    # Draw a placeholder box only for missing glyphs (not space).
                    if char != " ":
                        self.canvas.create_rectangle(x, y, x + b_size * zoom, y + b_size * zoom, outline="#333")
                    continue

                if self.show_pixel.get():
                    px_set = set()
                    for s in strokes:
                        pts = [(int(p[0] * b_size), int(p[1] * b_size)) for p in s]
                        for j in range(len(pts) - 1):
                            px_set.update(get_line_pixels(pts[j], pts[j + 1], brush, b_size))

                    if px_set:
                        tmp = tk.PhotoImage(width=b_size, height=b_size)
                        for px, py in px_set:
                            tmp.put("#EEE", (px, py))
                        f_img = tmp.zoom(zoom)
                        self.canvas.create_image(x, y, image=f_img, anchor="nw")
                        self.img_refs.append(f_img)

                if self.show_skel.get():
                    for s in strokes:
                        pts = []
                        for p in s:
                            pts.extend([x + p[0] * b_size * zoom + zoom / 2, y + p[1] * b_size * zoom + zoom / 2])
                        if len(pts) >= 4:
                            self.canvas.create_line(pts, fill="#FF2222", width=1.5, capstyle="round")

            # Advance current_row by number of auto-wrapped rows for this manual line.
            chars_in_line = len(line)
            auto_wrapped_rows = (chars_in_line - 1) // wrap_limit + 1 if chars_in_line > 0 else 1
            current_row += auto_wrapped_rows

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    FontViewer(root)
    root.mainloop()