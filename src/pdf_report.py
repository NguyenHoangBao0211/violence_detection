"""Generate downloadable Vietnamese reports using the existing Matplotlib stack."""
from html.parser import HTMLParser
from io import BytesIO
from textwrap import wrap
import threading

_LOCK = threading.Lock()


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.capture = None
        self.parts = []
        self.row = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("h1", "h2", "p", "th", "td"):
            self.capture = tag
            self.parts = []

    def handle_data(self, text):
        if self.capture:
            self.parts.append(text)

    def handle_endtag(self, tag):
        if tag == self.capture:
            text = " ".join("".join(self.parts).split())
            if tag in ("th", "td") and self.row is not None:
                self.row.append(text)
            else:
                self.blocks.append((tag, text))
            self.capture = None
        if tag == "tr" and self.row is not None:
            self.blocks.append(("row", " | ".join(self.row)))
            self.row = None


def make_report_pdf(html: str) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, findfont
    import matplotlib as mpl

    parser = ReportParser()
    parser.feed(html)
    if not parser.blocks:
        raise ValueError("Báo cáo không có nội dung")
    font = FontProperties(fname=findfont("DejaVu Sans"))
    output = BytesIO()
    with _LOCK, mpl.rc_context({"pdf.fonttype": 42}), PdfPages(output, metadata={"Title": "Báo cáo phát hiện bạo lực"}) as pdf:
        fig = None
        y = 0
        page = 0

        def new_page():
            nonlocal fig, y, page
            if fig is not None:
                pdf.savefig(fig)
                fig.clear()
            page += 1
            fig = Figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(.08, .96, "VIOLENCE DETECTION - BÁO CÁO", fontproperties=font, fontsize=9, color="#64748b")
            fig.text(.92, .035, f"Trang {page}", ha="right", fontproperties=font, fontsize=9, color="#64748b")
            y = .91

        new_page()
        for kind, text in parser.blocks:
            size = 17 if kind == "h1" else 13 if kind == "h2" else 10
            width = 52 if kind == "h1" else 70 if kind == "h2" else 94
            lines = wrap(text, width=width, break_long_words=True) or [""]
            line_height = .025 if kind == "h1" else .020 if kind == "h2" else .018
            if y - len(lines) * line_height < .075:
                new_page()
            for line in lines:
                if y < .075:
                    new_page()
                fig.text(.08, y, line, va="top", fontproperties=font, fontsize=size,
                         color="#0f172a" if kind != "p" else "#475569")
                y -= line_height
            y -= .013 if kind in ("h1", "h2", "p") else .007
        pdf.savefig(fig)
        fig.clear()
    return output.getvalue()
