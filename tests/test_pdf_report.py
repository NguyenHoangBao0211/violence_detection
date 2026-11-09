import unittest
import re
from src.pdf_report import make_report_pdf, ReportParser


class PdfReportTests(unittest.TestCase):
    def test_vietnamese_report_is_pdf(self):
        pdf = make_report_pdf("<h1>Báo cáo bạo lực</h1><tr><th>Video</th><td>thử nghiệm.mp4</td></tr>")
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn(b"%%EOF", pdf)

    def test_long_report_has_multiple_pages(self):
        pdf = make_report_pdf("<h1>Webcam</h1>" + "<p>Sự kiện bạo lực đã lưu.</p>" * 120)
        self.assertGreater(len(re.findall(rb"/Type /Page\b", pdf)), 1)

    def test_empty_report_rejected(self):
        with self.assertRaises(ValueError):
            make_report_pdf("<html></html>")

    def test_markup_is_parsed_as_text(self):
        parser = ReportParser()
        parser.feed("<style>hidden</style><p>&lt;Tên video&gt;</p>")
        self.assertEqual(parser.blocks, [("p", "<Tên video>")])
