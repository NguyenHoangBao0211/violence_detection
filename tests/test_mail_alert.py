import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from src.mail_alert import send_violence_alert


class MailAlertTests(unittest.TestCase):
    def test_sources_and_single_send(self):
        cfg = dict(to="recipient@example.com", host="smtp.example.com", port=587,
                   user="sender@example.com", password="test")
        for source, expected in [("upload", "video tải lên"), ("webcam", "webcam")]:
            with self.subTest(source=source), patch("src.mail_alert._cfg", return_value=cfg), patch("src.mail_alert.smtplib.SMTP") as smtp:
                result = send_violence_alert(note="Video test", peak_prob=.9, history_id=12, source=source)
                self.assertTrue(result["sent"])
                connection = smtp.return_value.__enter__.return_value
                connection.send_message.assert_called_once()
                message = connection.send_message.call_args.args[0]
                self.assertIn(expected, message["Subject"])
                self.assertIn("#12", message.get_content())
                self.assertIn("90.0%", message.get_content())
                connection.starttls.assert_called_once()

    def test_missing_config_does_not_connect(self):
        with patch("src.mail_alert._cfg", return_value=dict(to="", user="", password="")), patch("src.mail_alert.smtplib.SMTP") as smtp:
            self.assertFalse(send_violence_alert(note="", peak_prob=.9, history_id=1)["sent"])
            smtp.assert_not_called()

    def test_bad_config_and_smtp_failure_are_nonfatal(self):
        with patch("src.mail_alert._cfg", side_effect=ValueError("invalid port")):
            self.assertFalse(send_violence_alert(note="", peak_prob=.9, history_id=1)["sent"])
        with patch("src.mail_alert._cfg", return_value=dict(to="r@example.com", host="smtp", port=587, user="u", password="secret")), patch("src.mail_alert.smtplib.SMTP", side_effect=OSError("offline")):
            result = send_violence_alert(note="", peak_prob=.9, history_id=1)
            self.assertFalse(result["sent"])
            self.assertNotIn("secret", result["error"])

    def test_backend_syntax(self):
        root = Path(__file__).resolve().parents[1]
        ast.parse((root / "src/api.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
