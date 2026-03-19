import unittest

from bridge.model_feed import _extract_openai_text


class ModelFeedHelpersTest(unittest.TestCase):
    def test_extract_openai_text_from_string(self):
        self.assertEqual(_extract_openai_text("hello"), "hello")

    def test_extract_openai_text_from_structured_content(self):
        content = [
            {"type": "text", "text": "hello "},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            {"type": "text", "text": "world"},
        ]
        self.assertEqual(_extract_openai_text(content), "hello world")


if __name__ == "__main__":
    unittest.main()