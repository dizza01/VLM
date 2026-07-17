from __future__ import annotations

import unittest

from gi_vqa.identifiers import (
    RecordFormatError,
    canonical_text,
    question_text,
    source_image_id,
    stable_item_id,
)


class IdentifierTests(unittest.TestCase):
    def test_source_id_precedence_and_message_question(self) -> None:
        record = {
            "img_id": "fallback",
            "metadata": {"source_img_id": "source-7", "img_id": "metadata-fallback"},
            "messages": [{"role": "user", "content": " <image>  Where is it? "}],
        }
        self.assertEqual(source_image_id(record), "source-7")
        self.assertEqual(question_text(record), "Where is it?")

    def test_stable_id_normalises_unicode_and_whitespace(self) -> None:
        first = {"img_id": "image-1", "question": "How   many polyps?"}
        second = {
            "metadata": {"source_img_id": "image-1"},
            "messages": [
                {"role": "user", "content": "<image>How many polyps？"}
            ],
        }
        self.assertEqual(canonical_text("polyps？"), "polyps?")
        self.assertEqual(stable_item_id(first), stable_item_id(second))

    def test_structured_hash_avoids_delimiter_ambiguity(self) -> None:
        left = {"img_id": "a|b", "question": "c"}
        right = {"img_id": "a", "question": "b|c"}
        self.assertNotEqual(stable_item_id(left), stable_item_id(right))

    def test_missing_required_fields_raise(self) -> None:
        with self.assertRaises(RecordFormatError):
            source_image_id({"question": "Q"})
        with self.assertRaises(RecordFormatError):
            question_text({"img_id": "1"})

    def test_identifier_length_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            stable_item_id({"img_id": "1", "question": "Q"}, length=7)


if __name__ == "__main__":
    unittest.main()
