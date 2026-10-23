import sys
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from interview import _selftest_questions  # noqa: E402
from kb_lib import get_card_by_id, load_cards  # noqa: E402


class SelftestQuestionParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = load_cards()

    def test_qa_format_only_extracts_questions(self):
        card = get_card_by_id("position-encoding-rope", self.cards)
        questions = _selftest_questions(card)

        self.assertEqual(len(questions), 6)
        self.assertEqual(questions[0], "RoPE 是绝对位置编码还是相对位置编码？")
        self.assertTrue(all(not q.startswith(("A：", "A:")) for q in questions))

    def test_numbered_format_is_still_supported(self):
        card = get_card_by_id("position-encoding-alibi-ntk", self.cards)
        questions = _selftest_questions(card)

        self.assertEqual(len(questions), 5)
        self.assertTrue(questions[0].startswith("为什么 ALiBi"))


if __name__ == "__main__":
    unittest.main()
