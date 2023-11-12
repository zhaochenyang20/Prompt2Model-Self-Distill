import unittest

from prompt2model.quality_evaluator import (
    ablation_list_filter,
    self_consistency_filter,
    semantic_filter,
)
from prompt2model.quality_evaluator.length_filter import length_filter


class TestAblationListFilter(unittest.TestCase):
    def test_filter_with_greetings(self):
        test_strings = [
            "Sure, I'd be happy to help!",
            "This is a test string.",
            "Great question!",
        ]
        expected_output = ["This is a test string."]
        self.assertEqual(ablation_list_filter(test_strings), expected_output)

    def test_filter_without_greetings(self):
        test_strings = ["This is a test string.", "Another test string."]
        expected_output = ["This is a test string.", "Another test string."]
        self.assertEqual(ablation_list_filter(test_strings), expected_output)

    def test_empty_input(self):
        test_strings = []
        self.assertIsNone(ablation_list_filter(test_strings))

    def test_none_input(self):
        self.assertIsNone(ablation_list_filter(None))

    def test_none_output(self):
        test_strings = ["Great question!", "Sure, I'd be happy to help!"]
        self.assertIsNone(ablation_list_filter(test_strings))


class TestSelfConsistencyFilter(unittest.TestCase):
    def test_most_common_shortest(self):
        test_strings = ["apple", "apple", "banana", "banana", "cat"]
        expected_output = "apple"
        self.assertEqual(self_consistency_filter(test_strings), expected_output)

    def test_single_most_common(self):
        test_strings = ["apple", "apple", "banana", "cat"]
        expected_output = "apple"
        self.assertEqual(self_consistency_filter(test_strings), expected_output)

    def test_empty_input(self):
        test_strings = []
        self.assertIsNone(self_consistency_filter(test_strings))

    def test_none_input(self):
        self.assertIsNone(self_consistency_filter(None))

    def test_all_unique(self):
        test_strings = ["apple", "banana", "cat"]
        expected_output = "cat"
        self.assertEqual(self_consistency_filter(test_strings), expected_output)


class TestCheckParagraphCoherence(unittest.TestCase):
    def test_coherent_paragraph(self):

        result = semantic_filter.check_paragraph_coherence(
            ["This is sentence one. This is sentence two."]
        )
        self.assertEqual(result, ["This is sentence one. This is sentence two."])

    def test_incoherent_paragraph(self):
        result = semantic_filter.check_paragraph_coherence(
            ["I like apple. Why you always think they are consistent?"]
        )
        self.assertIsNone(result)

    def test_empty_input(self):
        result = semantic_filter.check_paragraph_coherence([])
        self.assertIsNone(result)

    def test_none_input(self):
        result = semantic_filter.check_paragraph_coherence(None)
        self.assertIsNone(result)


class TestLengthFilter(unittest.TestCase):
    def test_normal_functionality(self):
        self.assertEqual(
            length_filter(["hello", "world", "a", "Python", "code"], 3),
            ["hello", "world", "Python", "code"],
        )

    def test_default_min_length(self):
        self.assertEqual(
            length_filter(
                [
                    "short",
                    "medium length string",
                    "this string is definitely longer than thirty characters",
                ]
            ),
            ["this string is definitely longer than thirty characters"],
        )

    def test_empty_list(self):
        self.assertIsNone(length_filter([], 3))

    def test_none_input(self):
        self.assertIsNone(length_filter(None, 3))

    def test_invalid_min_length(self):
        self.assertIsNone(length_filter(["hello", "world"], "three"))
        self.assertIsNone(length_filter(["hello", "world"], -1))

    def test_no_strings_meet_criteria(self):
        self.assertIsNone(length_filter(["a", "b", "c", "d"], 5))


if __name__ == "__main__":
    unittest.main()
