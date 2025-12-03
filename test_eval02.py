from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Import data from local input folder
with open("input/001-en.txt", "r", encoding="utf-8") as f:
    input_text = f.read().strip()
with open("input/001-sv-actual.txt", "r", encoding="utf-8") as f:
    actual_output_text = f.read().strip()
with open("input/001-sv-expected.txt", "r", encoding="utf-8") as f:
    expected_output_text = f.read().strip()

# Define custom translation quality criteria
translation_quality = GEval(
    name="Translation Quality",
    criteria="""Evaluate the translation based on:
    1. Accuracy: Does it preserve the meaning of the source?
    2. Fluency: Is it natural in the target language?
    3. Completeness: Are all details translated?""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,  # Source text
        LLMTestCaseParams.ACTUAL_OUTPUT,  # Your translation
        LLMTestCaseParams.EXPECTED_OUTPUT  # Reference translation (optional)
    ],
    threshold=0.7
)

# Create test case
test_case = LLMTestCase(
    input=input_text,  # Source text
    actual_output=actual_output_text,  # Your translation
    expected_output=expected_output_text  # Reference translation
)

# Evaluate
translation_quality.measure(test_case)
print(translation_quality.score, translation_quality.reason)