from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
import glob

# Find all unique prefixes (001-, 002-, etc.)
input_files = glob.glob("input/*-en.txt")
prefixes = [os.path.basename(f).split('-')[0] for f in input_files]
prefixes.sort()

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

# Process each test case
for prefix in prefixes:
    print(f"\n=== Processing test case {prefix} ===")
    
    # Import data for this test case
    with open(f"input/{prefix}-en.txt", "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(f"input/{prefix}-sv-actual.txt", "r", encoding="utf-8") as f:
        actual_output_text = f.read().strip()
    with open(f"input/{prefix}-sv-expected.txt", "r", encoding="utf-8") as f:
        expected_output_text = f.read().strip()

    # Create test case
    test_case = LLMTestCase(
        input=input_text,  # Source text
        actual_output=actual_output_text,  # Your translation
        expected_output=expected_output_text  # Reference translation
    )

    # Evaluate
    translation_quality.measure(test_case)
    print(f"Score: {translation_quality.score}")
    print(f"Reason: {translation_quality.reason}")