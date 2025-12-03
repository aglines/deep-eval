from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
import glob

# Find all unique prefixes (001-, 002-, etc.)
input_files = glob.glob("input/*-en.txt")
prefixes = [os.path.basename(f).split('-')[0] for f in input_files]
prefixes.sort()

# Define custom translation quality criteria for each aspect
accuracy_metric = GEval(
    name="Accuracy",
    criteria="Evaluate the accuracy: Does it preserve the meaning of the source?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)

fluency_metric = GEval(
    name="Fluency", 
    criteria="Evaluate the fluency: Is it natural in the target language?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)

completeness_metric = GEval(
    name="Completeness",
    criteria="Evaluate the completeness: Are all details translated?",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
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

    # Evaluate with all three metrics
    accuracy_metric.measure(test_case)
    fluency_metric.measure(test_case)
    completeness_metric.measure(test_case)
    
    print(f"Accuracy Score: {accuracy_metric.score}")
    print(f"Fluency Score: {fluency_metric.score}")
    print(f"Completeness Score: {completeness_metric.score}")
    print(f"Accuracy Reason: {accuracy_metric.reason}")
    print(f"Fluency Reason: {fluency_metric.reason}")
    print(f"Completeness Reason: {completeness_metric.reason}")