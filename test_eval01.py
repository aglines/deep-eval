from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

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
    input="Hello, how are you?",  # Source text
    actual_output="Hola, ¿cómo estás?",  # Your translation
    expected_output="Hola, ¿cómo está?"  # Reference translation
)

# Evaluate
translation_quality.measure(test_case)
print(translation_quality.score, translation_quality.reason)