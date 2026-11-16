"""
Configuration settings and constants for Extractive vs Generative QA application.
"""

from typing import Dict, List


# Model Identifiers
EXTRACTIVE_MODEL_NAME: str = "distilbert/distilbert-base-cased-distilled-squad"
GENERATIVE_MODEL_NAME: str = "qwen3:8b"

# Ollama API Configuration
OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
OLLAMA_NATIVE_URL: str = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT: int = 60  # seconds

# Groq API Configuration
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"

# Extractive QA Sliding Window Parameters (to support long passages)
MAX_SEQ_LENGTH: int = 512
DOC_STRIDE: int = 128
MAX_ANSWER_LENGTH: int = 50

# Generative QA Generation Parameters
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 1024

# Inter-Model Agreement Thresholds
# NOTE: These are experimental heuristic thresholds for comparative demonstration
# and inter-model divergence detection, NOT universal NLP standards or ground-truth accuracy metrics.
F1_STRONG_AGREEMENT_THRESHOLD: float = 0.70
F1_PARTIAL_AGREEMENT_THRESHOLD: float = 0.30

# Preset Sample Test Cases
SAMPLE_PRESETS: List[Dict[str, str]] = [
    {
        "title": "Apollo 11 Moon Landing (Factual / History)",
        "passage": (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module "
            "Eagle on July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the "
            "lunar surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him "
            "19 minutes later. They spent about two and a quarter hours together outside the spacecraft, "
            "and they collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. "
            "Command module pilot Michael Collins flew the command module Columbia alone in lunar orbit "
            "while they were on the Moon's surface."
        ),
        "question": "How much lunar material did Armstrong and Aldrin collect?",
    },
    {
        "title": "James Webb Space Telescope (Science / Tech)",
        "passage": (
            "The James Webb Space Telescope (JWST) is an infrared space observatory launched on "
            "December 25, 2021, aboard an Ariane 5 rocket from Kourou, French Guiana. Named after "
            "James E. Webb, who was the administrator of NASA from 1961 to 1968, JWST was designed "
            "to conduct infrared astronomy. Its primary mirror consists of 18 hexagonal mirror segments "
            "made of gold-plated beryllium, which combine to create a 6.5-meter-diameter mirror. "
            "The observatory operates at the Sun-Earth L2 Lagrange point, approximately 1.5 million "
            "kilometers from Earth."
        ),
        "question": "Where does the James Webb Space Telescope operate?",
    },
    {
        "title": "Photosynthesis (Biology / Mechanism)",
        "passage": (
            "Photosynthesis is a biological process used by plants, algae, and certain bacteria to "
            "convert light energy into chemical energy stored in glucose. During oxygenic photosynthesis, "
            "water and carbon dioxide are consumed, and oxygen is released as a byproduct. Light is "
            "absorbed by green pigments called chlorophylls located in cellular organelles called chloroplasts."
        ),
        "question": "What pigments absorb light during photosynthesis?",
    },
    {
        "title": "Unanswerable Query Test (Boundary Evaluation)",
        "passage": (
            "The Turing Award is an annual prize given by the Association for Computing Machinery (ACM) "
            "for contributions of lasting and major technical importance to computer science. It is "
            "generally recognized as the highest distinction in computer science."
        ),
        "question": "Who won the Nobel Prize in Physics in 1921?",
    },
]
