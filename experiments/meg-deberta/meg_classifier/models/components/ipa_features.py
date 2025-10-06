"""
IPA (International Phonetic Alphabet) Feature Matrix

This module defines the phonetic feature representation for 39 ARPABET phonemes
used in multi-task learning for improved phoneme classification.
"""

import torch


def get_ipa_feature_matrix():
    """
    Returns the IPA feature matrix for 39 ARPABET phonemes.

    Each phoneme is represented by 14 binary phonetic features:
    - consonantal: Sound produced with constriction in vocal tract
    - syllabic: Can form the nucleus of a syllable
    - sonorant: Produced with continuous, non-turbulent airflow
    - voice: Produced with vocal cord vibration
    - nasal: Produced with airflow through nose
    - continuant: Produced with continuous airflow through mouth
    - labial: Produced with lips
    - coronal: Produced with tongue tip/blade
    - dorsal: Produced with tongue body
    - high: Tongue body raised
    - low: Tongue body lowered
    - back: Tongue body retracted
    - round: Lips rounded
    - diphthong: Contains vowel transition

    Returns:
        feature_matrix: Tensor of shape [39, 14] with binary features
        feature_names: List of feature names
    """
    feature_names = [
        "consonantal", "syllabic", "sonorant", "voice", "nasal", "continuant",
        "labial", "coronal", "dorsal", "high", "low", "back", "round", "diphthong"
    ]

    # Feature vectors for each phoneme
    # Format: [consonantal, syllabic, sonorant, voice, nasal, continuant,
    #          labial, coronal, dorsal, high, low, back, round, diphthong]
    rows = {
        # Vowels
        "AA": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],  # "ah" as in "father"
        "AE": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # "ae" as in "cat"
        "AH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "uh" as in "but"
        "AO": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],  # "aw" as in "bought"
        "AW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],  # "ow" as in "bout"
        "AY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # "ai" as in "bite"
        "EH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "eh" as in "bet"
        "ER": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "er" as in "bird"
        "EY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # "ay" as in "bait"
        "IH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # "ih" as in "bit"
        "IY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # "ee" as in "beat"
        "OW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],  # "oh" as in "boat"
        "OY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],  # "oy" as in "boy"
        "UH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],  # "uh" as in "book"
        "UW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],  # "oo" as in "boot"

        # Stops (plosives)
        "P":  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # voiceless bilabial stop
        "B":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # voiced bilabial stop
        "T":  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless alveolar stop
        "D":  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiced alveolar stop
        "K":  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # voiceless velar stop
        "G":  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # voiced velar stop

        # Affricates
        "CH": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless postalveolar affricate
        "JH": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiced postalveolar affricate

        # Fricatives
        "F":  [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # voiceless labiodental fricative
        "V":  [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # voiced labiodental fricative
        "TH": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless dental fricative
        "DH": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiced dental fricative
        "S":  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless alveolar fricative
        "Z":  [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiced alveolar fricative
        "SH": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless postalveolar fricative
        "ZH": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # voiced postalveolar fricative
        "HH": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # voiceless glottal fricative

        # Nasals
        "M":  [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # bilabial nasal
        "N":  [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # alveolar nasal
        "NG": [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # velar nasal

        # Approximants
        "L":  [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # lateral alveolar approximant
        "R":  [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # alveolar approximant
        "W":  [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],  # labio-velar approximant
        "Y":  [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # palatal approximant
    }

    # Order of phonemes (must match the label indices used in training)
    phoneme_order = [
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
        "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
        "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
    ]

    # Create feature matrix
    feature_matrix = torch.tensor([rows[p] for p in phoneme_order], dtype=torch.float32)

    return feature_matrix, feature_names