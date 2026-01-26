from inference import is_grammatically_valid

test_cases = [
    # Repeats
    ("நான்", "நான்", False),
    ("அவன்", "போனான்", True),
    
    # Consecutive Verbs
    ("அவன் நடந்தான்", "வந்தான்", False),
    ("அவன் நடந்தான்", "பள்ளிக்கு", True),

    # Postpositions
    ("புத்தகம்", "பற்றி", False), # Standalone PP
    ("புத்தகம்", "படி", True), # 'படி' can be a verb 'read' or PP 'according to'. In our list?

    # Agreement
    ("நான் நேற்று", "வந்தேன்", True),
    ("நான் நேற்று", "வந்தான்", False),
    ("நாங்கள் நேற்று", "வந்தோம்", True),
    ("நாங்கள் நேற்று", "வந்தான்", False),
]

print("Running Validity Tests...")
for context, word, expected in test_cases:
    result = is_grammatically_valid(context, word)
    status = "✅ PASS" if result == expected else f"❌ FAIL (Expected {expected}, got {result})"
    print(f"Context: '{context}' + Word: '{word}' -> {status}")
