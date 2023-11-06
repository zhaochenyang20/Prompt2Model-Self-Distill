greetings = [
    "Sure, I'd be happy to help!",
    "Of course!",
    "Let me check on that.",
    "I'm here to help.",
    "Great question!",
    "Let me see.",
    "That's a good question.",
    "Thanks for asking.",
    "Let me help you with that.",
    "No problem, here's the answer.",
    "I'm glad you asked.",
    "Alright, let me explain.",
    "Absolutely!",
    "I understand your question."
]

def ablation_list_filter(strings):
    """
    Filter out strings that contain any of the predefined greetings.

    Parameters:
    strings (list of str): A list of strings to be filtered.

    Returns:
    list of str: A list of strings that do not contain any of the greetings.
    """


    filtered_strings = []

    for s in strings:
        if not any(greeting in s for greeting in greetings):
            filtered_strings.append(s)
    
    return filtered_strings
