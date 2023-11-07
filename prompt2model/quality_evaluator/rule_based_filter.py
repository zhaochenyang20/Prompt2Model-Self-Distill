def rule_based_filter(strings, pattern_type):
    """
    Filters a list of strings based on a specified pattern type.

    This function searches for strings that start with a specific pattern,
    depending on the `pattern_type` specified.
    If the pattern type does not match these two options, the function raises a ValueError.

    Parameters:
    strings (list of str): A list of strings to be filtered.
    pattern_type (str): A string that specifies the type of pattern to filter by.
                        Accepts either 'input' or 'output'.

    Returns:
    list of str: A list containing strings that start with the specified pattern.

    Raises:
    ValueError: If `pattern_type` is not 'input' or 'output'.
    """

    if pattern_type == "input":
        pattern = "[input]"
    elif pattern_type == "output":
        pattern = "[output]"
    else:
        raise ValueError("Invalid pattern type specified. Choose 'input' or 'output'.")

    matching_strings = []

    for s in strings:
        if s.startswith(pattern):
            matching_strings.append(s)

    return matching_strings
