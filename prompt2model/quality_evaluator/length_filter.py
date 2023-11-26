import random

from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("QualityEvaluator")


def min_max_length_filter(strings, min_length=120, max_length=None):
    """
    The function filters and returns strings from the input list that are longer than the specified minimum length.

    Parameters:
    strings (list of str): A list of strings to be filtered based on length.
    min_length (int): The minimum length a string must have to be included in the output list.

    Returns:
    list: A list of strings from the input that are longer than or equal to the specified minimum length.
          Returns None if no strings meet the criteria.

    Note:
    If the input list is None or the min_length is not a valid integer, the function logs an appropriate message
    and returns None.
    """

    if strings is None:
        logger.info("Input strings in min_max_length_filter is None.")
        return None

    if not isinstance(min_length, int) or min_length <= 0:
        logger.info(
            "Invalid min_length value in min_max_length_filter. It should be a positive integer."
        )
        return None

    try:
        if max_length is not None:
            filtered_strings = [
                string for string in strings if max_length > len(string) >= min_length
            ]
        else:
            filtered_strings = [
                string for string in strings if len(string) >= min_length
            ]
        if len(filtered_strings) == 0:
            logger.info("min_max_length_filter filtered result is empty.")
            return None
        else:
            return filtered_strings

    except Exception as e:
        logger.warning(f"Error in min_max_length_filter: {e}")
        return None


def get_middle_portion(strings, portion=0.5):
    # Check if strings is a list
    if not isinstance(strings, list):
        raise ValueError("The strings argument must be a list.")

    # Check if portion is a valid number between 0 and 1
    if not (0 <= portion <= 1):
        raise ValueError("The portion must be a number between 0 and 1.")

    # Handle empty list
    if not strings:
        return []

    # Sort the list by the length of strings
    sorted_strings = sorted(strings, key=len)

    # Handle portion being 0 or 1
    if portion == 0:
        return []
    elif portion == 1:
        return sorted_strings

    # Calculate start and end indices for the middle portion
    start_index = int(len(sorted_strings) * ((1 - portion) / 2))
    end_index = int(len(sorted_strings) * (1 - ((1 - portion) / 2)))

    # Get the middle portion
    middle_portion = sorted_strings[start_index:end_index]
    random.shuffle(middle_portion)
    return middle_portion
