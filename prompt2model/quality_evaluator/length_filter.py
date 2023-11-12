from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("QualityEvaluator")

def length_filter(strings, min_length=120):
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
        logger.info("Input strings in length_filter is None.")
        return None
    
    if not isinstance(min_length, int) or min_length <= 0:
        logger.info("Invalid min_length value in length_filter. It should be a positive integer.")
        return None

    try:
        filtered_strings = [string for string in strings if len(string) >= min_length]
        if len(filtered_strings)==0:
            logger.info("length_filter filtered result is empty.")
            return None
        return filtered_strings
    
    except Exception as e:
        logger.warning(f"Error in length_filter: {e}")
        return None
