from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("QualityEvaluator")
def empty_filter(strings):
    """
    Filter out strings that is None or empty.

    Parameters:
    strings (list of str): A list of strings to be filtered.

    Returns:
    list of str: A list of strings that is not None and not empty.
    """
    try:
        if strings is None:
            logger.info("strings passed in empty_filter is None.")
            return None

        if len(strings) == 0:
            logger.info("strings passed in empty_filter is empty.")
            return None

        filtered_strings = []
        for s in strings:
            if s is not None and s != '':
                filtered_strings.append(s)

        if len(filtered_strings) == 0:
            logger.info("empty_filter filtered result is empty.")
            return None
        else:
            return filtered_strings

    except Exception as e:
        logger.warning(f"Error in ablation_list_filter: {e}")
        return None
