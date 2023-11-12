from collections import Counter
from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("QualityEvaluator")

def self_consistency_filter(strings, min_frequency=0.2) -> str:
    """
    The function counts the occurrences of each string in the input list, finds the highest frequency,
    and then identifies all strings that occur with this frequency. Out of these most frequent strings,
    it then returns the shortest one by length.

    Parameters:
    strings (list of str): A list of strings to be analyzed for self-consistency.

    Returns:
    str: The shortest string among the most frequently occurring strings in the input list.
         Returns None if the input list is empty.

    Note:
    If multiple strings have the highest frequency and are of the same shortest length, the one that
    appears first in the list will be returned.
    """

    if strings is None:
        logger.info("strings passed in self_consistency_filter is None.")
        return None
    
    if len(strings)==0:
        logger.info("strings passed in self_consistency_filter is empty.")
        return None


    try:
        frequency_counter = Counter(strings)
        highest_frequency = frequency_counter.most_common(1)[0][1]
        if highest_frequency/len(strings)<min_frequency:
            logger.info("self_consistency_filter didn't pass min_frequency")
            return None

        most_common_strings = [
            string
            for string, count in frequency_counter.items()
            if count == highest_frequency
        ]

        shortest_of_most_common = min(most_common_strings, key=len)
        return shortest_of_most_common
    
    except Exception as e:
        logger.warning(f"Error in self_consistency_filter: {e}")
        return None
    