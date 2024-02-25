def apply_and_track_filter(inputs, data, filter_function, reason):
    """Applies a filter function to inputs, tracking reasons for any dropped inputs."""
    invalid_indices = []
    for index, input_element in inputs:
        if filter_function([input_element]):
            data['drop_reason'][index] = reason
            invalid_indices.append(index)
    filtered_new_inputs_with_idx = [(idx, item) for idx, item in enumerate(inputs) if idx not in invalid_indices]
    return filtered_new_inputs_with_idx, data
