def apply_and_track_filter(inputs, data, filter_function, reason):
    """Applies a filter function to inputs, tracking reasons for any dropped inputs."""
    invalid_indices = []
    for _, (index, input_element) in enumerate(inputs):
        if not filter_function([input_element]):
            data['drop_reason'][index] = reason
            invalid_indices.append(index)
    filtered_new_inputs_with_idx = [(index, input_element) for  _, (index, input_element) in enumerate(inputs) if index not in invalid_indices]
    return filtered_new_inputs_with_idx, data
