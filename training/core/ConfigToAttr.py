def apply_dict_as_attributes(target_object, attributes_dict):
    """Applying all key-value pairs from a dictionary as attributes to an object.
    key: attribute name
    value: attribute value"""
    for key, value in attributes_dict.items():
        setattr(target_object, key, value)
