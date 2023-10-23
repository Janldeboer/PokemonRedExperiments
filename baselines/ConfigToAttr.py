

# applying a dict as attributes to an object

def apply_dict_as_attributes(target_object, attributes_dict):
    for key, value in attributes_dict.items():
        setattr(target_object, key, value)
