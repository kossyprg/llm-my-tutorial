# (デバッグ用)
def print_dict_contents(input_dict):
    """
    Print all keys and their corresponding values from a given dictionary.

    Args:
        input_dict (dict): The dictionary to process.
    """
    print("=" * 30)
    if not isinstance(input_dict, dict):
        print("Error: The provided input is not a dictionary.")
        return

    for key, value in input_dict.items():
        print(f"Key: {key}, Value: {value}")
        
    print("=" * 30 + "\n")