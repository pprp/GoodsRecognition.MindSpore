




def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def str2bool(str_):
    """
    Args:
        str_: string

    Returns:
        bool
    """
    result = False
    if str_.lower() == "true":
        result = True
    return result