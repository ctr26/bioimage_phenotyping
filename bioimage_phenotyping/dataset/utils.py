def grouped_median(df, group="ObjectNumber"):
    return df.groupby(level=drop_from_index(df, group)).median()


def drop_from_index(df, item):
    return drop_from_list(list(df.index.names), item)


def drop_from_list(list_in, item):
    item = [item] if isinstance(item, str) else item
    return list(set(list_in) - set(item))

