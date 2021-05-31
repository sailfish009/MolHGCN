import dgl


def get_aggregator(mode, from_field='m', to_field='agg_m', **kwargs):
    AGGR_TYPES = ['sum', 'mean', 'max']
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = dgl.function.sum(from_field, to_field)
        if mode == 'mean':
            aggr = dgl.function.mean(from_field, to_field)
        if mode == 'max':
            aggr = dgl.function.max(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))
    return aggr


