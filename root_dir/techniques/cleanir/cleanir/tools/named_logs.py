def named_logs(model, *logs):
    result = {}
    logs = [item for item in logs]
    for l in zip(model.metrics_names, [logs]):
        result[l[0]] = l[1]
    result = result['loss'][0]
    return result