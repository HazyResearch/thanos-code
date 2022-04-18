def categorize_value(level, value_range, type="int"):
    val = value_range[0] + level * (value_range[1] - value_range[0])

    return int(val) if type == "int" else float(val)
