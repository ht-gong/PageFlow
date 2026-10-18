import argparse

def enum_by_value(enum_cls, *, case_insensitive: bool = True):
    value_map = {str(m.value): m for m in enum_cls}
    if case_insensitive:
        value_map_ci = {k.lower(): v for k, v in value_map.items()}

    def convert(s: str):
        key = s.lower() if case_insensitive else s
        m = (value_map_ci.get(key) if case_insensitive else value_map.get(key))
        if m is None:
            valid = ", ".join([m.value for m in enum_cls])
            raise argparse.ArgumentTypeError(f"invalid choice: {s!r} (choose from: {valid})")
        return m

    return convert