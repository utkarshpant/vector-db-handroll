from app.core.Filter import Filter


def passes_filter(meta: dict, filt: Filter) -> bool:
    """
    Check if the metadata dictionary passes filter conditions.
    """
    for key, cond in filt.root.items():
        if key not in meta:
            return False
        val = meta[key]
        if cond.eq is not None and val != cond.eq:
            return False
        if cond.ne is not None and val == cond.ne:
            return False
        if cond.gt is not None and val <= cond.gt:
            return False
        if cond.lt is not None and val >= cond.lt:
            return False
        if cond.contains is not None and cond.contains.lower() not in str(val).lower():
            return False
        if cond.gte is not None and val < cond.gte:
            return False
        if cond.lte is not None and val > cond.lte:
            return False
    return True
