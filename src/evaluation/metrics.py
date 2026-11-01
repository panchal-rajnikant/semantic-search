def hit_at_k(
    results,
    relevant_sources
) -> int:

    return int(
        any(
            result.source in relevant_sources
            for result in results
        )
    )