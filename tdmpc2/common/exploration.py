def schedule_progress(step, start, steps):
	"""Return clamped progress through an exploration schedule."""
	return min(max((step - start) / steps, 0.), 1.)


def exploration_coefficient(step, schedule, peak, start, steps, peak_fraction):
	"""Compute a planning-only exploration coefficient."""
	if schedule == 'constant':
		return peak if step >= start else 0.

	progress = schedule_progress(step, start, steps)
	if schedule == 'linear_decay':
		return peak * (1 - progress) if step >= start else 0.

	if step <= start or progress >= 1:
		return 0.
	if progress <= peak_fraction:
		return peak * progress / peak_fraction
	return peak * (1 - progress) / (1 - peak_fraction)
