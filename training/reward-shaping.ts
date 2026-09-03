export function approachPotential(distance: number, distanceNormalizer: number): number {
	if (!Number.isFinite(distance) || distanceNormalizer <= 0) return 0;
	return 1 - Math.max(0, Math.min(distance / distanceNormalizer, 1));
}

export function potentialBasedApproachReward(
	previousDistance: number,
	currentDistance: number,
	distanceNormalizer: number,
	gamma: number,
	scale: number,
	terminal = false
): number {
	const previousPotential = approachPotential(previousDistance, distanceNormalizer);
	const currentPotential = terminal ? 0 : approachPotential(currentDistance, distanceNormalizer);
	return scale * (gamma * currentPotential - previousPotential);
}
