export type HealthUiState = "checking" | "connected" | "unavailable";

export function healthUiState(health: {
  isLoading: boolean;
  isError: boolean;
}): HealthUiState {
  if (health.isLoading) return "checking";
  if (health.isError) return "unavailable";
  return "connected";
}
