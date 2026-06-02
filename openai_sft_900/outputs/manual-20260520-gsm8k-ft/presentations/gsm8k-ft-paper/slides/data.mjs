export const RESULTS = [
  { label: "base", correct: 95, incorrectPct: 0 },
  { label: "ft_01", correct: 83, incorrectPct: 1 },
  { label: "ft_05", correct: 79, incorrectPct: 5 },
  { label: "ft_10", correct: 80, incorrectPct: 10 },
  { label: "ft_20", correct: 74, incorrectPct: 20 },
  { label: "ft_50", correct: 70, incorrectPct: 50 },
  { label: "ft_100", correct: 71, incorrectPct: 100 },
];

export const ERROR_TYPES = [
  "calculation_error",
  "referencing_context_value_error",
  "referencing_previous_step_value_error",
  "confusing_formula_error",
  "counting_error",
  "missing_step",
  "adding_irrelevant_information",
  "operator_error",
  "unit_conversion_error",
];
