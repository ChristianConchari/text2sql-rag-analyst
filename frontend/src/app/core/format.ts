/** Display helpers. Numbers and durations follow es-CL: 131.720 and "0,4 s". */
import type { Route } from '../api/types';

const NUMBER = new Intl.NumberFormat('es-CL', { maximumFractionDigits: 2 });
const SECONDS = new Intl.NumberFormat('es-CL', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

/** Milliseconds as seconds with one decimal and a comma: 412 -> "0,4 s". */
export function seconds(ms: number): string {
  return `${SECONDS.format(ms / 1000)} s`;
}

/** A table cell: numbers in es-CL, null as an empty cell, everything else as text. */
export function cell(value: unknown): string {
  if (typeof value === 'number') return NUMBER.format(value);
  if (value === null || value === undefined) return '';
  return String(value);
}

export const MASK = '[ENMASCARADO: persona natural]';

export const ROUTE_LABEL: Record<Route, string> = {
  sql: 'Datos · sql',
  docs: 'Ley y reglamento · docs',
  out_of_scope: 'Fuera de alcance',
  refuse: 'Rechazado',
  unknown: 'Sin clasificar',
};

/** CSS custom property (see styles.scss) for each route's colour. */
export const ROUTE_VAR: Record<Route, string> = {
  sql: 'sql',
  docs: 'docs',
  out_of_scope: 'oos',
  refuse: 'ref',
  unknown: 'muted',
};
