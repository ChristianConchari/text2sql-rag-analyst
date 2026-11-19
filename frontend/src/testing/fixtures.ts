/** AskResponse examples shaped like the real /api/ask output (see docs/ui-design and the phase 1 checkpoint). */
import type { AskResponse, TraceStep } from '../app/api/types';

const step = (agent: string, detail: string, outcome: TraceStep['outcome'] = 'ok', duration_ms = 400, attempt: number | null = null): TraceStep =>
  ({ agent, detail, outcome, duration_ms, attempt });

const base = {
  request_id: '5fde9187-8504-4410-965e-5ac1f6bb130d', sql: null, columns: [], rows: [], warnings: [],
  attempts: 0, citations: [], latency_ms: 3100, model: 'qwen2.5-coder:14b-8k',
} satisfies Partial<AskResponse>;

export const SQL_OK: AskResponse = {
  ...base, route: 'sql', status: 'ok', attempts: 2,
  answer: 'Se enviaron 131.720 órdenes de compra en enero de 2026.',
  sql: 'SELECT COUNT(DISTINCT Codigo) AS ordenes FROM oc_lineas LIMIT 100',
  columns: ['ordenes', 'NombreProveedor'],
  rows: [[131720, '[ENMASCARADO: persona natural]']],
  trace: [
    step('Enrutador', 'Pregunta por cifras de las órdenes de compra: ruta sql'),
    step('Generador SQL', 'Intento 1: COUNT sobre oc_lineas', 'ok', 900, 1),
    step('Verificador SQL', 'Objeta: agrega un valor de la orden en cada línea; deduplicar por orden', 'objection', 100, 1),
    step('Generador SQL', 'Intento 2: COUNT(DISTINCT) sobre oc_lineas', 'ok', 800, 2),
    step('Verificador SQL', 'Aprueba: solo lectura, tablas y columnas permitidas, LIMIT aplicado', 'ok', 100, 2),
    step('Ejecutor', '1 fila, DuckDB de solo lectura', 'ok', 50, 2),
    step('Redactor', 'Redacta la respuesta solo con las filas ya enmascaradas', 'ok', 700),
  ],
};

export const DOCS_OK: AskResponse = {
  ...base, route: 'docs', status: 'ok', attempts: 1, latency_ms: 4200,
  answer: 'Está a cargo de la Dirección de Compras y Contratación Pública [Ley 19.886 art. 19].',
  citations: [{ label: 'Ley 19.886 art. 19', norma: 'Ley 19886', articulo: '19', excerpt: 'Artículo 19.- Créase un Sistema…' }],
  trace: [
    step('Enrutador', 'Pregunta sobre la Ley 19.886 o su reglamento: ruta docs'),
    step('Recuperador', 'k=5 · 5 artículos sobre el umbral', 'ok', 600),
    step('Redactor', 'Intento 1: escribe solo con los fragmentos recuperados', 'ok', 3100, 1),
    step('Verificador de citas', 'Aprueba: art. 19 fue recuperado', 'ok', 10, 1),
  ],
};

export const OUT_OF_SCOPE: AskResponse = {
  ...base, route: 'out_of_scope', status: 'out_of_scope', latency_ms: 400,
  answer: 'Solo puedo responder sobre órdenes de compra de enero de 2026 (ChileCompra) y sobre la Ley 19.886 y su reglamento (Decreto 661).',
  trace: [
    step('Enrutador', 'Fuera del dominio de la aplicación: ruta out_of_scope'),
    step('Respuesta fija', 'Mensaje de alcance, sin herramientas ni otra llamada al modelo', 'ok', 0),
  ],
};

export const REFUSED: AskResponse = {
  ...base, route: 'refuse', status: 'refused', latency_ms: 400, answer: 'No puedo ayudar con esa solicitud.',
  trace: [
    step('Enrutador', 'Solicitud insegura: ruta refuse'),
    step('Respuesta fija', 'Rechazo genérico, sin herramientas ni otra llamada al modelo', 'ok', 0),
  ],
};
