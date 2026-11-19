import { ChangeDetectionStrategy, Component, computed, input, signal } from '@angular/core';

import type { Route, TraceStep } from '../../api/types';
import { ROUTE_VAR, seconds } from '../../core/format';

let nextId = 0;

/** "Cómo lo resolvieron los agentes": a collapsible timeline with one coloured dot per step. */
@Component({
  selector: 'app-trace-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './trace-panel.html',
  styleUrl: './trace-panel.scss',
})
export class TracePanel {
  readonly steps = input.required<TraceStep[]>();
  readonly route = input.required<Route>();
  readonly latencyMs = input.required<number>();

  protected readonly open = signal(true);
  protected readonly listId = `trace-${nextId++}`;
  protected readonly seconds = seconds;
  protected readonly summary = computed(() => {
    const n = this.steps().length;
    return `${n} ${n === 1 ? 'paso' : 'pasos'} · ${seconds(this.latencyMs())}`;
  });

  /** Dot colour: objections and errors in red, approvals in green, the router in its route colour. */
  protected dot(step: TraceStep): string {
    if (step.outcome === 'objection' || step.outcome === 'error') return 'var(--ref)';
    if (step.outcome === 'skipped') return 'var(--muted)';
    switch (step.agent) {
      case 'Enrutador':
        return `var(--${ROUTE_VAR[this.route()]})`;
      case 'Generador SQL':
        return 'var(--sql)';
      case 'Verificador SQL':
      case 'Verificador de citas':
      case 'Recuperador':
      case 'Guardia de salida':
        return 'var(--docs)';
      default:
        return 'var(--muted)';
    }
  }

  protected outcomeLabel(step: TraceStep): string {
    return { ok: 'correcto', objection: 'objeción', error: 'error', skipped: 'omitido' }[step.outcome];
  }
}
