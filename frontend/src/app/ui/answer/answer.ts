import { Clipboard } from '@angular/cdk/clipboard';
import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatSnackBar } from '@angular/material/snack-bar';

import type { AskResponse } from '../../api/types';
import { Segment, splitCitations, sourceTitle } from '../../core/citations';
import { MASK, ROUTE_LABEL, ROUTE_VAR, cell, seconds } from '../../core/format';
import { TracePanel } from '../trace-panel/trace-panel';

export const SUGGESTIONS = [
  { label: '¿Cuántas órdenes se enviaron en enero?', question: '¿Cuántas órdenes de compra se enviaron en enero de 2026?' },
  {
    label: '¿Quién administra el sistema de compras?',
    question: '¿Quién está a cargo del sistema de información y gestión de compras del Estado?',
  },
];

/** One answer from /api/ask, rendered according to its route and status. */
@Component({
  selector: 'app-answer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TracePanel, MatButtonModule],
  templateUrl: './answer.html',
  styleUrl: './answer.scss',
})
export class Answer {
  private readonly clipboard = inject(Clipboard);
  private readonly snack = inject(MatSnackBar);

  readonly response = input.required<AskResponse>();
  /** Unique per turn, used for the source anchors. */
  readonly turnId = input.required<number>();
  readonly suggest = output<string>();

  protected readonly suggestions = SUGGESTIONS;
  protected readonly cell = cell;
  protected readonly sourceTitle = sourceTitle;
  protected readonly MASK = MASK;

  protected readonly r = computed(() => this.response());
  protected readonly routeLabel = computed(() => ROUTE_LABEL[this.r().route]);
  protected readonly routeVar = computed(() => ROUTE_VAR[this.r().route]);
  protected readonly isSqlResult = computed(() => this.r().route === 'sql' && this.r().status === 'ok');
  protected readonly isDocsResult = computed(() => this.r().route === 'docs' && this.r().status === 'ok');
  protected readonly segments = computed(() => splitCitations(this.r().answer, this.r().citations));
  protected readonly meta = computed(() => {
    const r = this.r();
    const t = seconds(r.latency_ms);
    if (r.route === 'out_of_scope' || r.route === 'refuse' || r.route === 'unknown') return `sin herramientas · ${t}`;
    const n = r.attempts;
    return `${r.status} · ${n} ${n === 1 ? 'intento' : 'intentos'} · ${t}`;
  });
  protected readonly shortId = computed(() => this.r().request_id.slice(0, 8));

  protected anchor(i: number): string {
    return `fuente-${this.turnId()}-${i + 1}`;
  }

  protected asCite(s: Segment): { cite: number; label: string } | null {
    return 'cite' in s ? s : null;
  }

  protected asText(s: Segment): string {
    return 'text' in s ? s.text : '';
  }

  protected isNumber(v: unknown): boolean {
    return typeof v === 'number';
  }

  protected copy(text: string, what: string): void {
    const ok = this.clipboard.copy(text);
    this.snack.open(ok ? `${what} copiada` : 'No se pudo copiar', undefined, { duration: 1800 });
  }
}
