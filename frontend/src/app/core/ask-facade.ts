import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';

import type { AskRequest, AskResponse } from '../api/types';

export type FacadeStatus = 'idle' | 'loading' | 'done' | 'error';

export interface AskError {
  kind: 'network' | 'invalid' | 'server' | 'cancelled';
  message: string;
  httpStatus?: number;
}

export interface Turn {
  id: number;
  question: string;
  startedAt: number;               // Date.now() when sent
  response?: AskResponse;
  error?: AskError;
}

export const MAX_QUESTION = 1000;  // AskRequest.question max_length in the API

/**
 * The only state holder of the UI: the conversation (in memory, lost on reload) and the request status.
 * One question at a time; a new question while one is in flight is ignored.
 */
@Injectable({ providedIn: 'root' })
export class AskFacade {
  private readonly http = inject(HttpClient);
  private inFlight?: Subscription;
  private nextId = 1;

  private readonly _turns = signal<Turn[]>([]);
  private readonly _status = signal<FacadeStatus>('idle');
  private readonly _online = signal<boolean | null>(null);

  readonly turns = this._turns.asReadonly();
  readonly status = this._status.asReadonly();
  /** null until the first health check answers. */
  readonly online = this._online.asReadonly();
  readonly loading = computed(() => this._status() === 'loading');
  readonly empty = computed(() => this._turns().length === 0);
  /** Model id reported by the last answer, if any. */
  readonly model = computed(() => {
    const done = this._turns().filter((t) => t.response);
    return done.length ? done[done.length - 1].response!.model : null;
  });

  checkHealth(): void {
    this.http.get<{ status: string }>('/api/health').subscribe({
      next: (r) => this._online.set(r.status === 'ok'),
      error: () => this._online.set(false),
    });
  }

  ask(raw: string): void {
    const question = raw.trim();
    if (!question || this.loading()) return;
    const turn: Turn = { id: this.nextId++, question: question.slice(0, MAX_QUESTION), startedAt: Date.now() };
    this._turns.update((ts) => [...ts, turn]);
    this._status.set('loading');
    const body: AskRequest = { question: turn.question };
    this.inFlight = this.http.post<AskResponse>('/api/ask', body).subscribe({
      next: (response) => {
        this.patch(turn.id, { response });
        this._status.set('done');
        this._online.set(true);
      },
      error: (e: unknown) => {
        const error = toAskError(e);
        this.patch(turn.id, { error });
        this._status.set('error');
        if (error.kind === 'network') this._online.set(false);
      },
    });
  }

  /** Re-send the question of a failed turn (the failed turn is replaced). */
  retry(turnId: number): void {
    const turn = this._turns().find((t) => t.id === turnId);
    if (!turn || this.loading()) return;
    this._turns.update((ts) => ts.filter((t) => t.id !== turnId));
    this.ask(turn.question);
  }

  /** Stop waiting for the current answer. The server still finishes the request; its result is discarded. */
  cancel(): void {
    if (!this.loading() || !this.inFlight) return;
    this.inFlight.unsubscribe();
    const last = this._turns()[this._turns().length - 1];
    this.patch(last.id, { error: { kind: 'cancelled', message: 'Cancelaste la consulta.' } });
    this._status.set('error');
  }

  /** Start a new, empty conversation. */
  reset(): void {
    this.inFlight?.unsubscribe();
    this._turns.set([]);
    this._status.set('idle');
  }

  private patch(id: number, change: Partial<Turn>): void {
    this._turns.update((ts) => ts.map((t) => (t.id === id ? { ...t, ...change } : t)));
  }
}

export function toAskError(e: unknown): AskError {
  if (!(e instanceof HttpErrorResponse)) {
    return { kind: 'server', message: 'Ocurrió un error inesperado al procesar la respuesta.' };
  }
  const s = e.status;
  if (s === 0 || s === 502 || s === 503 || s === 504) {
    return { kind: 'network', httpStatus: s, message: 'No pude conectar con el servidor. ¿Está corriendo la API?' };
  }
  if (s === 422) {
    return { kind: 'invalid', httpStatus: s, message: `La pregunta no es válida: debe tener entre 1 y ${MAX_QUESTION} caracteres.` };
  }
  return { kind: 'server', httpStatus: s, message: `El servidor falló al responder (error ${s}).` };
}
