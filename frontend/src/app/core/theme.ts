import { DOCUMENT, Injectable, effect, inject, signal } from '@angular/core';

export type Theme = 'light' | 'dark';
const KEY = 'theme';

/** Follows the system theme until the user picks one with the header toggle; the pick is kept in localStorage. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly doc = inject(DOCUMENT);
  readonly choice = signal<Theme | null>(read());

  constructor() {
    effect(() => {
      const t = this.choice();
      const root = this.doc.documentElement;
      if (t) root.setAttribute('data-theme', t);
      else root.removeAttribute('data-theme');
    });
  }

  toggle(): void {
    const next: Theme = this.effective() === 'dark' ? 'light' : 'dark';
    this.choice.set(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage blocked: the choice still applies to this page */
    }
  }

  effective(): Theme {
    const t = this.choice();
    if (t) return t;
    const win = this.doc.defaultView;
    return win?.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
}

function read(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch {
    return null;
  }
}
