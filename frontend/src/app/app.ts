import {
  ChangeDetectionStrategy, Component, ElementRef, afterRenderEffect, inject, untracked, viewChild,
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

import { AskFacade } from './core/ask-facade';
import { ThemeService } from './core/theme';
import { Answer } from './ui/answer/answer';
import { Composer } from './ui/composer/composer';
import { Pending } from './ui/pending/pending';
import { StartScreen } from './ui/start-screen/start-screen';

@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatButtonModule, Answer, Composer, Pending, StartScreen],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected readonly facade = inject(AskFacade);
  protected readonly theme = inject(ThemeService);
  private readonly conversation = viewChild<ElementRef<HTMLElement>>('conversation');
  private lastCount = 0;

  constructor() {
    this.facade.checkHealth();
    // Bring the newest turn into view when one is added.
    afterRenderEffect(() => {
      const n = this.facade.turns().length;
      const el = this.conversation()?.nativeElement;
      untracked(() => {
        if (el && n > this.lastCount) {
          const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
          el.lastElementChild?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
        }
        this.lastCount = n;
      });
    });
  }

  protected ask(question: string): void {
    this.facade.ask(question);
  }
}
