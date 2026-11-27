import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, input, signal } from '@angular/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';

import { seconds } from '../../core/format';

/**
 * Answer in progress. There is no streaming yet, so it shows elapsed time and generic progress, and names the
 * agents that may take part (the same names the trace uses); the real steps appear when the answer arrives.
 */
@Component({
  selector: 'app-pending',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatProgressBarModule],
  templateUrl: './pending.html',
  styleUrl: './pending.scss',
})
export class Pending {
  readonly startedAt = input.required<number>();
  private readonly now = signal(Date.now());
  protected readonly elapsed = computed(() => seconds(Math.max(0, this.now() - this.startedAt())));

  constructor() {
    const timer = setInterval(() => this.now.set(Date.now()), 100);
    inject(DestroyRef).onDestroy(() => clearInterval(timer));
  }
}
