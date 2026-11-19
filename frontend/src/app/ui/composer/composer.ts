import { ChangeDetectionStrategy, Component, ElementRef, input, model, output, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { MAX_QUESTION } from '../../core/ask-facade';

let nextId = 0;

/** Question input with a round send button; while a request runs the button becomes "Detener". */
@Component({
  selector: 'app-composer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule],
  templateUrl: './composer.html',
  styleUrl: './composer.scss',
  host: { '[class.large]': 'large()' },
})
export class Composer {
  readonly loading = input(false);
  /** Start-screen variant: taller, with the "Enter para enviar" hint. */
  readonly large = input(false);
  readonly value = model('');
  readonly send = output<string>();
  readonly stop = output<void>();

  protected readonly max = MAX_QUESTION;
  protected readonly inputId = `pregunta-${nextId++}`;
  private readonly field = viewChild.required<ElementRef<HTMLInputElement>>('field');

  focus(): void {
    this.field().nativeElement.focus();
  }

  protected submit(event: Event): void {
    event.preventDefault();
    const q = this.value().trim();
    if (this.loading() || !q) return;
    this.send.emit(q);
    this.value.set('');
  }
}
